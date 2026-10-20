"""Narrow PostgreSQL repository for distributed expansion.

All ownership-changing statements are fenced.  The repository never keeps a database
transaction open while retrieval or LLM inference is running.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from datetime import timedelta
import hashlib
import json
import logging
from typing import Any, Iterator, Sequence
import uuid

from .models import (
    IdentityConflictError,
    OutboxLease,
    StaleLeaseError,
    TaskLease,
    TaskResult,
)
from .schema import MIGRATIONS, SCHEMA_VERSION
from ..names import mention_key

logger = logging.getLogger(__name__)


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stable_id(kind: str, *parts: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "schain:" + kind + ":" + ":".join(parts)))


class PostgresRepository:
    """Authoritative shared state for one or more independent workers."""

    def __init__(
        self, database_url: str, *, pool_min: int = 1, pool_max: int = 4,
        open_pool: bool = True,
    ) -> None:
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as error:  # pragma: no cover - exercised without extra
            raise RuntimeError(
                "PostgreSQL state requires: pip install 'schain[postgres]'"
            ) from error
        self._dict_row = dict_row
        # check= is intentionally false: transient Neon startup errors are surfaced by
        # the first operation with the original psycopg exception and no secret logging.
        self.pool = ConnectionPool(
            conninfo=database_url,
            min_size=pool_min,
            max_size=pool_max,
            open=open_pool,
            kwargs={"autocommit": False, "row_factory": dict_row},
        )

    def open(self, *, wait_seconds: float = 30.0) -> None:
        self.pool.open(wait=True, timeout=wait_seconds)

    def close(self) -> None:
        self.pool.close()

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        with self.pool.connection() as connection:
            yield connection

    @staticmethod
    def _json(value: Any) -> Any:
        from psycopg.types.json import Jsonb
        return Jsonb(value)

    def migrate(self) -> int:
        """Apply additive schema migrations under a transaction-scoped advisory lock."""
        with self._connection() as connection, connection.transaction():
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended('schain-schema-migrations', 0))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schain_schema_migrations ("
                "version integer PRIMARY KEY, "
                "applied_at timestamptz NOT NULL DEFAULT clock_timestamp())"
            )
            applied = {
                int(row["version"])
                for row in connection.execute(
                    "SELECT version FROM schain_schema_migrations"
                ).fetchall()
            }
            for version in sorted(MIGRATIONS):
                if version in applied:
                    continue
                connection.execute(MIGRATIONS[version])
                connection.execute(
                    "INSERT INTO schain_schema_migrations(version) VALUES (%s)",
                    (version,),
                )
        return SCHEMA_VERSION

    def experiment_exists(self, experiment_id: str) -> bool:
        with self._connection() as connection:
            return connection.execute(
                "SELECT 1 FROM schain_experiments WHERE experiment_id=%s",
                (experiment_id,),
            ).fetchone() is not None

    def initialize_or_join_experiment(
        self,
        experiment_id: str,
        manifest: dict[str, Any],
        *,
        max_tier: int,
        save_enabled: bool,
        seeds: Sequence[dict[str, Any]],
        max_task_attempts: int,
        discovery_profile: dict[str, Any] | None = None,
        document_limit: int | None = None,
    ) -> str:
        """Create once, otherwise verify the immutable compatibility manifest."""
        manifest_hash = _json_hash(manifest)
        with self._connection() as connection, connection.transaction():
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                ("schain-experiment:" + experiment_id,),
            )
            created = bool(connection.execute(
                "INSERT INTO schain_experiments "
                "(experiment_id,schema_version,manifest,manifest_hash,max_tier,save_enabled,"
                "discovery_profile_version,discovery_profile,document_limit) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (
                    experiment_id, SCHEMA_VERSION, self._json(manifest), manifest_hash,
                    max_tier, save_enabled,
                    (discovery_profile or {}).get("version"),
                    self._json(discovery_profile or {}), document_limit,
                ),
            ).rowcount)
            row = connection.execute(
                "SELECT * FROM schain_experiments WHERE experiment_id=%s FOR UPDATE",
                (experiment_id,),
            ).fetchone()
            if int(row["schema_version"]) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"experiment schema {row['schema_version']} is incompatible with {SCHEMA_VERSION}"
                )
            if row["manifest_hash"] != manifest_hash:
                raise ValueError(
                    "experiment configuration is incompatible with its stored manifest; "
                    "use the original settings or a new experiment id"
                )
            if int(row["max_tier"]) != int(max_tier) or bool(row["save_enabled"]) != bool(save_enabled):
                raise ValueError("experiment max_tier/save policy does not match stored settings")
            stored_profile = row.get("discovery_profile_version")
            requested_profile = (discovery_profile or {}).get("version")
            if stored_profile and stored_profile != requested_profile:
                raise ValueError("experiment discovery profile does not match stored settings")
            for seed in seeds:
                self._schedule_locked(
                    connection,
                    experiment_id,
                    entity_key=str(seed["entity_key"]),
                    entity_name=str(seed["entity_name"]),
                    entity_type=str(seed.get("entity_type", "Company")),
                    tier=0,
                    discovered_from=None,
                    provenance={"kind": "seed", "group": seed.get("group")},
                    priority=float(seed.get("priority", 0)),
                    max_task_attempts=max_task_attempts,
                    max_tier=max_tier,
                )
            self._event_locked(
                connection, experiment_id, "experiment_initialized" if created else "experiment_joined",
                details={"schema_version": SCHEMA_VERSION},
            )
        return "created" if created else row["status"]

    def record_stale_completion(self, lease: TaskLease) -> None:
        """Record a fenced-out worker without changing authoritative task state."""
        with self._connection() as connection, connection.transaction():
            self._event_locked(
                connection, lease.experiment_id, "stale_worker_completion_rejected",
                worker_id=lease.worker_id, task_id=lease.task_id,
                details={"lease_token": lease.lease_token},
            )

    def register_worker(
        self,
        experiment_id: str,
        worker_id: str,
        session_id: str,
        *,
        git_commit: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._connection() as connection, connection.transaction():
            connection.execute(
                "INSERT INTO schain_workers "
                "(experiment_id,worker_id,session_id,status,git_commit,metadata) "
                "VALUES (%s,%s,%s,'active',%s,%s) "
                "ON CONFLICT (experiment_id,worker_id) DO UPDATE SET "
                "session_id=excluded.session_id,status='active',git_commit=excluded.git_commit,"
                "metadata=excluded.metadata,heartbeat_at=clock_timestamp(),stopped_at=NULL",
                (experiment_id, worker_id, session_id, git_commit, self._json(metadata or {})),
            )
            self._event_locked(
                connection, experiment_id, "worker_started", worker_id=worker_id,
                details={"session_id": session_id, "git_commit": git_commit},
            )

    def heartbeat_worker(
        self, experiment_id: str, worker_id: str, statistics: dict[str, Any] | None = None,
    ) -> bool:
        with self._connection() as connection, connection.transaction():
            cursor = connection.execute(
                "UPDATE schain_workers SET heartbeat_at=clock_timestamp(), statistics=statistics||%s "
                "WHERE experiment_id=%s AND worker_id=%s AND status='active'",
                (self._json(statistics or {}), experiment_id, worker_id),
            )
            return cursor.rowcount == 1

    def stop_worker(
        self, experiment_id: str, worker_id: str, *, failed: bool = False,
    ) -> None:
        with self._connection() as connection, connection.transaction():
            connection.execute(
                "UPDATE schain_workers SET status=%s,heartbeat_at=clock_timestamp(),"
                "stopped_at=clock_timestamp() WHERE experiment_id=%s AND worker_id=%s",
                ("failed" if failed else "stopped", experiment_id, worker_id),
            )

    def claim_task(
        self, experiment_id: str, worker_id: str, *, lease_seconds: int,
    ) -> TaskLease | None:
        """Reap expired leases and atomically claim one task at the current tier."""
        with self._connection() as connection, connection.transaction():
            expired = connection.execute(
                "UPDATE schain_tasks SET status=CASE WHEN attempts>=max_attempts THEN 'failed' "
                "ELSE 'retry' END, worker_id=NULL, lease_expires_at=NULL, "
                "available_at=clock_timestamp(),updated_at=clock_timestamp(),"
                "last_error=CASE WHEN attempts>=max_attempts THEN 'lease expired; attempts exhausted' "
                "ELSE 'lease expired; reclaimed' END "
                "WHERE experiment_id=%s AND status='leased' "
                "AND lease_expires_at < clock_timestamp() RETURNING task_id,status",
                (experiment_id,),
            ).fetchall()
            for item in expired:
                self._event_locked(
                    connection, experiment_id,
                    "task_failed" if item["status"] == "failed" else "task_reclaimed",
                    task_id=item["task_id"], details={"reason": "lease_expired"},
                )
            connection.execute(
                "UPDATE schain_frontier f SET status='failed',updated_at=clock_timestamp() "
                "FROM schain_tasks t WHERE t.experiment_id=%s AND t.status='failed' "
                "AND f.experiment_id=t.experiment_id AND f.entity_key=t.entity_key",
                (experiment_id,),
            )
            row = connection.execute(
                "WITH candidate AS ("
                " SELECT t.task_id FROM schain_tasks t "
                " WHERE t.experiment_id=%s AND t.status IN ('pending','retry') "
                " AND NOT EXISTS (SELECT 1 FROM schain_tasks failed "
                "                 WHERE failed.experiment_id=t.experiment_id AND failed.status='failed') "
                " AND NOT EXISTS (SELECT 1 FROM schain_graph_outbox failed_outbox "
                "                 WHERE failed_outbox.experiment_id=t.experiment_id "
                "                 AND failed_outbox.status='failed') "
                " AND t.available_at<=clock_timestamp() AND t.attempts<t.max_attempts "
                " AND ((SELECT e.document_limit FROM schain_experiments e "
                "       WHERE e.experiment_id=t.experiment_id) IS NULL "
                "      OR (SELECT count(*) FROM schain_documents d "
                "          WHERE d.experiment_id=t.experiment_id) < "
                "         (SELECT e.document_limit FROM schain_experiments e "
                "          WHERE e.experiment_id=t.experiment_id)) "
                " AND t.tier=(SELECT min(tier) FROM schain_tasks "
                "             WHERE experiment_id=%s AND status IN ('pending','retry','leased')) "
                " AND NOT EXISTS (SELECT 1 FROM schain_graph_outbox o "
                "                 WHERE o.experiment_id=t.experiment_id "
                "                 AND o.status IN ('pending','retry','leased') AND o.tier<t.tier) "
                " ORDER BY t.priority DESC,t.created_at,t.task_id "
                " FOR UPDATE SKIP LOCKED LIMIT 1"
                ") UPDATE schain_tasks t SET status='leased',worker_id=%s,"
                "lease_token=t.lease_token+1,attempts=t.attempts+1,"
                "lease_expires_at=clock_timestamp()+(%s * interval '1 second'),"
                "updated_at=clock_timestamp() FROM candidate c WHERE t.task_id=c.task_id "
                "RETURNING t.*",
                (experiment_id, experiment_id, worker_id, lease_seconds),
            ).fetchone()
            if row is None:
                self._refresh_experiment_status_locked(connection, experiment_id)
                return None
            connection.execute(
                "UPDATE schain_frontier SET status='leased',updated_at=clock_timestamp() "
                "WHERE experiment_id=%s AND entity_key=%s",
                (experiment_id, row["entity_key"]),
            )
            self._event_locked(
                connection, experiment_id, "task_claimed", worker_id=worker_id,
                task_id=row["task_id"],
                details={"lease_token": row["lease_token"], "tier": row["tier"]},
            )
            return TaskLease(
                task_id=row["task_id"], experiment_id=experiment_id,
                task_key=row["task_key"], worker_id=worker_id,
                lease_token=int(row["lease_token"]), tier=int(row["tier"]),
                entity_key=row["entity_key"], entity_name=row["entity_name"],
                payload=dict(row["payload"] or {}), attempt=int(row["attempts"]),
            )

    def renew_task(self, lease: TaskLease, *, lease_seconds: int) -> bool:
        with self._connection() as connection, connection.transaction():
            cursor = connection.execute(
                "UPDATE schain_tasks SET lease_expires_at=clock_timestamp()+"
                "(%s * interval '1 second'),updated_at=clock_timestamp() "
                "WHERE task_id=%s AND experiment_id=%s AND worker_id=%s "
                "AND lease_token=%s AND status='leased' AND lease_expires_at>clock_timestamp()",
                (
                    lease_seconds, lease.task_id, lease.experiment_id,
                    lease.worker_id, lease.lease_token,
                ),
            )
            if cursor.rowcount:
                self._event_locked(
                    connection, lease.experiment_id, "lease_renewed",
                    worker_id=lease.worker_id, task_id=lease.task_id,
                    details={"lease_token": lease.lease_token},
                )
            return cursor.rowcount == 1

    def _assert_task_lease(self, connection: Any, lease: TaskLease) -> None:
        row = connection.execute(
            "SELECT 1 FROM schain_tasks WHERE task_id=%s AND experiment_id=%s "
            "AND worker_id=%s AND lease_token=%s AND status='leased' "
            "AND lease_expires_at>clock_timestamp() FOR UPDATE",
            (
                lease.task_id, lease.experiment_id, lease.worker_id, lease.lease_token,
            ),
        ).fetchone()
        if row is None:
            raise StaleLeaseError(
                f"task {lease.task_id} lease {lease.lease_token} is no longer authoritative"
            )

    def save_retrieval_checkpoint(
        self, lease: TaskLease, results: list[dict[str, Any]], *,
        checkpoint_entity_key: str | None = None,
    ) -> None:
        content_hash = _json_hash(results)
        with self._connection() as connection, connection.transaction():
            self._assert_task_lease(connection, lease)
            connection.execute(
                "INSERT INTO schain_retrieval_checkpoints "
                "(experiment_id,entity_key,tier,results,content_hash) VALUES (%s,%s,%s,%s,%s) "
                "ON CONFLICT (experiment_id,entity_key,tier) DO UPDATE SET "
                "results=excluded.results,content_hash=excluded.content_hash,"
                "updated_at=clock_timestamp()",
                (
                    lease.experiment_id, checkpoint_entity_key or lease.entity_key, lease.tier,
                    self._json(results), content_hash,
                ),
            )
            self._event_locked(
                connection, lease.experiment_id, "postgres_checkpoint_committed",
                worker_id=lease.worker_id, task_id=lease.task_id,
                details={"stage": "retrieval", "items": len(results)},
            )

    def load_retrieval_checkpoint(
        self, experiment_id: str, entity_key: str, tier: int,
    ) -> list[dict[str, Any]] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT results FROM schain_retrieval_checkpoints "
                "WHERE experiment_id=%s AND entity_key=%s AND tier=%s",
                (experiment_id, entity_key, tier),
            ).fetchone()
            return list(row["results"]) if row else None

    def save_document_checkpoint(
        self,
        lease: TaskLease,
        *,
        document: dict[str, Any],
        stage: str,
        extraction: dict[str, Any] | None = None,
        resolution: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        ranks = {"retrieved": 0, "extracted": 1, "resolved": 2, "completed": 3}
        if stage not in ranks:
            raise ValueError(f"invalid document checkpoint stage {stage!r}")
        with self._connection() as connection, connection.transaction():
            self._assert_task_lease(connection, lease)
            existing = connection.execute(
                "SELECT stage FROM schain_documents WHERE experiment_id=%s AND document_id=%s FOR UPDATE",
                (lease.experiment_id, document["doc_id"]),
            ).fetchone()
            if existing and ranks[existing["stage"]] > ranks[stage]:
                return
            connection.execute(
                "INSERT INTO schain_documents "
                "(experiment_id,document_id,content_hash,task_id,origin_entity,tier,stage,"
                "document,extraction,resolution,metrics) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (experiment_id,document_id) DO UPDATE SET "
                "task_id=excluded.task_id,origin_entity=excluded.origin_entity,tier=excluded.tier,"
                "stage=excluded.stage,document=excluded.document,"
                "extraction=COALESCE(excluded.extraction,schain_documents.extraction),"
                "resolution=COALESCE(excluded.resolution,schain_documents.resolution),"
                "metrics=schain_documents.metrics||excluded.metrics,updated_at=clock_timestamp()",
                (
                    lease.experiment_id, document["doc_id"], document["content_hash"],
                    lease.task_id, lease.entity_key, lease.tier, stage,
                    self._json(document), self._json(extraction) if extraction is not None else None,
                    self._json(resolution) if resolution is not None else None,
                    self._json(metrics or {}),
                ),
            )
            if stage == "resolved" and resolution:
                for entry in (resolution.get("resolution") or {}).values():
                    entity_type = entry.get("label") or entry.get("entity_type")
                    mention = entry.get("mention")
                    if not entity_type or not mention or not entry.get("canonical_id"):
                        continue
                    self._upsert_alias_locked(connection, lease, {
                        "mention_key": mention_key(str(entity_type), str(mention)),
                        "entity_type": str(entity_type),
                        "mention": str(mention),
                        "canonical_id": str(entry["canonical_id"]),
                        "canonical_name": str(entry.get("canonical_name") or mention),
                        "method": str(entry.get("method", "deterministic")),
                        "evidence": [{"document_id": document["doc_id"]}],
                    })
            self._event_locked(
                connection, lease.experiment_id, "postgres_checkpoint_committed",
                worker_id=lease.worker_id, task_id=lease.task_id,
                details={"stage": stage, "document_id": document["doc_id"]},
            )

    def task_documents(self, lease: TaskLease) -> list[dict[str, Any]]:
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM schain_documents WHERE experiment_id=%s AND "
                "(task_id=%s OR origin_entity=%s) ORDER BY created_at",
                (lease.experiment_id, lease.task_id, lease.entity_key),
            ).fetchall()]

    def shared_aliases(self, experiment_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM schain_entity_aliases WHERE experiment_id=%s ORDER BY mention_key",
                (experiment_id,),
            ).fetchall()]

    def complete_task(self, lease: TaskLease, result: TaskResult, *, save_enabled: bool) -> None:
        """Atomically persist all authoritative effects and fence task completion."""
        with self._connection() as connection, connection.transaction():
            self._assert_task_lease(connection, lease)
            for document in result.documents:
                self._upsert_document_locked(connection, lease, document)
            for alias in result.aliases:
                self._upsert_alias_locked(connection, lease, alias)
            for accepted, rows in ((True, result.accepted_facts), (False, result.rejected_facts)):
                for fact in rows:
                    fact_id = str(
                        fact.get("fact_id") or fact.get("candidate_fact_id")
                        or _json_hash(fact)
                    )
                    document_id = str(fact.get("doc_id") or fact.get("document_id") or "")
                    if not document_id:
                        raise ValueError("fact is missing doc_id")
                    connection.execute(
                        "INSERT INTO schain_facts "
                        "(experiment_id,fact_id,document_id,task_id,tier,accepted,payload) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                        (
                            lease.experiment_id, fact_id, document_id, lease.task_id,
                            lease.tier, accepted, self._json(fact),
                        ),
                    )
            if save_enabled:
                grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for fact in result.accepted_facts:
                    grouped[str(fact.get("doc_id") or fact.get("document_id"))].append(fact)
                for document_id, facts in grouped.items():
                    outbox_id = _stable_id("outbox", lease.experiment_id, document_id)
                    payload = {"facts": facts}
                    inserted = connection.execute(
                        "INSERT INTO schain_graph_outbox "
                        "(outbox_id,experiment_id,task_id,document_id,tier,payload) "
                        "VALUES (%s,%s,%s,%s,%s,%s) "
                        "ON CONFLICT (experiment_id,document_id) DO NOTHING",
                        (
                            outbox_id, lease.experiment_id, lease.task_id, document_id,
                            lease.tier, self._json(payload),
                        ),
                    ).rowcount
                    if inserted:
                        self._event_locked(
                            connection, lease.experiment_id, "neo4j_outbox_item_created",
                            worker_id=lease.worker_id, task_id=lease.task_id,
                            details={"outbox_id": outbox_id, "document_id": document_id},
                        )
            experiment = connection.execute(
                "SELECT max_tier FROM schain_experiments WHERE experiment_id=%s",
                (lease.experiment_id,),
            ).fetchone()
            preexisting = {
                str(discovery["entity_key"]): connection.execute(
                    "SELECT tier,status FROM schain_frontier "
                    "WHERE experiment_id=%s AND entity_key=%s",
                    (lease.experiment_id, str(discovery["entity_key"])),
                ).fetchone()
                for discovery in result.discoveries
            }
            for discovery in result.discoveries:
                self._schedule_locked(
                    connection, lease.experiment_id,
                    entity_key=str(discovery["entity_key"]),
                    entity_name=str(discovery["entity_name"]),
                    entity_type=str(discovery.get("entity_type", "Company")),
                    tier=int(discovery.get("tier", lease.tier + 1)),
                    discovered_from=lease.entity_key,
                    provenance=discovery.get("provenance") or {"task_id": lease.task_id},
                    priority=float(discovery.get("priority", 0)),
                    max_task_attempts=int(discovery.get("max_attempts", 3)),
                    max_tier=int(experiment["max_tier"]),
                )
            self._record_discovery_paths_locked(
                connection, lease, result.accepted_facts, result.discoveries, preexisting,
            )
            updated = connection.execute(
                "UPDATE schain_tasks SET status='completed',result=%s,worker_id=NULL,"
                "lease_expires_at=NULL,completed_at=clock_timestamp(),updated_at=clock_timestamp() "
                "WHERE task_id=%s AND worker_id=%s AND lease_token=%s AND status='leased'",
                (
                    self._json(result.metrics), lease.task_id,
                    lease.worker_id, lease.lease_token,
                ),
            )
            if updated.rowcount != 1:
                raise StaleLeaseError("task ownership changed during completion")
            connection.execute(
                "UPDATE schain_frontier SET status='completed',updated_at=clock_timestamp() "
                "WHERE experiment_id=%s AND entity_key=%s",
                (lease.experiment_id, lease.entity_key),
            )
            self._event_locked(
                connection, lease.experiment_id, "task_completed",
                worker_id=lease.worker_id, task_id=lease.task_id,
                details={
                    "accepted_facts": len(result.accepted_facts),
                    "rejected_facts": len(result.rejected_facts),
                    "discoveries": len(result.discoveries),
                },
            )
            self._refresh_experiment_status_locked(connection, lease.experiment_id)

    def fail_task(self, lease: TaskLease, error: Exception, *, retry_delay_seconds: float = 5.0) -> None:
        with self._connection() as connection, connection.transaction():
            self._assert_task_lease(connection, lease)
            row = connection.execute(
                "SELECT attempts,max_attempts FROM schain_tasks WHERE task_id=%s FOR UPDATE",
                (lease.task_id,),
            ).fetchone()
            terminal = int(row["attempts"]) >= int(row["max_attempts"])
            connection.execute(
                "UPDATE schain_tasks SET status=%s,worker_id=NULL,lease_expires_at=NULL,"
                "available_at=clock_timestamp()+%s,updated_at=clock_timestamp(),last_error=%s "
                "WHERE task_id=%s AND worker_id=%s AND lease_token=%s",
                (
                    "failed" if terminal else "retry",
                    timedelta(seconds=0 if terminal else max(0, retry_delay_seconds)),
                    str(error)[:4000], lease.task_id, lease.worker_id, lease.lease_token,
                ),
            )
            connection.execute(
                "UPDATE schain_frontier SET status=%s,updated_at=clock_timestamp() "
                "WHERE experiment_id=%s AND entity_key=%s",
                ("failed" if terminal else "pending", lease.experiment_id, lease.entity_key),
            )
            self._event_locked(
                connection, lease.experiment_id, "task_failed",
                worker_id=lease.worker_id, task_id=lease.task_id,
                details={"terminal": terminal, "error_type": type(error).__name__},
            )
            if isinstance(error, IdentityConflictError):
                self._event_locked(
                    connection, lease.experiment_id, "entity_resolution_conflict",
                    worker_id=lease.worker_id, task_id=lease.task_id,
                    details={"error": str(error)[:500]},
                )
            self._refresh_experiment_status_locked(connection, lease.experiment_id)

    def claim_outbox(
        self, experiment_id: str, worker_id: str, *, lease_seconds: int,
    ) -> OutboxLease | None:
        with self._connection() as connection, connection.transaction():
            connection.execute(
                "UPDATE schain_graph_outbox SET status=CASE WHEN attempts>=max_attempts "
                "THEN 'failed' ELSE 'retry' END,worker_id=NULL,lease_expires_at=NULL,"
                "available_at=clock_timestamp(),updated_at=clock_timestamp() "
                "WHERE experiment_id=%s AND status='leased' AND lease_expires_at<clock_timestamp()",
                (experiment_id,),
            )
            row = connection.execute(
                "WITH candidate AS (SELECT outbox_id FROM schain_graph_outbox "
                "WHERE experiment_id=%s AND status IN ('pending','retry') "
                "AND available_at<=clock_timestamp() AND attempts<max_attempts "
                "ORDER BY tier,created_at,outbox_id FOR UPDATE SKIP LOCKED LIMIT 1) "
                "UPDATE schain_graph_outbox o SET status='leased',worker_id=%s,"
                "lease_token=o.lease_token+1,attempts=o.attempts+1,"
                "lease_expires_at=clock_timestamp()+(%s * interval '1 second'),"
                "updated_at=clock_timestamp() FROM candidate c WHERE o.outbox_id=c.outbox_id "
                "RETURNING o.*",
                (experiment_id, worker_id, lease_seconds),
            ).fetchone()
            if row is None:
                return None
            return OutboxLease(
                outbox_id=row["outbox_id"], experiment_id=experiment_id,
                worker_id=worker_id, lease_token=int(row["lease_token"]),
                document_id=row["document_id"], tier=int(row["tier"]),
                payload=dict(row["payload"] or {}),
            )

    def acknowledge_outbox(
        self, lease: OutboxLease, *, nodes: int, relationships: int,
    ) -> None:
        with self._connection() as connection, connection.transaction():
            cursor = connection.execute(
                "UPDATE schain_graph_outbox SET status='acknowledged',worker_id=NULL,"
                "lease_expires_at=NULL,acknowledged_at=clock_timestamp(),updated_at=clock_timestamp() "
                "WHERE outbox_id=%s AND experiment_id=%s AND worker_id=%s "
                "AND lease_token=%s AND status='leased' AND lease_expires_at>clock_timestamp()",
                (
                    lease.outbox_id, lease.experiment_id,
                    lease.worker_id, lease.lease_token,
                ),
            )
            if cursor.rowcount != 1:
                raise StaleLeaseError("outbox ownership changed before acknowledgement")
            self._event_locked(
                connection, lease.experiment_id, "neo4j_write_acknowledged",
                worker_id=lease.worker_id,
                details={
                    "outbox_id": lease.outbox_id, "nodes": nodes,
                    "relationships": relationships,
                },
            )
            self._refresh_experiment_status_locked(connection, lease.experiment_id)

    def retry_outbox(
        self, lease: OutboxLease, error: Exception, *, retry_delay_seconds: float = 5.0,
    ) -> None:
        with self._connection() as connection, connection.transaction():
            row = connection.execute(
                "SELECT attempts,max_attempts FROM schain_graph_outbox WHERE outbox_id=%s "
                "AND worker_id=%s AND lease_token=%s AND status='leased' "
                "AND lease_expires_at>clock_timestamp() FOR UPDATE",
                (lease.outbox_id, lease.worker_id, lease.lease_token),
            ).fetchone()
            if row is None:
                raise StaleLeaseError("outbox ownership changed before retry")
            terminal = int(row["attempts"]) >= int(row["max_attempts"])
            connection.execute(
                "UPDATE schain_graph_outbox SET status=%s,worker_id=NULL,lease_expires_at=NULL,"
                "available_at=clock_timestamp()+%s,last_error=%s,updated_at=clock_timestamp() "
                "WHERE outbox_id=%s",
                (
                    "failed" if terminal else "retry",
                    timedelta(seconds=0 if terminal else max(0, retry_delay_seconds)),
                    str(error)[:4000], lease.outbox_id,
                ),
            )
            self._event_locked(
                connection, lease.experiment_id, "graph_write_retried",
                worker_id=lease.worker_id,
                details={"outbox_id": lease.outbox_id, "terminal": terminal},
            )
            self._refresh_experiment_status_locked(connection, lease.experiment_id)

    def progress(self, experiment_id: str, *, active_within_seconds: int = 120) -> dict[str, Any]:
        with self._connection() as connection:
            experiment = connection.execute(
                "SELECT * FROM schain_experiments WHERE experiment_id=%s", (experiment_id,),
            ).fetchone()
            if experiment is None:
                raise KeyError(experiment_id)
            task_counts = {
                row["status"]: int(row["n"])
                for row in connection.execute(
                    "SELECT status,count(*) n FROM schain_tasks WHERE experiment_id=%s GROUP BY status",
                    (experiment_id,),
                ).fetchall()
            }
            task_metrics = connection.execute(
                "SELECT COALESCE(sum(attempts),0) attempts,"
                "COALESCE(sum(GREATEST(attempts-1,0)),0) retries,"
                "COALESCE(sum(COALESCE((result->'usage'->>'llm_calls')::bigint,"
                "(result->>'llm_calls')::bigint,0)),0) llm_calls "
                "FROM schain_tasks WHERE experiment_id=%s",
                (experiment_id,),
            ).fetchone()
            facts = connection.execute(
                "SELECT accepted,count(*) n FROM schain_facts WHERE experiment_id=%s GROUP BY accepted",
                (experiment_id,),
            ).fetchall()
            fact_counts = {bool(row["accepted"]): int(row["n"]) for row in facts}
            row = connection.execute(
                "SELECT count(*) FILTER (WHERE status='acknowledged') acknowledged,"
                "count(*) FILTER (WHERE status IN ('pending','retry','leased')) pending "
                "FROM schain_graph_outbox WHERE experiment_id=%s",
                (experiment_id,),
            ).fetchone()
            worker_rows = connection.execute(
                "SELECT worker_id,statistics FROM schain_workers WHERE experiment_id=%s "
                "AND status='active' AND heartbeat_at>=clock_timestamp()-(%s * interval '1 second')",
                (experiment_id, active_within_seconds),
            ).fetchall()
            frontier = connection.execute(
                "SELECT count(*) n,min(tier) FILTER (WHERE status IN ('pending','leased')) current_tier "
                "FROM schain_frontier WHERE experiment_id=%s", (experiment_id,),
            ).fetchone()
            documents = connection.execute(
                "SELECT count(*) n,count(*) FILTER "
                "(WHERE document->>'content_source'='full_text') full_text "
                "FROM schain_documents WHERE experiment_id=%s", (experiment_id,),
            ).fetchone()
            return {
                "experiment_id": experiment_id,
                "status": experiment["status"],
                "current_tier": frontier["current_tier"],
                "discovered_companies": int(frontier["n"]),
                "completed_companies": task_counts.get("completed", 0),
                "pending_tasks": task_counts.get("pending", 0) + task_counts.get("retry", 0),
                "running_tasks": task_counts.get("leased", 0),
                "failed_tasks": task_counts.get("failed", 0),
                "active_workers": len(worker_rows),
                "accepted_facts": fact_counts.get(True, 0),
                "rejected_facts": fact_counts.get(False, 0),
                "neo4j_committed_documents": int(row["acknowledged"] or 0),
                "pending_outbox_items": int(row["pending"] or 0),
                "total_llm_calls": int(task_metrics["llm_calls"] or 0),
                "task_attempts": int(task_metrics["attempts"] or 0),
                "retry_count": int(task_metrics["retries"] or 0),
                "documents": int(documents["n"] or 0),
                "full_text_documents": int(documents["full_text"] or 0),
                "document_limit": experiment.get("document_limit"),
                "discovery_profile_version": experiment.get("discovery_profile_version"),
                "progress_by_worker": {
                    item["worker_id"]: dict(item["statistics"] or {}) for item in worker_rows
                },
            }

    def export_facts(self, experiment_id: str, *, accepted: bool = True) -> list[dict[str, Any]]:
        with self._connection() as connection:
            return [dict(row["payload"]) for row in connection.execute(
                "SELECT payload FROM schain_facts WHERE experiment_id=%s AND accepted=%s "
                "ORDER BY document_id,fact_id",
                (experiment_id, accepted),
            ).fetchall()]

    def export_documents(self, experiment_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            return [dict(row["document"]) for row in connection.execute(
                "SELECT document FROM schain_documents WHERE experiment_id=%s "
                "ORDER BY created_at,document_id", (experiment_id,),
            ).fetchall()]

    def export_events(self, experiment_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            return [
                {
                    **dict(row),
                    "created_at": row["created_at"].isoformat(),
                    "details": dict(row["details"] or {}),
                }
                for row in connection.execute(
                    "SELECT event_id,worker_id,task_id,event_type,details,created_at "
                    "FROM schain_events WHERE experiment_id=%s ORDER BY event_id",
                    (experiment_id,),
                ).fetchall()
            ]

    def experiment_manifest(self, experiment_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT schema_version,manifest,status,max_tier,save_enabled,"
                "discovery_profile_version,discovery_profile,document_limit,manifest_history,created_at,"
                "updated_at,completed_at FROM schain_experiments WHERE experiment_id=%s",
                (experiment_id,),
            ).fetchone()
            if row is None:
                raise KeyError(experiment_id)
            return {
                **dict(row), "manifest": dict(row["manifest"]),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
            }

    def prune_events(self, *, retention_days: int) -> int:
        with self._connection() as connection, connection.transaction():
            cursor = connection.execute(
                "DELETE FROM schain_events WHERE created_at<clock_timestamp()-%s",
                (timedelta(days=max(1, retention_days)),),
            )
            return cursor.rowcount

    def frontier_audit(self, experiment_id: str) -> list[dict[str, Any]]:
        """Explain each accepted upstream path independently of global entity tier."""
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT path_id,task_id,fact_id,supplier_id,supplier_name,customer_id,"
                "customer_name,origin_entity,path_depth,frontier_action,details,created_at "
                "FROM schain_discovery_paths WHERE experiment_id=%s "
                "ORDER BY created_at,path_id",
                (experiment_id,),
            ).fetchall()]

    def _record_discovery_paths_locked(
        self,
        connection: Any,
        lease: TaskLease,
        facts: list[dict[str, Any]],
        discoveries: list[dict[str, Any]],
        preexisting: dict[str, Any],
    ) -> None:
        discovery_keys = {str(item["entity_key"]) for item in discoveries}
        proposed_depth = lease.tier + 1
        origin_ids = {lease.entity_key}
        origin_ids.update(str(row["canonical_id"]) for row in connection.execute(
            "SELECT canonical_id FROM schain_entity_aliases WHERE experiment_id=%s "
            "AND (mention_key=%s OR canonical_id=%s)",
            (lease.experiment_id, lease.entity_key, lease.entity_key),
        ).fetchall())
        for fact in facts:
            if fact.get("relation") != "suppliesTo":
                continue
            supplier_id = str(fact.get("subject_id") or "")
            customer_id = str(fact.get("object_id") or "")
            if not supplier_id or not customer_id:
                continue
            fact_id = str(fact.get("fact_id") or fact.get("candidate_fact_id") or _json_hash(fact))
            old = preexisting.get(supplier_id)
            if old is None:
                old = connection.execute(
                    "SELECT tier,status FROM schain_frontier "
                    "WHERE experiment_id=%s AND entity_key=%s",
                    (lease.experiment_id, supplier_id),
                ).fetchone()
            if supplier_id == customer_id:
                action = "self_edge_not_frontier"
            elif customer_id not in origin_ids:
                action = "accepted_relationship_other_customer"
            elif supplier_id in discovery_keys and preexisting.get(supplier_id) is None:
                action = "new_unique_supplier"
            elif old is not None and int(old["tier"]) < proposed_depth:
                action = "existing_entity_new_deeper_path"
            elif old is not None and int(old["tier"]) == proposed_depth:
                action = "existing_entity_new_peer_path"
            elif old is not None:
                action = "existing_entity_shorter_path"
            else:
                action = "valid_fact_omitted_from_frontier"
            path_id = _stable_id(
                "path", lease.experiment_id, lease.task_id, fact_id,
                supplier_id, customer_id, str(proposed_depth),
            )
            connection.execute(
                "INSERT INTO schain_discovery_paths "
                "(experiment_id,path_id,task_id,fact_id,supplier_id,supplier_name,"
                "customer_id,customer_name,origin_entity,path_depth,frontier_action,details) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (
                    lease.experiment_id, path_id, lease.task_id, fact_id,
                    supplier_id, fact.get("subject_name"), customer_id,
                    fact.get("object_name"), lease.entity_key, proposed_depth, action,
                    self._json({
                        "global_tier": int(old["tier"]) if old is not None else None,
                        "scheduled_by_task": supplier_id in discovery_keys,
                        "evidence": fact.get("evidence"),
                    }),
                ),
            )

    def database_size_bytes(self) -> int:
        with self._connection() as connection:
            return int(connection.execute("SELECT pg_database_size(current_database()) n").fetchone()["n"])

    def _upsert_document_locked(
        self, connection: Any, lease: TaskLease, document: dict[str, Any],
    ) -> None:
        payload = document.get("document") or document
        doc_id = str(payload.get("doc_id") or document.get("document_id"))
        content_hash = str(payload.get("content_hash") or document.get("content_hash"))
        if not doc_id or not content_hash:
            raise ValueError("document is missing doc_id/content_hash")
        connection.execute(
            "INSERT INTO schain_documents "
            "(experiment_id,document_id,content_hash,task_id,origin_entity,tier,stage,"
            "document,extraction,resolution,metrics) VALUES (%s,%s,%s,%s,%s,%s,'completed',%s,%s,%s,%s) "
            "ON CONFLICT (experiment_id,document_id) DO UPDATE SET stage='completed',"
            "document=excluded.document,extraction=COALESCE(excluded.extraction,schain_documents.extraction),"
            "resolution=COALESCE(excluded.resolution,schain_documents.resolution),"
            "metrics=schain_documents.metrics||excluded.metrics,updated_at=clock_timestamp()",
            (
                lease.experiment_id, doc_id, content_hash, lease.task_id, lease.entity_key,
                lease.tier, self._json(payload),
                self._json(document.get("extraction")) if document.get("extraction") is not None else None,
                self._json(document.get("resolution")) if document.get("resolution") is not None else None,
                self._json(document.get("metrics") or {}),
            ),
        )

    def _upsert_alias_locked(self, connection: Any, lease: TaskLease, alias: dict[str, Any]) -> None:
        key = str(alias["mention_key"])
        existing = connection.execute(
            "SELECT * FROM schain_entity_aliases WHERE experiment_id=%s AND mention_key=%s FOR UPDATE",
            (lease.experiment_id, key),
        ).fetchone()
        if existing and (
            existing["canonical_id"] != alias["canonical_id"]
            or existing["entity_type"] != alias["entity_type"]
        ):
            self._event_locked(
                connection, lease.experiment_id, "entity_resolution_conflict",
                worker_id=lease.worker_id, task_id=lease.task_id,
                details={
                    "mention_key": key, "stored_canonical_id": existing["canonical_id"],
                    "proposed_canonical_id": alias["canonical_id"],
                },
            )
            raise IdentityConflictError(f"conflicting shared entity resolution for {key}")
        connection.execute(
            "INSERT INTO schain_entity_aliases "
            "(experiment_id,mention_key,entity_type,mention,canonical_id,canonical_name,method,evidence) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (experiment_id,mention_key) "
            "DO UPDATE SET canonical_name=excluded.canonical_name,method=excluded.method,"
            "evidence=schain_entity_aliases.evidence||excluded.evidence,updated_at=clock_timestamp()",
            (
                lease.experiment_id, key, alias["entity_type"], alias["mention"],
                alias["canonical_id"], alias["canonical_name"], alias.get("method", "deterministic"),
                self._json(alias.get("evidence") or []),
            ),
        )

    def _schedule_locked(
        self,
        connection: Any,
        experiment_id: str,
        *,
        entity_key: str,
        entity_name: str,
        entity_type: str,
        tier: int,
        discovered_from: str | None,
        provenance: dict[str, Any],
        priority: float,
        max_task_attempts: int,
        max_tier: int,
    ) -> bool:
        terminal = tier >= max_tier
        inserted = connection.execute(
            "INSERT INTO schain_frontier "
            "(experiment_id,entity_key,entity_name,entity_type,tier,expandable,discovered_from,"
            "discovery_provenance,status,priority) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (experiment_id,entity_key) DO NOTHING",
            (
                experiment_id, entity_key, entity_name, entity_type, tier, not terminal,
                discovered_from, self._json([provenance]), "terminal" if terminal else "pending",
                priority,
            ),
        ).rowcount
        if not inserted:
            existing = connection.execute(
                "SELECT tier,status FROM schain_frontier WHERE experiment_id=%s AND entity_key=%s FOR UPDATE",
                (experiment_id, entity_key),
            ).fetchone()
            lowered = tier < int(existing["tier"])
            connection.execute(
                "UPDATE schain_frontier SET tier=LEAST(tier,%s),"
                "discovery_provenance=discovery_provenance||%s,priority=GREATEST(priority,%s),"
                "updated_at=clock_timestamp() WHERE experiment_id=%s AND entity_key=%s",
                (tier, self._json([provenance]), priority, experiment_id, entity_key),
            )
            if not lowered:
                self._event_locked(
                    connection, experiment_id, "duplicate_task_avoided",
                    details={"entity_key": entity_key, "tier": tier},
                )
                return False
        if terminal:
            return bool(inserted)
        task_key = "expand:" + entity_key
        task_id = _stable_id("task", experiment_id, task_key)
        task_inserted = connection.execute(
            "INSERT INTO schain_tasks "
            "(task_id,experiment_id,task_key,entity_key,entity_name,tier,priority,max_attempts) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (experiment_id,task_key) DO NOTHING",
            (
                task_id, experiment_id, task_key, entity_key, entity_name, tier,
                priority, max_task_attempts,
            ),
        ).rowcount
        if not task_inserted and not inserted:
            # A shorter path may make an already completed entity eligible again.
            connection.execute(
                "UPDATE schain_tasks SET tier=LEAST(tier,%s),priority=GREATEST(priority,%s),"
                "status=CASE WHEN tier>%s AND status='completed' THEN 'pending' ELSE status END,"
                "completed_at=CASE WHEN tier>%s THEN NULL ELSE completed_at END,updated_at=clock_timestamp() "
                "WHERE experiment_id=%s AND task_key=%s",
                (tier, priority, tier, tier, experiment_id, task_key),
            )
        if task_inserted:
            self._event_locked(
                connection, experiment_id, "frontier_item_scheduled",
                task_id=task_id,
                details={"entity_key": entity_key, "tier": tier, "discovered_from": discovered_from},
            )
        return bool(task_inserted)

    def _refresh_experiment_status_locked(self, connection: Any, experiment_id: str) -> str:
        previous = connection.execute(
            "SELECT status FROM schain_experiments WHERE experiment_id=%s FOR UPDATE",
            (experiment_id,),
        ).fetchone()["status"]
        failed = connection.execute(
            "SELECT EXISTS(SELECT 1 FROM schain_tasks WHERE experiment_id=%s AND status='failed') "
            "OR EXISTS(SELECT 1 FROM schain_graph_outbox WHERE experiment_id=%s AND status='failed') value",
            (experiment_id, experiment_id),
        ).fetchone()["value"]
        active = connection.execute(
            "SELECT EXISTS(SELECT 1 FROM schain_tasks WHERE experiment_id=%s "
            "AND status IN ('pending','retry','leased')) OR EXISTS(SELECT 1 FROM schain_graph_outbox "
            "WHERE experiment_id=%s AND status IN ('pending','retry','leased')) value",
            (experiment_id, experiment_id),
        ).fetchone()["value"]
        status = "failed" if failed else "running" if active else "completed"
        connection.execute(
            "UPDATE schain_experiments SET status=%s,updated_at=clock_timestamp(),"
            "completed_at=CASE WHEN %s='completed' THEN clock_timestamp() ELSE NULL END "
            "WHERE experiment_id=%s",
            (status, status, experiment_id),
        )
        if status == "completed" and previous != "completed":
            self._event_locked(connection, experiment_id, "experiment_completed")
        return status

    def _event_locked(
        self,
        connection: Any,
        experiment_id: str,
        event_type: str,
        *,
        worker_id: str | None = None,
        task_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        connection.execute(
            "INSERT INTO schain_events(experiment_id,worker_id,task_id,event_type,details) "
            "VALUES (%s,%s,%s,%s,%s)",
            (experiment_id, worker_id, task_id, event_type, self._json(details or {})),
        )
