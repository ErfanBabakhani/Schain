"""Read-only status/plan and confirmed, transactional experiment extension.

This module intentionally has its own CLI so a completed PostgreSQL experiment can be
extended without using ``--restart`` or weakening normal worker compatibility checks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from ..config import load_config, save_settings
from .config import distributed_settings
from .manifest import build_distributed_manifest, immutable_manifest_mismatches
from .postgres import PostgresRepository, _json_hash, _stable_id
from .profile import validate_deep_discovery
from .schema import SCHEMA_VERSION


DEFAULT_CONFIG = "config/distributed_neon_deep_discovery.yaml"


def _repository(config: dict[str, Any]) -> PostgresRepository:
    settings = distributed_settings(config, require_credentials=True)
    assert settings.database_url is not None
    return PostgresRepository(
        settings.database_url, pool_min=settings.pool_min, pool_max=settings.pool_max,
    )


def _task_counts(connection: Any, experiment_id: str) -> dict[str, int]:
    return {
        str(row["status"]): int(row["n"])
        for row in connection.execute(
            "SELECT status,count(*) n FROM schain_tasks "
            "WHERE experiment_id=%s GROUP BY status", (experiment_id,),
        ).fetchall()
    }


def maintenance_status(repository: PostgresRepository, experiment_id: str) -> dict[str, Any]:
    """Read status without applying schema migrations or changing any row."""
    with repository._connection() as connection:
        experiment = connection.execute(
            "SELECT * FROM schain_experiments WHERE experiment_id=%s", (experiment_id,),
        ).fetchone()
        if experiment is None:
            raise KeyError(experiment_id)
        tiers = [dict(row) for row in connection.execute(
            "SELECT tier,status,count(*) n FROM schain_frontier "
            "WHERE experiment_id=%s GROUP BY tier,status ORDER BY tier,status",
            (experiment_id,),
        ).fetchall()]
        workers = [dict(row) for row in connection.execute(
            "SELECT worker_id,status,heartbeat_at FROM schain_workers "
            "WHERE experiment_id=%s AND status='active' "
            "AND heartbeat_at>=clock_timestamp()-interval '5 minutes' ORDER BY worker_id",
            (experiment_id,),
        ).fetchall()]
        leases = connection.execute(
            "SELECT "
            "(SELECT count(*) FROM schain_tasks WHERE experiment_id=%s AND status='leased' "
            " AND lease_expires_at>clock_timestamp()) task_leases,"
            "(SELECT count(*) FROM schain_graph_outbox WHERE experiment_id=%s AND status='leased' "
            " AND lease_expires_at>clock_timestamp()) outbox_leases",
            (experiment_id, experiment_id),
        ).fetchone()
        documents = connection.execute(
            "SELECT count(*) n FROM schain_documents WHERE experiment_id=%s", (experiment_id,),
        ).fetchone()
        facts = connection.execute(
            "SELECT accepted,count(*) n FROM schain_facts WHERE experiment_id=%s GROUP BY accepted",
            (experiment_id,),
        ).fetchall()
        audit_rows: list[dict[str, Any]] = []
        has_path_table = connection.execute(
            "SELECT to_regclass('public.schain_discovery_paths') name"
        ).fetchone()["name"]
        if has_path_table:
            audit_rows = [dict(row) for row in connection.execute(
                "SELECT fact_id,supplier_id,supplier_name,customer_id,customer_name,"
                "path_depth,frontier_action,details FROM schain_discovery_paths "
                "WHERE experiment_id=%s AND path_depth=2 ORDER BY fact_id,path_id",
                (experiment_id,),
            ).fetchall()]
        return {
            "experiment_id": experiment_id,
            "schema_version": int(experiment["schema_version"]),
            "status": experiment["status"],
            "max_tier": int(experiment["max_tier"]),
            "save_enabled": bool(experiment["save_enabled"]),
            "discovery_profile_version": experiment.get("discovery_profile_version"),
            "document_limit": experiment.get("document_limit"),
            "documents": int(documents["n"]),
            "facts": {str(bool(row["accepted"])): int(row["n"]) for row in facts},
            "tasks": _task_counts(connection, experiment_id),
            "frontier_by_tier_and_status": tiers,
            "active_workers": workers,
            "active_task_leases": int(leases["task_leases"]),
            "active_outbox_leases": int(leases["outbox_leases"]),
            "tier2_frontier_audit": audit_rows,
        }


def _toyota_row(connection: Any, experiment_id: str) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT * FROM schain_frontier WHERE experiment_id=%s AND tier=0 "
        "AND (entity_key='company:toyota' OR lower(trim(entity_name))='toyota') "
        "ORDER BY CASE WHEN entity_key='company:toyota' THEN 0 ELSE 1 END LIMIT 1",
        (experiment_id,),
    ).fetchone()
    if row is not None:
        return dict(row)
    # Only an already-evidenced alias may connect Toyota to a different canonical id.
    alias = connection.execute(
        "SELECT canonical_id FROM schain_entity_aliases "
        "WHERE experiment_id=%s AND mention_key='company:toyota' LIMIT 1",
        (experiment_id,),
    ).fetchone()
    if alias is None:
        return None
    row = connection.execute(
        "SELECT * FROM schain_frontier WHERE experiment_id=%s AND tier=0 AND entity_key=%s",
        (experiment_id, alias["canonical_id"]),
    ).fetchone()
    return dict(row) if row is not None else None


def extension_plan(
    repository: PostgresRepository,
    experiment_id: str,
    *,
    new_max_tier: int,
    topup_tier: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build a reproducible plan using SELECTs only."""
    profile = validate_deep_discovery(config)
    status = maintenance_status(repository, experiment_id)
    with repository._connection() as connection:
        experiment = connection.execute(
            "SELECT * FROM schain_experiments WHERE experiment_id=%s", (experiment_id,),
        ).fetchone()
        if experiment is None:
            raise KeyError(experiment_id)
        toyota = _toyota_row(connection, experiment_id)
        tier1 = [dict(row) for row in connection.execute(
            "SELECT * FROM schain_frontier WHERE experiment_id=%s AND tier BETWEEN 1 AND %s "
            "ORDER BY tier,entity_key", (experiment_id, topup_tier),
        ).fetchall()]
        reopen = [dict(row) for row in connection.execute(
            "SELECT * FROM schain_frontier WHERE experiment_id=%s AND tier>=%s AND tier<%s "
            "AND (expandable=false OR status='terminal') ORDER BY tier,entity_key",
            (experiment_id, int(experiment["max_tier"]), new_max_tier),
        ).fetchall()]
        new_manifest = build_distributed_manifest(config)
        mismatches = immutable_manifest_mismatches(dict(experiment["manifest"]), new_manifest)
        blockers = []
        if status["active_workers"]:
            blockers.append("active_workers")
        if status["active_task_leases"] or status["active_outbox_leases"]:
            blockers.append("active_leases")
        if bool(experiment["save_enabled"]) or save_settings(config)["enabled"]:
            blockers.append("save_policy_must_remain_disabled")
        if mismatches:
            blockers.append("immutable_manifest_mismatch")
        if toyota is None:
            blockers.append("toyota_tier0_not_found_or_not_evidenced")
        if new_max_tier < int(experiment["max_tier"]):
            blockers.append("new_max_tier_would_decrease_depth")
        return {
            "operation": "extend_existing_experiment",
            "read_only": True,
            "experiment_id": experiment_id,
            "schema_version_before": int(experiment["schema_version"]),
            "max_tier_before": int(experiment["max_tier"]),
            "new_max_tier": new_max_tier,
            "profile": profile,
            "document_limit": int(profile["document_limit"]),
            "toyota_recovery": (
                {"entity_key": toyota["entity_key"], "entity_name": toyota["entity_name"], "tier": 0}
                if toyota else None
            ),
            "tier_topups": [
                {"entity_key": row["entity_key"], "entity_name": row["entity_name"], "tier": int(row["tier"])}
                for row in tier1
            ],
            "terminal_nodes_to_reopen": [
                {"entity_key": row["entity_key"], "entity_name": row["entity_name"], "tier": int(row["tier"])}
                for row in reopen
            ],
            "ordering": ["toyota_tier0_recovery", "tier1_topups", "reopen_terminal_tier2"],
            "immutable_manifest_mismatches": mismatches,
            "blockers": blockers,
            "status": status,
        }


def _insert_task(
    repository: PostgresRepository,
    connection: Any,
    experiment_id: str,
    row: dict[str, Any],
    *,
    generation: str,
    task_type: str,
    priority: float,
    max_attempts: int,
) -> bool:
    task_key = f"{task_type}:{generation}:{row['entity_key']}"
    task_id = _stable_id("task", experiment_id, task_key)
    payload = {
        "generation": generation,
        "maintenance_topup": task_type == "topup_company",
        "preserve_existing": True,
    }
    inserted = connection.execute(
        "INSERT INTO schain_tasks "
        "(task_id,experiment_id,task_key,entity_key,entity_name,tier,task_type,status,"
        "priority,payload,max_attempts) VALUES (%s,%s,%s,%s,%s,%s,%s,'pending',%s,%s,%s) "
        "ON CONFLICT (experiment_id,task_key) DO NOTHING",
        (
            task_id, experiment_id, task_key, row["entity_key"], row["entity_name"],
            int(row["tier"]), task_type, priority, repository._json(payload), max_attempts,
        ),
    ).rowcount
    if inserted:
        connection.execute(
            "UPDATE schain_frontier SET status='pending',expandable=true,"
            "updated_at=clock_timestamp() "
            "WHERE experiment_id=%s AND entity_key=%s",
            (experiment_id, row["entity_key"]),
        )
        repository._event_locked(
            connection, experiment_id, "maintenance_task_scheduled", task_id=task_id,
            details={"task_type": task_type, "generation": generation, "tier": int(row["tier"])},
        )
    return bool(inserted)


def _backfill_path_audit(
    repository: PostgresRepository, connection: Any, experiment_id: str,
) -> int:
    """Add audit rows for historical accepted suppliesTo facts without changing facts."""
    inserted = 0
    rows = connection.execute(
        "SELECT f.fact_id,f.payload,f.task_id,COALESCE(t.entity_key,d.origin_entity) origin_entity,"
        "COALESCE(t.tier,f.tier,d.tier,0) origin_tier "
        "FROM schain_facts f LEFT JOIN schain_tasks t ON t.task_id=f.task_id "
        "LEFT JOIN schain_documents d ON d.experiment_id=f.experiment_id "
        "AND d.document_id=f.document_id "
        "WHERE f.experiment_id=%s AND f.accepted=true "
        "AND f.payload->>'relation'='suppliesTo' ORDER BY f.fact_id",
        (experiment_id,),
    ).fetchall()
    for row in rows:
        fact = dict(row["payload"] or {})
        supplier_id = str(fact.get("subject_id") or "")
        customer_id = str(fact.get("object_id") or "")
        origin = str(row["origin_entity"] or customer_id)
        if not supplier_id or not customer_id:
            continue
        depth = int(row["origin_tier"] or 0) + 1
        frontier = connection.execute(
            "SELECT tier,status,entity_type,expandable FROM schain_frontier "
            "WHERE experiment_id=%s AND entity_key=%s",
            (experiment_id, supplier_id),
        ).fetchone()
        origin_ids = {origin}
        origin_ids.update(str(alias["canonical_id"]) for alias in connection.execute(
            "SELECT canonical_id FROM schain_entity_aliases WHERE experiment_id=%s "
            "AND (mention_key=%s OR canonical_id=%s)",
            (experiment_id, origin, origin),
        ).fetchall())
        if supplier_id == customer_id:
            action = "self_edge_not_frontier"
        elif customer_id not in origin_ids:
            action = "accepted_relationship_other_customer"
        elif str(fact.get("subject_type") or "Company") != "Company":
            action = "accepted_non_expandable_supplier_type"
        elif frontier is None:
            action = "valid_fact_omitted_from_frontier"
        elif int(frontier["tier"]) < depth:
            action = "existing_entity_new_deeper_path"
        elif int(frontier["tier"]) == depth:
            action = "frontier_entity_at_expected_tier"
        else:
            action = "frontier_has_nonminimal_tier"
        path_id = _stable_id(
            "path", experiment_id, str(row["task_id"] or "historical"),
            str(row["fact_id"]), supplier_id, customer_id, str(depth),
        )
        inserted += connection.execute(
            "INSERT INTO schain_discovery_paths "
            "(experiment_id,path_id,task_id,fact_id,supplier_id,supplier_name,customer_id,"
            "customer_name,origin_entity,path_depth,frontier_action,details) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            (
                experiment_id, path_id, row["task_id"], row["fact_id"], supplier_id,
                fact.get("subject_name"), customer_id, fact.get("object_name"), origin,
                depth, action, repository._json({
                    "historical_backfill": True,
                    "global_tier": int(frontier["tier"]) if frontier is not None else None,
                    "frontier_status": frontier["status"] if frontier is not None else None,
                    "frontier_eligible": fact.get("frontier_eligible"),
                    "evidence": fact.get("evidence"),
                }),
            ),
        ).rowcount
    return inserted


def apply_extension(
    repository: PostgresRepository,
    experiment_id: str,
    *,
    new_max_tier: int,
    topup_tier: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Apply exactly one locked, idempotent migration; never deletes or resets work."""
    repository.migrate()
    profile = validate_deep_discovery(config)
    new_manifest = build_distributed_manifest(config)
    generation = str(profile["version"])
    operation_key = f"extend:{generation}:tier{new_max_tier}:topup{topup_tier}"
    settings = distributed_settings(config)
    with repository._connection() as connection, connection.transaction():
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
            ("schain-experiment:" + experiment_id,),
        )
        previous = connection.execute(
            "SELECT result FROM schain_maintenance_runs "
            "WHERE experiment_id=%s AND operation_key=%s",
            (experiment_id, operation_key),
        ).fetchone()
        if previous is not None:
            return {**dict(previous["result"]), "idempotent_replay": True}
        experiment = connection.execute(
            "SELECT * FROM schain_experiments WHERE experiment_id=%s FOR UPDATE",
            (experiment_id,),
        ).fetchone()
        if experiment is None:
            raise KeyError(experiment_id)
        active_workers = connection.execute(
            "SELECT count(*) n FROM schain_workers WHERE experiment_id=%s AND status='active' "
            "AND heartbeat_at>=clock_timestamp()-interval '5 minutes'",
            (experiment_id,),
        ).fetchone()["n"]
        active_leases = connection.execute(
            "SELECT "
            "(SELECT count(*) FROM schain_tasks WHERE experiment_id=%s AND status='leased' "
            " AND lease_expires_at>clock_timestamp()) + "
            "(SELECT count(*) FROM schain_graph_outbox WHERE experiment_id=%s AND status='leased' "
            " AND lease_expires_at>clock_timestamp()) n",
            (experiment_id, experiment_id),
        ).fetchone()["n"]
        if active_workers or active_leases:
            raise RuntimeError("workers must be stopped and all leases expired before extension")
        mismatches = immutable_manifest_mismatches(dict(experiment["manifest"]), new_manifest)
        if mismatches:
            raise ValueError("immutable configuration mismatch: " + ", ".join(mismatches))
        if bool(experiment["save_enabled"]) or save_settings(config)["enabled"]:
            raise ValueError("safe extension requires SAVE_TO_NEO4J=False")
        if new_max_tier < int(experiment["max_tier"]):
            raise ValueError("new_max_tier cannot reduce the existing experiment depth")
        toyota = _toyota_row(connection, experiment_id)
        if toyota is None:
            raise RuntimeError("Tier-0 Toyota identity was not found; refusing an inferred merge")
        tier1 = [dict(row) for row in connection.execute(
            "SELECT * FROM schain_frontier WHERE experiment_id=%s AND tier BETWEEN 1 AND %s "
            "ORDER BY tier,entity_key FOR UPDATE", (experiment_id, topup_tier),
        ).fetchall()]
        reopen = [dict(row) for row in connection.execute(
            "SELECT * FROM schain_frontier WHERE experiment_id=%s AND tier>=%s AND tier<%s "
            "AND (expandable=false OR status='terminal') ORDER BY tier,entity_key FOR UPDATE",
            (experiment_id, int(experiment["max_tier"]), new_max_tier),
        ).fetchall()]
        history_item = {
            "changed_at": datetime.now(timezone.utc).isoformat(),
            "manifest": dict(experiment["manifest"]),
            "manifest_hash": experiment["manifest_hash"],
            "max_tier": int(experiment["max_tier"]),
        }
        connection.execute(
            "UPDATE schain_experiments SET schema_version=%s,manifest_history=manifest_history||%s,"
            "manifest=%s,manifest_hash=%s,max_tier=%s,discovery_profile_version=%s,"
            "discovery_profile=%s,document_limit=%s,status='running',completed_at=NULL,"
            "updated_at=clock_timestamp() WHERE experiment_id=%s",
            (
                SCHEMA_VERSION, repository._json([history_item]), repository._json(new_manifest),
                _json_hash(new_manifest), new_max_tier, generation, repository._json(profile),
                int(profile["document_limit"]), experiment_id,
            ),
        )
        audit_rows = _backfill_path_audit(repository, connection, experiment_id)
        inserted = {"toyota": 0, "tier_topups": 0, "reopened": 0}
        inserted["toyota"] += _insert_task(
            repository, connection, experiment_id, toyota, generation=generation,
            task_type="topup_company", priority=3000, max_attempts=settings.max_task_attempts,
        )
        for row in tier1:
            inserted["tier_topups"] += _insert_task(
                repository, connection, experiment_id, row, generation=generation,
                task_type="topup_company", priority=2000, max_attempts=settings.max_task_attempts,
            )
        for row in reopen:
            inserted["reopened"] += _insert_task(
                repository, connection, experiment_id, row, generation=generation,
                task_type="expand_company", priority=1000, max_attempts=settings.max_task_attempts,
            )
        result = {
            "experiment_id": experiment_id,
            "operation_key": operation_key,
            "schema_version": SCHEMA_VERSION,
            "max_tier_before": int(experiment["max_tier"]),
            "max_tier_after": new_max_tier,
            "profile_version": generation,
            "document_limit": int(profile["document_limit"]),
            "scheduled": inserted,
            "historical_tasks_preserved": True,
            "historical_documents_preserved": True,
            "historical_facts_preserved": True,
            "tier_barrier_order": [0, 1, 2],
            "historical_path_audit_rows_added": audit_rows,
            "idempotent_replay": False,
        }
        connection.execute(
            "INSERT INTO schain_maintenance_runs "
            "(experiment_id,operation_key,operation,parameters,result) VALUES (%s,%s,%s,%s,%s)",
            (
                experiment_id, operation_key, "extend", repository._json({
                    "new_max_tier": new_max_tier, "topup_tier": topup_tier,
                    "profile_version": generation,
                }), repository._json(result),
            ),
        )
        repository._event_locked(
            connection, experiment_id, "experiment_extended", details=result,
        )
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m schain.distributed.maintenance")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status")
    status.add_argument("--experiment-id", required=True)
    status.add_argument("--json", action="store_true")
    extend = sub.add_parser("extend")
    extend.add_argument("--experiment-id", required=True)
    extend.add_argument("--new-max-tier", type=int, required=True)
    extend.add_argument("--topup-tier", type=int, default=1)
    mode = extend.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    extend.add_argument("--confirm-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(Path(args.config))
    config.setdefault("state", {})["experiment_id"] = args.experiment_id
    repository = _repository(config)
    try:
        if args.command == "status":
            result = maintenance_status(repository, args.experiment_id)
        elif args.plan:
            result = extension_plan(
                repository, args.experiment_id, new_max_tier=args.new_max_tier,
                topup_tier=args.topup_tier, config=config,
            )
        else:
            if args.confirm_id != args.experiment_id:
                raise SystemExit("--confirm-id must exactly match --experiment-id")
            result = apply_extension(
                repository, args.experiment_id, new_max_tier=args.new_max_tier,
                topup_tier=args.topup_tier, config=config,
            )
        print(json.dumps(result, indent=2, default=str, sort_keys=True))
        return 0
    finally:
        repository.close()


if __name__ == "__main__":
    raise SystemExit(main())
