"""SQLite checkpoint state. Persistence only - no pipeline logic lives here.

Resume is the default, and the rule it implements is: no expensive operation reruns
just because a later one crashed. That means each expensive stage records its own
completion.

Document status flow (INGEST and EXPAND both use it):

    PENDING -> PROCESSING -> EXTRACTED -> RESOLVED -> COMMITTED
                          \\-> FAILED

If Prompt 1 succeeded and the graph write failed, the document sits at RESOLVED; the
resumed run rebuilds the graph rows from the facts Parquet instead of calling the LLM
again. If retrieval succeeded and extraction failed, the search rows are already stored,
so retrieval is not repeated either.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .names import canonical_id, mention_key

logger = logging.getLogger(__name__)

# Document states
PENDING = "PENDING"
PROCESSING = "PROCESSING"
EXTRACTED = "EXTRACTED"
RESOLVED = "RESOLVED"
COMMITTED = "COMMITTED"
FAILED = "FAILED"
DOCUMENT_STATES = (PENDING, PROCESSING, EXTRACTED, RESOLVED, COMMITTED, FAILED)

# Frontier / search states
DONE = "DONE"
FRONTIER_STATES = (PENDING, PROCESSING, DONE, FAILED)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    content_hash  TEXT NOT NULL UNIQUE,
    source_url    TEXT,
    source_path   TEXT,
    origin_entity TEXT,
    tier          INTEGER,
    status        TEXT NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_status_idx ON documents(status);
CREATE INDEX IF NOT EXISTS documents_url_idx    ON documents(source_url);

CREATE TABLE IF NOT EXISTS document_origins (
    doc_id TEXT NOT NULL,
    entity_key TEXT NOT NULL,
    tier INTEGER,
    PRIMARY KEY (doc_id, entity_key)
);

CREATE TABLE IF NOT EXISTS expansion_document_usage (
    doc_id TEXT NOT NULL,
    entity_key TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    completed INTEGER NOT NULL DEFAULT 0,
    elapsed_seconds REAL NOT NULL DEFAULT 0,
    expansion_edges INTEGER NOT NULL DEFAULT 0,
    metrics_json TEXT,
    PRIMARY KEY (doc_id, entity_key)
);

CREATE TABLE IF NOT EXISTS expansion_run_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS frontier (
    entity_key      TEXT PRIMARY KEY,
    entity_name     TEXT NOT NULL,
    tier            INTEGER NOT NULL,
    status          TEXT NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    discovered_from TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS frontier_status_idx ON frontier(status, tier);

CREATE TABLE IF NOT EXISTS searches (
    search_id    TEXT PRIMARY KEY,
    entity_key   TEXT NOT NULL,
    tier         INTEGER NOT NULL,
    provider     TEXT NOT NULL,
    query        TEXT NOT NULL,
    status       TEXT NOT NULL,
    result_count INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS searches_entity_idx ON searches(entity_key);

CREATE TABLE IF NOT EXISTS entity_resolution (
    mention_key    TEXT PRIMARY KEY,
    label          TEXT NOT NULL,
    mention        TEXT NOT NULL,
    canonical_id   TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    method         TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS entity_resolution_label_idx ON entity_resolution(label);

CREATE TABLE IF NOT EXISTS fact_validation (
    candidate_fact_id TEXT PRIMARY KEY,
    ontology_status   TEXT,
    evidence_status   TEXT,
    repair_status     TEXT,
    validated_at      TEXT NOT NULL,
    model             TEXT
);

CREATE TABLE IF NOT EXISTS edge_verification (
    edge_key                  TEXT PRIMARY KEY,
    status                    TEXT NOT NULL,
    independent_support_count INTEGER NOT NULL DEFAULT 0,
    last_checked_at           TEXT NOT NULL,
    verification_complete     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS verification_searches (
    search_id    TEXT PRIMARY KEY,
    edge_key     TEXT NOT NULL,
    query        TEXT NOT NULL,
    provider     TEXT NOT NULL,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS verification_searches_edge_idx ON verification_searches(edge_key);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value):
    """Arrow/NumPy collections in read-back provenance retain JSON structure."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def connect(path: Path) -> sqlite3.Connection:
    """Open the state database, creating the schema if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(SCHEMA)
    # Additive migration: existing checkpoint and Parquet identities remain valid.
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(frontier)")}
    for name, definition in {
        "entity_type": "TEXT NOT NULL DEFAULT 'Company'",
        "expandable": "INTEGER NOT NULL DEFAULT 1",
        "priority": "REAL NOT NULL DEFAULT 0",
        "priority_reason": "TEXT",
        "stop_reason": "TEXT",
        "zero_gain_streak": "INTEGER NOT NULL DEFAULT 0",
        "rescue_mode": "INTEGER NOT NULL DEFAULT 0",
        "escalation_stage": "TEXT NOT NULL DEFAULT 'primary'",
        "query_stages_json": "TEXT",
    }.items():
        if name not in columns:
            connection.execute(f"ALTER TABLE frontier ADD COLUMN {name} {definition}")
    connection.execute(
        "INSERT OR IGNORE INTO document_origins(doc_id, entity_key, tier) "
        "SELECT doc_id, origin_entity, tier FROM documents WHERE origin_entity IS NOT NULL"
    )
    # Existing selected expansion documents already consumed budget, even when a
    # later stage failed. Never give an old checkpoint a fresh document allowance.
    connection.execute(
        "INSERT OR IGNORE INTO expansion_document_usage(doc_id, entity_key, attempts, completed) "
        "SELECT doc_id, origin_entity, attempts, CASE WHEN status IN (?, ?) THEN 1 ELSE 0 END "
        "FROM documents WHERE origin_entity IS NOT NULL AND (attempts > 0 OR status IN (?, ?))",
        (RESOLVED, COMMITTED, RESOLVED, COMMITTED),
    )
    connection.commit()
    return connection


def reset(path: Path) -> None:
    """Delete the state database, for an explicit fresh run (--restart)."""
    path = Path(path)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()
    logger.info("checkpoint state cleared: %s", path)


# --- documents ---------------------------------------------------------------------


def find_document(
    connection: sqlite3.Connection,
    content_hash: str,
    source_url: Optional[str] = None,
) -> Optional[sqlite3.Row]:
    """Look a document up by content hash, then by normalized URL."""
    row = connection.execute(
        "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    if row is None and source_url:
        row = connection.execute(
            "SELECT * FROM documents WHERE source_url = ?", (source_url,)
        ).fetchone()
    return row


def register_document(
    connection: sqlite3.Connection,
    record: dict[str, Any],
    origin_entity: Optional[str] = None,
    tier: Optional[int] = None,
) -> tuple[str, bool]:
    """Register a document, or return the existing one.

    Returns ``(doc_id, is_new)``. ``is_new`` is False when the same content hash or
    normalized URL is already known, which is the document-level deduplication rule.
    """
    existing = find_document(connection, record["content_hash"], record.get("source_url"))
    if existing is not None:
        if origin_entity:
            add_document_origin(connection, existing["doc_id"], origin_entity, tier)
        return existing["doc_id"], False

    now = _now()
    connection.execute(
        "INSERT INTO documents (doc_id, content_hash, source_url, source_path, "
        "origin_entity, tier, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record["doc_id"], record["content_hash"], record.get("source_url"),
            record.get("source_path"), origin_entity, tier, PENDING, now, now,
        ),
    )
    connection.commit()
    if origin_entity:
        add_document_origin(connection, record["doc_id"], origin_entity, tier)
    return record["doc_id"], True


def set_document_status(
    connection: sqlite3.Connection,
    doc_id: str,
    status: str,
    error: Optional[str] = None,
    increment_attempts: bool = False,
) -> None:
    if status not in DOCUMENT_STATES:
        raise ValueError(f"unknown document status {status!r}")
    connection.execute(
        "UPDATE documents SET status = ?, error = ?, updated_at = ?, "
        "attempts = attempts + ? WHERE doc_id = ?",
        (status, error, _now(), 1 if increment_attempts else 0, doc_id),
    )
    connection.commit()


def document_status(connection: sqlite3.Connection, doc_id: str) -> Optional[str]:
    row = connection.execute(
        "SELECT status FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    return row["status"] if row else None


def documents_with_status(
    connection: sqlite3.Connection, statuses: Iterable[str]
) -> list[sqlite3.Row]:
    statuses = tuple(statuses)
    placeholders = ",".join("?" * len(statuses))
    return connection.execute(
        f"SELECT * FROM documents WHERE status IN ({placeholders}) ORDER BY created_at",
        statuses,
    ).fetchall()


def document_counts(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute(
        "SELECT status, COUNT(*) AS n FROM documents GROUP BY status"
    ).fetchall()
    return {row["status"]: row["n"] for row in rows}


def document_ids_for_entity(connection: sqlite3.Connection, entity_key: str) -> list[str]:
    """Documents retrieved for one frontier entity."""
    rows = connection.execute(
        "SELECT doc_id FROM document_origins WHERE entity_key = ? "
        "UNION SELECT doc_id FROM documents WHERE origin_entity = ?", (entity_key, entity_key)
    ).fetchall()
    return [row["doc_id"] for row in rows]


def add_document_origin(connection, doc_id: str, entity_key: str, tier: Optional[int]) -> None:
    connection.execute(
        "INSERT INTO document_origins(doc_id, entity_key, tier) VALUES (?, ?, ?) "
        "ON CONFLICT(doc_id, entity_key) DO UPDATE SET tier = CASE "
        "WHEN tier IS NULL THEN excluded.tier WHEN excluded.tier IS NULL THEN tier "
        "ELSE MIN(tier, excluded.tier) END",
        (doc_id, entity_key, tier),
    )
    connection.commit()


def document_origins(connection) -> dict[str, set[str]]:
    """Read origin associations without changing original document provenance."""
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "document_origins" in tables:
        rows = connection.execute("SELECT doc_id, entity_key FROM document_origins")
    elif "documents" in tables:
        rows = connection.execute("SELECT doc_id, origin_entity FROM documents WHERE origin_entity IS NOT NULL")
    else:
        return {}
    output: dict[str, set[str]] = {}
    for doc_id, entity_key in rows:
        output.setdefault(doc_id, set()).add(entity_key)
    return output


def document_usage_count(connection, entity_key: Optional[str] = None) -> int:
    if entity_key is not None:
        row = connection.execute("SELECT COUNT(*) FROM expansion_document_usage WHERE entity_key=?", (entity_key,))
    else:
        row = connection.execute("SELECT COUNT(DISTINCT doc_id) FROM expansion_document_usage")
    return int(row.fetchone()[0])


def document_usage(connection, doc_id: str, entity_key: str):
    return connection.execute(
        "SELECT * FROM expansion_document_usage WHERE doc_id=? AND entity_key=?", (doc_id, entity_key)
    ).fetchone()


def document_attempt_count(connection, doc_id: str) -> int:
    return int(connection.execute(
        "SELECT COALESCE(SUM(attempts),0) FROM expansion_document_usage WHERE doc_id=?", (doc_id,)
    ).fetchone()[0])


def incomplete_usage_entities(connection, max_attempts: int) -> set[str]:
    return {row[0] for row in connection.execute(
        "SELECT DISTINCT u.entity_key FROM expansion_document_usage u JOIN "
        "(SELECT doc_id, SUM(attempts) AS n FROM expansion_document_usage GROUP BY doc_id) a "
        "ON a.doc_id=u.doc_id WHERE u.completed=0 AND a.n < ?", (max_attempts,))}


def reconcile_completed_usage(connection, doc_id: str) -> None:
    """Repair crash window after pipeline commit but before scheduler bookkeeping."""
    connection.execute("UPDATE expansion_document_usage SET completed=1 WHERE doc_id=?", (doc_id,))
    connection.commit()


def reserve_document(connection, doc_id: str, entity_key: str) -> None:
    """Charge a unique selected document before expensive work, including failures."""
    connection.execute(
        "INSERT OR IGNORE INTO expansion_document_usage(doc_id, entity_key) VALUES (?, ?)", (doc_id, entity_key)
    )
    connection.execute(
        "UPDATE expansion_document_usage SET attempts=attempts+1 WHERE doc_id=? AND entity_key=?", (doc_id, entity_key)
    )
    connection.commit()


def finish_document_usage(connection, doc_id: str, entity_key: str, *, completed: bool,
                          elapsed_seconds: float, expansion_edges: int = 0, metrics=None) -> None:
    connection.execute(
        "UPDATE expansion_document_usage SET completed=?, elapsed_seconds=elapsed_seconds+?, "
        "expansion_edges=MAX(expansion_edges, ?), metrics_json=? WHERE doc_id=? AND entity_key=?",
        (int(completed), max(0, elapsed_seconds), expansion_edges,
         json.dumps(metrics or {}, default=str), doc_id, entity_key),
    )
    connection.commit()


def run_value(connection, key: str, default=None):
    row = connection.execute("SELECT value FROM expansion_run_state WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_run_value(connection, key: str, value) -> None:
    connection.execute(
        "INSERT INTO expansion_run_state(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value)),
    )
    connection.commit()


# --- expansion frontier ------------------------------------------------------------


def enqueue_entity(
    connection: sqlite3.Connection,
    entity_name: str,
    tier: int,
    discovered_from: Optional[str] = None,
    *,
    entity_id: Optional[str] = None,
    entity_type: str = "Company",
    expandable: bool = True,
    priority: float = 0.0,
    priority_reason: Optional[dict] = None,
    stop_reason: Optional[str] = None,
) -> bool:
    """Add a company to the frontier, or lower its tier when a shorter path is found.

    Keyed by canonical id, so "BMW" and "BMW AG" cannot both occupy the frontier.
    If the entity already exists at a higher tier, its tier is lowered to the shorter
    supply-chain distance and ``discovered_from`` is updated. Same-or-higher tiers
    leave the existing row unchanged (returns False).
    """
    key = entity_id or canonical_id(entity_type, entity_name)
    if not key.startswith(entity_type.lower() + ":"):
        raise ValueError(f"entity id/type mismatch: {key!r} vs {entity_type!r}")
    now = _now()
    existing = connection.execute(
        "SELECT * FROM frontier WHERE entity_key = ?", (key,)
    ).fetchone()
    if existing is None:
        connection.execute(
            "INSERT INTO frontier (entity_key, entity_name, tier, status, "
            "discovered_from, created_at, updated_at, entity_type, expandable, priority, priority_reason, stop_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, entity_name, tier, PENDING if expandable else DONE, discovered_from, now, now,
             entity_type, int(expandable), priority, json.dumps(priority_reason or {}, default=_json_default), stop_reason),
        )
        connection.commit()
        return True

    if int(existing["tier"]) <= int(tier):
        if priority > float(existing["priority"]):
            connection.execute(
                "UPDATE frontier SET priority=?, priority_reason=? WHERE entity_key=?",
                (priority, json.dumps(priority_reason or {}, default=_json_default), key),
            )
            connection.commit()
        return False

    # Shorter valid path discovered: lower the tier. Re-queue if already DONE so the
    # entity can be expanded again at the corrected depth when still within max_tier.
    new_status = existing["status"]
    if existing["status"] == DONE and expandable:
        new_status = PENDING
    connection.execute(
        "UPDATE frontier SET tier = ?, discovered_from = ?, status = ?, "
        "updated_at = ?, entity_name = ?, expandable=?, stop_reason=? WHERE entity_key = ?",
        (tier, discovered_from or existing["discovered_from"], new_status, now,
         entity_name, int(expandable), stop_reason, key),
    )
    connection.commit()
    logger.info(
        "frontier tier lowered entity=%s from=%s to=%s via=%s",
        entity_name, existing["tier"], tier, discovered_from,
    )
    return True


def next_frontier_entity(
    connection: sqlite3.Connection,
    max_tier: int,
    *,
    prefer_under_covered: bool = True,
    max_expandable_entities_by_tier: Optional[dict] = None,
    allowed_entity_keys: Optional[set[str]] = None,
    run_new_claims_by_tier: Optional[dict] = None,
) -> Optional[sqlite3.Row]:
    """Claim the next pending entity within the tier boundary, lowest tier first.

    Priority (and optional per-run scheduling batch sizes) decide ORDER only.
    An accepted supplier that is PENDING is never permanently excluded.

    ``max_expandable_entities_by_tier`` is a soft per-run scheduling batch size:
    prefer not to *start* more than N never-attempted entities at a tier while
    other tiers still have pending work. When nothing else is claimable, the
    batch limit is ignored so every valid supplier eventually runs.
    """
    caps = max_expandable_entities_by_tier or {}
    run_claims = run_new_claims_by_tier if run_new_claims_by_tier is not None else {}
    rows = connection.execute(
        "SELECT f.*, COALESCE(d.doc_n, 0) AS doc_n FROM frontier f LEFT JOIN "
        "(SELECT entity_key, COUNT(*) AS doc_n FROM expansion_document_usage GROUP BY entity_key) d "
        "ON d.entity_key=f.entity_key WHERE f.status=? AND f.tier < ? AND f.expandable=1 "
        "ORDER BY f.tier, f.priority DESC, f.entity_key", (PENDING, max_tier),
    ).fetchall()
    # Clear obsolete permanent-exclusion markers from older builds.
    connection.execute(
        "UPDATE frontier SET stop_reason=NULL WHERE status=? AND stop_reason='frontier_beam_cap'",
        (PENDING,),
    )
    pending_by_tier: dict[int, int] = {}
    for candidate in rows:
        if allowed_entity_keys is not None and candidate["entity_key"] not in allowed_entity_keys:
            continue
        tier = int(candidate["tier"])
        pending_by_tier[tier] = pending_by_tier.get(tier, 0) + 1

    def _other_tiers_have_work(skip_tier: int) -> bool:
        return any(count > 0 for tier, count in pending_by_tier.items() if tier != skip_tier)

    eligible = []
    deferred_batch: list = []
    for candidate in rows:
        if allowed_entity_keys is not None and candidate["entity_key"] not in allowed_entity_keys:
            continue
        tier = int(candidate["tier"])
        cap = caps.get(tier, caps.get(str(tier)))
        # Soft per-run batch: defer never-attempted entities only while other tiers
        # still have claimable work. Never a lifetime / permanent exclusion.
        if (
            tier
            and cap is not None
            and int(candidate["attempts"] or 0) == 0
            and int(run_claims.get(tier, 0)) >= int(cap)
            and _other_tiers_have_work(tier)
        ):
            deferred_batch.append(candidate)
            continue
        eligible.append(candidate)
    if not eligible and deferred_batch:
        # Batch full but no alternative work — process deferred suppliers now.
        eligible = deferred_batch
    if prefer_under_covered:
        eligible.sort(key=lambda item: (item["tier"], item["doc_n"], -float(item["priority"] or 0), item["entity_key"]))
    row = eligible[0] if eligible else None
    if row is None:
        connection.commit()
        return None
    connection.execute(
        "UPDATE frontier SET status = ?, attempts = attempts + 1, updated_at = ?, stop_reason=NULL "
        "WHERE entity_key = ?",
        (PROCESSING, _now(), row["entity_key"]),
    )
    connection.commit()
    return connection.execute(
        "SELECT * FROM frontier WHERE entity_key = ?", (row["entity_key"],)
    ).fetchone()


def set_frontier_status(
    connection: sqlite3.Connection,
    entity_key: str,
    status: str,
    error: Optional[str] = None,
    *,
    stop_reason: Optional[str] = None,
) -> None:
    if status not in FRONTIER_STATES:
        raise ValueError(f"unknown frontier status {status!r}")
    connection.execute(
        "UPDATE frontier SET status = ?, error = ?, updated_at = ?, stop_reason=? WHERE entity_key = ?",
        (status, error, _now(), stop_reason, entity_key),
    )
    connection.commit()


def requeue_stale_processing(connection: sqlite3.Connection) -> int:
    """Return entities left PROCESSING by an interrupted run to PENDING."""
    cursor = connection.execute(
        "UPDATE frontier SET status = ?, updated_at = ? WHERE status = ?",
        (PENDING, _now(), PROCESSING),
    )
    connection.commit()
    return cursor.rowcount


def frontier_counts(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute(
        "SELECT status, COUNT(*) AS n FROM frontier GROUP BY status"
    ).fetchall()
    return {row["status"]: row["n"] for row in rows}


def frontier_tier_counts(connection: sqlite3.Connection) -> dict[int, int]:
    rows = connection.execute(
        "SELECT tier, COUNT(*) AS n FROM frontier GROUP BY tier ORDER BY tier"
    ).fetchall()
    return {row["tier"]: row["n"] for row in rows}


def pending_frontier_count(connection: sqlite3.Connection, max_tier: int) -> int:
    """How many PENDING frontier entities remain within the tier boundary."""
    row = connection.execute(
        "SELECT COUNT(*) AS n FROM frontier WHERE status = ? AND tier < ? AND expandable=1",
        (PENDING, max_tier),
    ).fetchone()
    return int(row["n"] if row else 0)

# --- searches ----------------------------------------------------------------------


def search_id(entity_key: str, tier: int, provider: str, query: str) -> str:
    payload = "|".join([entity_key, str(tier), provider, query])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def search_completed(connection: sqlite3.Connection, sid: str) -> bool:
    """True when this exact query already ran, so retrieval is not repeated."""
    row = connection.execute(
        "SELECT status FROM searches WHERE search_id = ?", (sid,)
    ).fetchone()
    return bool(row and row["status"] == DONE)


def record_search(
    connection: sqlite3.Connection,
    sid: str,
    entity_key: str,
    tier: int,
    provider: str,
    query: str,
    status: str,
    result_count: int = 0,
    error: Optional[str] = None,
) -> None:
    now = _now()
    connection.execute(
        "INSERT INTO searches (search_id, entity_key, tier, provider, query, status, "
        "result_count, error, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(search_id) DO UPDATE SET status = excluded.status, "
        "result_count = excluded.result_count, error = excluded.error, "
        "updated_at = excluded.updated_at",
        (sid, entity_key, tier, provider, query, status, result_count, error, now, now),
    )
    connection.commit()


# --- entity resolution cache -------------------------------------------------------


def load_resolution_cache(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """All previously resolved mentions, keyed by ``mention_key``.

    Prompt 2 is skipped for any mention already in here, which is what stops a resumed
    run from paying for the same disambiguation twice.
    """
    rows = connection.execute("SELECT * FROM entity_resolution").fetchall()
    return {row["mention_key"]: dict(row) for row in rows}


def save_resolution(
    connection: sqlite3.Connection, entries: Iterable[dict[str, Any]]
) -> int:
    now = _now()
    payload = [
        (
            mention_key(entry["label"], entry["mention"]),
            entry["label"], entry["mention"], entry["canonical_id"],
            entry["canonical_name"], entry.get("method", "deterministic"), now,
        )
        for entry in entries
    ]
    connection.executemany(
        "INSERT INTO entity_resolution (mention_key, label, mention, canonical_id, "
        "canonical_name, method, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(mention_key) DO UPDATE SET canonical_id = excluded.canonical_id, "
        "canonical_name = excluded.canonical_name, method = excluded.method, "
        "updated_at = excluded.updated_at",
        payload,
    )
    connection.commit()
    return len(payload)
