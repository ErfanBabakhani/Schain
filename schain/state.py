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
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path) -> sqlite3.Connection:
    """Open the state database, creating the schema if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(SCHEMA)
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
        "SELECT doc_id FROM documents WHERE origin_entity = ?", (entity_key,)
    ).fetchall()
    return [row["doc_id"] for row in rows]


# --- expansion frontier ------------------------------------------------------------


def enqueue_entity(
    connection: sqlite3.Connection,
    entity_name: str,
    tier: int,
    discovered_from: Optional[str] = None,
) -> bool:
    """Add a company to the frontier. Returns False if it is already known.

    Keyed by canonical id, so "BMW" and "BMW AG" cannot both occupy the frontier, and a
    company already seen at a lower tier is not re-added at a higher one.
    """
    key = canonical_id("Company", entity_name)
    now = _now()
    cursor = connection.execute(
        "INSERT OR IGNORE INTO frontier (entity_key, entity_name, tier, status, "
        "discovered_from, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (key, entity_name, tier, PENDING, discovered_from, now, now),
    )
    connection.commit()
    return cursor.rowcount > 0


def next_frontier_entity(
    connection: sqlite3.Connection, max_tier: int
) -> Optional[sqlite3.Row]:
    """Claim the next pending entity within the tier boundary, lowest tier first.

    Marking it PROCESSING here is what stops an interrupted run from redoing it silently
    on resume; a PROCESSING row is visible as unfinished work.
    """
    row = connection.execute(
        "SELECT * FROM frontier WHERE status = ? AND tier <= ? "
        "ORDER BY tier, created_at LIMIT 1",
        (PENDING, max_tier),
    ).fetchone()
    if row is None:
        return None
    connection.execute(
        "UPDATE frontier SET status = ?, attempts = attempts + 1, updated_at = ? "
        "WHERE entity_key = ?",
        (PROCESSING, _now(), row["entity_key"]),
    )
    connection.commit()
    return row


def set_frontier_status(
    connection: sqlite3.Connection,
    entity_key: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    if status not in FRONTIER_STATES:
        raise ValueError(f"unknown frontier status {status!r}")
    connection.execute(
        "UPDATE frontier SET status = ?, error = ?, updated_at = ? WHERE entity_key = ?",
        (status, error, _now(), entity_key),
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
