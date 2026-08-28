"""Extracted facts as a flat table, kept outside Neo4j.

PROJECT_DECISIONS.md section 8: "Intermediate documents and facts are stored outside
Neo4j as well", so the graph can be rebuilt without repeating LLM calls.

One row per extracted relationship, carrying both the raw mention and the canonical
entity it resolved to, plus the document and chunk it came from. neo4j-graphrag does
ship a Parquet formatter, but it emits a Neo4j bulk-import layout; a flat triple table
is what a human actually reads when checking whether an extraction was correct.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from neo4j_graphrag.components.types import Neo4jGraph

logger = logging.getLogger(__name__)

FACT_COLUMNS = [
    "fact_id", "doc_id", "chunk_index",
    "subject_id", "subject_type", "subject_name", "subject_mention",
    "relation",
    "object_id", "object_type", "object_name", "object_mention",
    "evidence", "extracted_at",
]


def fact_id(doc_id: str, subject_id: str, relation: str, object_id: str) -> str:
    """Deterministic row identifier, so re-running a document cannot duplicate facts."""
    payload = "|".join([doc_id, subject_id, relation, object_id])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def chunk_index_of(node_id: str, chunk_index_by_id: dict[str, int]) -> int:
    """Recover the chunk a node came from; the extractor prefixes ids with the chunk id."""
    prefix = node_id.split(":", 1)[0]
    return chunk_index_by_id.get(prefix, -1)


def graph_to_facts(
    graph: Neo4jGraph,
    document: dict[str, Any],
    chunk_index_by_id: dict[str, int],
    resolution: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build the facts table for one document."""
    extracted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []

    for relationship in graph.relationships:
        subject = resolution.get(relationship.start_node_id)
        target = resolution.get(relationship.end_node_id)
        if subject is None or target is None:
            logger.warning(
                "dropping %s: an endpoint was not resolved to a canonical entity",
                relationship.type,
            )
            continue
        rows.append(
            {
                "fact_id": fact_id(
                    document["doc_id"],
                    subject["canonical_id"],
                    relationship.type,
                    target["canonical_id"],
                ),
                "doc_id": document["doc_id"],
                "chunk_index": chunk_index_of(relationship.start_node_id, chunk_index_by_id),
                "subject_id": subject["canonical_id"],
                "subject_type": subject["label"],
                "subject_name": subject["canonical_name"],
                "subject_mention": subject["mention"],
                "relation": relationship.type,
                "object_id": target["canonical_id"],
                "object_type": target["label"],
                "object_name": target["canonical_name"],
                "object_mention": target["mention"],
                "evidence": (relationship.properties or {}).get("evidence", ""),
                "extracted_at": extracted_at,
            }
        )

    return pd.DataFrame(rows, columns=FACT_COLUMNS)


def upsert_facts_parquet(frame: pd.DataFrame, path: Path) -> int:
    """Append facts to the Parquet layer, replacing rows with the same fact_id.

    Returns the total row count after the write.
    """
    if frame.empty and not path.exists():
        frame.to_parquet(path, index=False)
        return 0
    if path.exists():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
    frame = frame.drop_duplicates(subset="fact_id", keep="last").reset_index(drop=True)
    frame.to_parquet(path, index=False)
    return len(frame)
