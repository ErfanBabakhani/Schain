"""Idempotent Neo4j writes.

Why this is not ``neo4j_graphrag.components.kg_writer.Neo4jWriter``: that writer issues
``CREATE (n:__KGBuilder__ {...})`` and relies on a follow-up resolver pass
(``apoc.refactor.mergeNodes``) to collapse duplicates afterwards. PROJECT_DECISIONS.md
section 9 requires the opposite order - canonical ids plus MERGE, so a repeated or
resumed run cannot create a duplicate in the first place.

Every node gets its ontology label plus a shared ``:Entity`` label. The shared label
carries the uniqueness constraint on ``canonical_id`` and lets relationship writes match
endpoints without knowing their type.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

import neo4j

logger = logging.getLogger(__name__)

CONSTRAINT_NAME = "entity_canonical_id"


def _validate(value: str, allowed: Iterable[str], kind: str) -> str:
    """Guard the only two values interpolated into Cypher: labels and relation types."""
    allowed = tuple(allowed)
    if value not in allowed:
        raise ValueError(f"unknown {kind} {value!r}; allowed: {', '.join(allowed)}")
    return value


def ensure_constraints(driver: neo4j.Driver, database: str) -> None:
    """Create the uniqueness constraint that makes MERGE on canonical_id safe and fast."""
    driver.execute_query(
        f"CREATE CONSTRAINT {CONSTRAINT_NAME} IF NOT EXISTS "
        "FOR (n:Entity) REQUIRE n.canonical_id IS UNIQUE",
        database_=database,
    )


def _batched(rows: list[dict[str, Any]], size: int):
    """Yield fixed-size batches, so one statement does not carry an unbounded payload."""
    for start in range(0, len(rows), max(1, size)):
        yield rows[start : start + max(1, size)]


def merge_nodes(
    driver: neo4j.Driver,
    database: str,
    nodes: list[dict[str, Any]],
    entity_types: Iterable[str],
    batch_size: int = 1000,
) -> int:
    """MERGE canonical entities, one statement per ontology label.

    Labels cannot be parameterized in Cypher, so nodes are grouped by label and each
    label is checked against the ontology before it is interpolated.
    """
    written = 0
    by_label: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        by_label.setdefault(node["label"], []).append(node)

    for label, rows in by_label.items():
        _validate(label, entity_types, "entity label")
        query = (
            "UNWIND $rows AS row "
            f"MERGE (e:{label}:Entity {{canonical_id: row.canonical_id}}) "
            "ON CREATE SET e.name = row.name, e.first_seen_doc_id = row.doc_id "
            "SET e.name = row.name, "
            # set union, so aliases accumulate without duplicating
            "    e.aliases = [a IN coalesce(e.aliases, []) WHERE NOT a IN row.aliases]"
            "               + row.aliases"
        )
        for batch in _batched(rows, batch_size):
            driver.execute_query(query, rows=batch, database_=database)
        written += len(rows)
        logger.info("merged %d %s node(s)", len(rows), label)
    return written


def merge_relationships(
    driver: neo4j.Driver,
    database: str,
    relationships: list[dict[str, Any]],
    relation_types: Iterable[str],
    batch_size: int = 1000,
) -> int:
    """MERGE relationships, one statement per relation type.

    ``source_doc_ids`` accumulates the documents a relationship was seen in, so the same
    fact arriving from a second document is recorded without creating a second edge.
    """
    written = 0
    by_type: dict[str, list[dict[str, Any]]] = {}
    for relationship in relationships:
        by_type.setdefault(relationship["type"], []).append(relationship)

    for relation_type, rows in by_type.items():
        _validate(relation_type, relation_types, "relation type")
        query = (
            "UNWIND $rows AS row "
            "MATCH (a:Entity {canonical_id: row.start_id}) "
            "MATCH (b:Entity {canonical_id: row.end_id}) "
            f"MERGE (a)-[r:{relation_type}]->(b) "
            "ON CREATE SET r.first_seen_doc_id = row.doc_id "
            "SET r.evidence = row.evidence, "
            "    r.source_doc_ids = [d IN coalesce(r.source_doc_ids, []) "
            "                        WHERE d <> row.doc_id] + [row.doc_id]"
        )
        for batch in _batched(rows, batch_size):
            driver.execute_query(query, rows=batch, database_=database)
        written += len(rows)
        logger.info("merged %d %s relationship(s)", len(rows), relation_type)
    return written


def facts_to_graph_rows(
    facts: "Any", resolution: Optional[dict[str, dict[str, Any]]] = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Turn the facts table into the node and relationship rows the writer expects.

    Deriving the graph write from the facts table (rather than from the raw extraction)
    keeps Neo4j and the Parquet layer describing exactly the same set of facts.
    """
    nodes: dict[str, dict[str, Any]] = {}
    aliases_by_id: dict[str, set[str]] = {}
    for entry in (resolution or {}).values():
        aliases_by_id.setdefault(entry["canonical_id"], set()).update(entry["aliases"])

    relationships: list[dict[str, Any]] = []
    for row in facts.to_dict("records"):
        for prefix in ("subject", "object"):
            canonical = row[f"{prefix}_id"]
            aliases_by_id.setdefault(canonical, set()).add(row[f"{prefix}_mention"])
            nodes[canonical] = {
                "canonical_id": canonical,
                "label": row[f"{prefix}_type"],
                "name": row[f"{prefix}_name"],
                "aliases": sorted(aliases_by_id.get(canonical, set())),
                "doc_id": row["doc_id"],
            }
        relationships.append(
            {
                "type": row["relation"],
                "start_id": row["subject_id"],
                "end_id": row["object_id"],
                "evidence": row["evidence"],
                "doc_id": row["doc_id"],
            }
        )
    return list(nodes.values()), relationships
