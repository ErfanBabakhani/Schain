"""Referential-integrity checks for model-extracted graphs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from neo4j_graphrag.components.types import Neo4jGraph


def _chunk_index(node_id: str, chunk_index_by_id: dict[str, int]) -> int:
    prefix = str(node_id).split(":", 1)[0]
    return int(chunk_index_by_id.get(prefix, -1))


def enforce_referential_integrity(
    graph: Neo4jGraph,
    *,
    document_id: str,
    chunk_index_by_id: dict[str, int],
) -> tuple[Neo4jGraph, list[dict[str, Any]]]:
    """Return a graph without dangling edges plus lossless diagnostics.

    Missing entities are never synthesized from relationship ids.  The original
    relationship payload is retained so extraction failures remain inspectable.
    """
    node_ids = {node.id for node in graph.nodes}
    valid = []
    failures: list[dict[str, Any]] = []
    for relationship in graph.relationships:
        missing = [
            endpoint
            for endpoint in (relationship.start_node_id, relationship.end_node_id)
            if endpoint not in node_ids
        ]
        if not missing:
            valid.append(relationship)
            continue
        payload = relationship.model_dump(mode="json")
        fingerprint = hashlib.sha256(
            json.dumps(
                [document_id, payload, sorted(missing)],
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        chunk_id = str(relationship.start_node_id).split(":", 1)[0]
        if chunk_id not in chunk_index_by_id:
            chunk_id = str(relationship.end_node_id).split(":", 1)[0]
        failures.append(
            {
                "integrity_failure_id": fingerprint,
                "doc_id": document_id,
                "chunk_id": chunk_id if chunk_id in chunk_index_by_id else None,
                "chunk_index": _chunk_index(
                    relationship.start_node_id,
                    chunk_index_by_id,
                ),
                "pipeline_stage": "extraction_integrity",
                "relation_type": relationship.type,
                "start_node_id": relationship.start_node_id,
                "end_node_id": relationship.end_node_id,
                "missing_node_ids": missing,
                "evidence": (relationship.properties or {}).get("evidence", ""),
                "reason": "missing_relationship_endpoint",
                "repair_attempted": False,
                "repair_status": "not_attempted",
                "original_relationship": payload,
            }
        )
    return Neo4jGraph(nodes=list(graph.nodes), relationships=valid), failures
