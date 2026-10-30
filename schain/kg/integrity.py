"""Conservative referential-integrity recovery for model-extracted graphs."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional

from neo4j_graphrag.components.types import Neo4jGraph, Neo4jNode

from ..names import normalization_key


def _chunk_index(node_id: str, chunk_index_by_id: dict[str, int]) -> int:
    prefix = str(node_id).split(":", 1)[0]
    return int(chunk_index_by_id.get(prefix, -1))


def _fingerprint(document_id: str, payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps([document_id, payload], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _expected_labels(
    ontology: Optional[dict[str, Any]], relation: str, endpoint: str,
) -> set[str]:
    index = 0 if endpoint == "start" else 2
    labels = {
        str(pattern[index])
        for pattern in ((ontology or {}).get("patterns") or [])
        if len(pattern) == 3 and str(pattern[1]) == relation
    }
    if labels:
        return labels
    fallback = {
        ("locatedIn", "start"): {"Company", "Mine"},
        ("locatedIn", "end"): {"Location"},
        ("suppliesTo", "start"): {"Company", "Mine"},
        ("suppliesTo", "end"): {"Company"},
        ("produces", "start"): {"Company", "Mine"},
        ("produces", "end"): {"Product", "Material"},
        ("owns", "start"): {"Company", "Person"},
        ("owns", "end"): {"Company", "Mine"},
    }
    return fallback.get((relation, endpoint), set())


def _location_for_subject(evidence: str, subject_name: str) -> Optional[str]:
    """Extract one verbatim headquarters/location phrase tied to a named subject."""
    if not evidence or not subject_name:
        return None
    lowered = evidence.casefold()
    start = lowered.find(subject_name.casefold())
    if start < 0:
        return None
    segment = evidence[start + len(subject_name):]
    marker = re.search(
        r"(?:\bis\s+headquartered|\bhas\s+(?:its\s+)?(?:registered\s+)?headquarters|"
        r"\bmaintains\s+(?:its\s+)?(?:registered\s+)?headquarters|\bis\s+based|"
        r"\bis\s+located|\bhas\s+(?:a\s+)?(?:facility|plant))\s+in\s+",
        segment,
        flags=re.IGNORECASE,
    )
    if marker is None:
        return None
    value = segment[marker.end():]
    value = re.split(r"[.;!?]", value, maxsplit=1)[0]
    # Stop at a coordinated clause that introduces a second company's own
    # location statement.  A plain "City, Country" comma remains untouched.
    value = re.split(
        r",\s*(?:whereas|while|and)\s+(?=[^,.;!?]{1,100}\b(?:is\s+headquartered|"
        r"has\s+(?:its\s+)?(?:registered\s+)?headquarters|"
        r"maintains\s+(?:its\s+)?(?:registered\s+)?headquarters|is\s+based|"
        r"is\s+located)\b)",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = re.sub(r",\s*(?:whereas|while|and)\s*$", "", value, flags=re.IGNORECASE)
    value = value.strip(" \t\r\n,;:")
    if not value or len(value) > 120 or len(value.split()) > 8:
        return None
    if value.casefold() not in evidence.casefold():
        return None
    return value


def _recovered_node_id(original_id: str, relationship_payload: dict[str, Any], name: str) -> str:
    chunk = str(original_id).split(":", 1)[0]
    digest = hashlib.sha256(
        json.dumps([relationship_payload, original_id, name], sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"{chunk}:recovered-{digest}"


def enforce_referential_integrity(
    graph: Neo4jGraph,
    *,
    document_id: str,
    chunk_index_by_id: dict[str, int],
    ontology: Optional[dict[str, Any]] = None,
    recovery_outcomes: Optional[list[dict[str, Any]]] = None,
) -> tuple[Neo4jGraph, list[dict[str, Any]]]:
    """Recover only unambiguous dangling endpoints, then return remaining failures.

    Recovery uses the model output and verbatim relationship evidence only. Every
    repaired relationship still passes normal entity resolution, grounding, ontology,
    direction, and semantic validation downstream.
    """
    original_nodes = list(graph.nodes)
    original_node_ids = {node.id for node in original_nodes}
    node_by_id = {node.id: node for node in original_nodes}
    nodes_by_name: dict[str, list[Neo4jNode]] = {}
    for node in original_nodes:
        name = str((node.properties or {}).get("name") or "")
        if name:
            nodes_by_name.setdefault(normalization_key(name), []).append(node)

    valid_relationships = []
    reconstructed_nodes: list[Neo4jNode] = []
    failures: list[dict[str, Any]] = []
    outcomes = recovery_outcomes if recovery_outcomes is not None else []

    for relationship in graph.relationships:
        original_payload = relationship.model_dump(mode="json")
        relationship_key = _fingerprint(document_id, original_payload)
        updated = relationship
        unrecoverable: list[str] = []
        for endpoint, field in (("start", "start_node_id"), ("end", "end_node_id")):
            endpoint_id = str(getattr(updated, field))
            if endpoint_id in original_node_ids:
                outcomes.append({
                    "event_type": "endpoint_already_valid",
                    "relationship_integrity_id": relationship_key,
                    "relationship_type": relationship.type,
                    "endpoint": endpoint,
                    "original_endpoint_id": endpoint_id,
                    "recovered_endpoint_id": endpoint_id,
                    "reason": "exact_id_match",
                })
                continue

            expected = _expected_labels(ontology, relationship.type, endpoint)
            suffix = endpoint_id.split(":", 1)[-1]
            named_matches = nodes_by_name.get(normalization_key(suffix), [])
            compatible = [node for node in named_matches if not expected or node.label in expected]
            incompatible = [node for node in named_matches if node not in compatible]

            if len(compatible) == 1:
                recovered_id = compatible[0].id
                updated = updated.model_copy(update={field: recovered_id})
                outcomes.append({
                    "event_type": "endpoint_id_recovered",
                    "relationship_integrity_id": relationship_key,
                    "relationship_type": relationship.type,
                    "endpoint": endpoint,
                    "original_endpoint_id": endpoint_id,
                    "recovered_endpoint_id": recovered_id,
                    "recovered_name": (compatible[0].properties or {}).get("name"),
                    "recovered_type": compatible[0].label,
                    "reason": "endpoint_name_uniquely_matches_existing_node",
                })
                continue

            reconstructed_name: Optional[str] = None
            reconstructed_type: Optional[str] = None
            reconstruction_reason: Optional[str] = None
            if compatible and len({
                (node.label, normalization_key((node.properties or {}).get("name", "")))
                for node in compatible
            }) == 1:
                reconstructed_name = str((compatible[0].properties or {}).get("name") or "")
                reconstructed_type = compatible[0].label
                reconstruction_reason = "endpoint_name_matches_repeated_same_identity"
            elif relationship.type == "locatedIn" and endpoint == "end" and expected == {"Location"}:
                subject = node_by_id.get(str(updated.start_node_id))
                subject_name = str((subject.properties or {}).get("name") or "") if subject else ""
                reconstructed_name = _location_for_subject(
                    str((relationship.properties or {}).get("evidence") or ""), subject_name,
                )
                if reconstructed_name:
                    reconstructed_type = "Location"
                    reconstruction_reason = "verbatim_subject_location_in_evidence"

            if reconstructed_name and reconstructed_type and (
                not expected or reconstructed_type in expected
            ):
                recovered_id = _recovered_node_id(
                    endpoint_id, original_payload, reconstructed_name,
                )
                reconstructed_nodes.append(Neo4jNode(
                    id=recovered_id,
                    label=reconstructed_type,
                    properties={"name": reconstructed_name},
                ))
                updated = updated.model_copy(update={field: recovered_id})
                outcomes.append({
                    "event_type": "endpoint_node_reconstructed",
                    "relationship_integrity_id": relationship_key,
                    "relationship_type": relationship.type,
                    "endpoint": endpoint,
                    "original_endpoint_id": endpoint_id,
                    "recovered_endpoint_id": recovered_id,
                    "recovered_name": reconstructed_name,
                    "recovered_type": reconstructed_type,
                    "reason": reconstruction_reason,
                    "evidence": (relationship.properties or {}).get("evidence", ""),
                })
                continue

            ambiguous = len(compatible) > 1
            event_type = "endpoint_recovery_ambiguous" if ambiguous else "endpoint_unrecoverable"
            reason = (
                "multiple_or_type_incompatible_name_matches"
                if ambiguous else "type_incompatible_name_match"
                if incompatible else "no_unambiguous_evidence_grounded_recovery"
            )
            outcomes.append({
                "event_type": event_type,
                "relationship_integrity_id": relationship_key,
                "relationship_type": relationship.type,
                "endpoint": endpoint,
                "original_endpoint_id": endpoint_id,
                "candidate_node_ids": [node.id for node in named_matches],
                "expected_types": sorted(expected),
                "reason": reason,
            })
            unrecoverable.append(endpoint_id)

        if not unrecoverable:
            valid_relationships.append(updated)
            continue

        fingerprint = _fingerprint(
            document_id, {"relationship": original_payload, "missing": sorted(unrecoverable)},
        )
        chunk_id = str(relationship.start_node_id).split(":", 1)[0]
        if chunk_id not in chunk_index_by_id:
            chunk_id = str(relationship.end_node_id).split(":", 1)[0]
        failures.append({
            "integrity_failure_id": fingerprint,
            "doc_id": document_id,
            "chunk_id": chunk_id if chunk_id in chunk_index_by_id else None,
            "chunk_index": _chunk_index(relationship.start_node_id, chunk_index_by_id),
            "pipeline_stage": "extraction_integrity",
            "relation_type": relationship.type,
            "start_node_id": relationship.start_node_id,
            "end_node_id": relationship.end_node_id,
            "missing_node_ids": unrecoverable,
            "evidence": (relationship.properties or {}).get("evidence", ""),
            "reason": "missing_relationship_endpoint",
            "repair_attempted": True,
            "repair_status": "ambiguous" if any(
                item["event_type"] == "endpoint_recovery_ambiguous"
                and item["relationship_integrity_id"] == relationship_key
                for item in outcomes
            ) else "unrecoverable",
            "original_relationship": original_payload,
        })

    unique_nodes = {node.id: node for node in [*original_nodes, *reconstructed_nodes]}
    return Neo4jGraph(
        nodes=list(unique_nodes.values()), relationships=valid_relationships,
    ), failures
