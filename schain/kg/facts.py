"""Extracted facts as a flat table, kept outside Neo4j.

One row per extracted relationship, carrying the raw mention, the canonical entity it
resolved to, the document and chunk it came from, the source URL and the model that
produced it. neo4j-graphrag does ship a Parquet formatter, but it emits a Neo4j
bulk-import layout; a flat triple table is what a human actually reads when checking
whether an extraction was correct, and it is what lets the graph be rebuilt without
repeating LLM calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from ..checkpoint import atomic_write_parquet
from neo4j_graphrag.components.types import Neo4jGraph

logger = logging.getLogger(__name__)

FACT_COLUMNS = [
    "fact_id", "doc_id", "chunk_index", "source_url",
    "subject_id", "subject_type", "subject_name", "subject_mention",
    "relation",
    "object_id", "object_type", "object_name", "object_mention",
    "evidence", "model", "extracted_at",
    # Quality / provenance (additive; empty for legacy rows).
    "validation_status", "ontology_valid", "evidence_validated",
    "retrieval_url", "canonical_source_url", "source_domain",
    "source_known", "source_class", "source_base_score", "source_effective_score",
    "content_source", "source_group_id",
    "provider", "query", "origin_entity", "tier",
    "expansion_relevant",
    "direction_explicit", "evidence_quote_grounded",
    "original_evidence", "evidence_repair_attempted",
    "evidence_repair_status", "evidence_repair_reason",
]


def fact_id(
    doc_id: str, subject_id: str, relation: str, object_id: str,
    evidence: Optional[str] = None, chunk_index: Optional[int] = None,
) -> str:
    """Identify a witnessed triple without collapsing its other supporting quotes.

    The four-argument form retains the legacy document/triple identity for callers
    reading older artifacts. New extraction includes the quote and chunk. Resume
    continues to load existing artifacts; a fresh run is needed to recover quotes
    already discarded by older versions.
    """
    if evidence is None and chunk_index is None:
        payload = "|".join([doc_id, subject_id, relation, object_id])
    else:
        payload = json.dumps(
            [doc_id, subject_id, relation, object_id, evidence, chunk_index],
            ensure_ascii=False, separators=(",", ":"),
        )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def chunk_index_of(node_id: str, chunk_index_by_id: dict[str, int]) -> int:
    """Recover the chunk a node came from; the extractor prefixes ids with the chunk id."""
    prefix = node_id.split(":", 1)[0]
    return int(chunk_index_by_id.get(prefix, -1))


def extraction_relationship_id(doc_id: str, relationship: Any) -> str:
    payload = [doc_id, relationship.model_dump(mode="json")]
    return "relationship-" + hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def candidate_fact_id_for_audit(
    doc_id: str, subject_id: str, relation: str, object_id: str, evidence: str,
) -> str:
    """Mirror validation's stable pre-repair identity without adding a dependency."""
    payload = "|".join([doc_id, subject_id, relation, object_id, evidence or ""])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def is_expansion_relevant_fact(
    *,
    relation: str,
    object_id: str,
    subject_type: str,
    origin_entity: Optional[str],
) -> bool:
    """True when the fact is an upstream suppliesTo edge into the document origin entity."""
    if not origin_entity:
        return False
    if relation != "suppliesTo":
        return False
    if subject_type not in ("Company", "Mine"):
        return False
    return object_id == origin_entity


def graph_to_facts(
    graph: Neo4jGraph,
    document: dict[str, Any],
    chunk_index_by_id: dict[str, int],
    resolution: dict[str, dict[str, Any]],
    model: Optional[str] = None,
    audit=None,
) -> pd.DataFrame:
    """Build the facts table for one document."""
    extracted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    origin = document.get("origin_entity")

    for relationship in graph.relationships:
        relationship_id = extraction_relationship_id(document["doc_id"], relationship)
        subject = resolution.get(relationship.start_node_id)
        target = resolution.get(relationship.end_node_id)
        if subject is None or target is None:
            logger.warning(
                "dropping %s: an endpoint was not resolved to a canonical entity",
                relationship.type,
            )
            if audit is not None:
                audit.emit(
                    "relationship_lost_unresolved_endpoint",
                    stage="fact_construction",
                    extraction_relationship_id=relationship_id,
                    relation_type=relationship.type,
                    start_node_id=relationship.start_node_id,
                    end_node_id=relationship.end_node_id,
                    missing_resolution_ids=[
                        node_id for node_id in (
                            relationship.start_node_id, relationship.end_node_id,
                        ) if node_id not in resolution
                    ],
                    original_relationship=(
                        relationship.model_dump(mode="json")
                        if audit.capture_content else None
                    ),
                    action="excluded_before_candidate_fact",
                )
            continue
        expansion_relevant = is_expansion_relevant_fact(
            relation=relationship.type,
            object_id=target["canonical_id"],
            subject_type=subject["label"],
            origin_entity=origin,
        )
        evidence = (relationship.properties or {}).get("evidence", "")
        chunk_index = chunk_index_of(relationship.start_node_id, chunk_index_by_id)
        row = {
                "fact_id": fact_id(
                    document["doc_id"], subject["canonical_id"],
                    relationship.type, target["canonical_id"],
                    evidence=evidence, chunk_index=chunk_index,
                ),
                "doc_id": document["doc_id"],
                "chunk_index": chunk_index,
                "source_url": document.get("source_url"),
                "subject_id": subject["canonical_id"],
                "subject_type": subject["label"],
                "subject_name": subject["canonical_name"],
                "subject_mention": subject["mention"],
                "relation": relationship.type,
                "object_id": target["canonical_id"],
                "object_type": target["label"],
                "object_name": target["canonical_name"],
                "object_mention": target["mention"],
                "evidence": evidence,
                "model": model,
                "extracted_at": extracted_at,
                "validation_status": None,
                "ontology_valid": None,
                "evidence_validated": None,
                "retrieval_url": document.get("retrieval_url") or document.get("source_url"),
                "canonical_source_url": document.get("canonical_source_url"),
                "source_domain": document.get("source_domain"),
                "source_known": document.get("source_known"),
                "source_class": document.get("source_class"),
                "source_base_score": document.get("source_base_score"),
                "source_effective_score": document.get("source_effective_score"),
                "content_source": document.get("content_source"),
                "source_group_id": document.get("source_group_id"),
                "provider": document.get("provider"),
                "query": document.get("query"),
                "origin_entity": origin,
                "tier": document.get("tier"),
                "expansion_relevant": expansion_relevant,
                "direction_explicit": None,
                "original_evidence": evidence,
                "evidence_repair_attempted": False,
                "evidence_repair_status": "not_needed",
                "evidence_repair_reason": None,
            }
        rows.append(row)
        if audit is not None:
            candidate_id = candidate_fact_id_for_audit(
                str(row["doc_id"]), str(row["subject_id"]), str(row["relation"]),
                str(row["object_id"]), str(row["evidence"] or ""),
            )
            audit.emit(
                "candidate_fact_created",
                stage="fact_construction",
                candidate_fact_id=candidate_id,
                fact_id=row["fact_id"],
                extraction_relationship_id=relationship_id,
                chunk_id=str(relationship.start_node_id).split(":", 1)[0],
                chunk_index=chunk_index,
                subject_id=row["subject_id"], object_id=row["object_id"],
                subject_mention=row["subject_mention"] if audit.capture_content else None,
                object_mention=row["object_mention"] if audit.capture_content else None,
                subject_name=row["subject_name"] if audit.capture_content else None,
                object_name=row["object_name"] if audit.capture_content else None,
                predicate=row["relation"],
                original_evidence=evidence if audit.capture_content else None,
                originating_llm_call_ids=audit.contextual_llm_call_ids(
                    "extraction", "chunk_id",
                    str(relationship.start_node_id).split(":", 1)[0],
                ),
                creation_path="parsed_relationship_to_canonical_fact",
                status="candidate",
                created_at=extracted_at,
            )

    return pd.DataFrame(rows, columns=FACT_COLUMNS)


def upsert_facts_parquet(frame: pd.DataFrame, path: Path) -> int:
    """Append facts to the Parquet layer, replacing rows with the same fact_id.

    Returns the total row count after the write.
    """
    if frame.empty and not path.exists():
        atomic_write_parquet(frame, path)
        return 0
    if path.exists():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
    frame = frame.drop_duplicates(subset="fact_id", keep="last").reset_index(drop=True)
    atomic_write_parquet(frame, path)
    return len(frame)


def facts_for_document(path: Path, doc_id: str) -> pd.DataFrame:
    """Read back the facts of one document, used when resuming after a failed write."""
    if not path.exists():
        return pd.DataFrame(columns=FACT_COLUMNS)
    frame = pd.read_parquet(path)
    return frame[frame.doc_id == doc_id].reset_index(drop=True)


def upstream_companies(
    facts: pd.DataFrame, entity_ids: Iterable[str], tier_relations: Iterable[str]
) -> list[str]:
    """Names of companies that supply ``entity_id``, i.e. the next tier upstream.

    Direction is supplier --suppliesTo--> customer, so the upstream companies are the
    subjects of a tier-advancing relation whose object is this entity. Relations such as
    locatedIn, owns and produces are not in ``tier_relations`` and therefore never add
    supply-chain depth.

    When ``expansion_relevant`` is present, only those facts advance the frontier.
    """
    if facts.empty:
        return []
    matches = facts[
        facts.relation.isin(list(tier_relations))
        & facts.object_id.isin(list(entity_ids))
        & (facts.subject_type == "Company")
    ]
    if "expansion_relevant" in matches.columns:
        matches = matches[matches.expansion_relevant == True]  # noqa: E712
    return list(dict.fromkeys(matches.subject_name.tolist()))
