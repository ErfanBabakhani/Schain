"""Semantic edge aggregation for accepted facts.

Source reputation is informational metadata only.
Frontier eligibility does NOT use independent-source or multi-source thresholds.
"""

from __future__ import annotations

import hashlib
import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..checkpoint import atomic_write_parquet
from ..config import storage_path

logger = logging.getLogger(__name__)

EDGE_SUPPORT_COLUMNS = [
    "edge_key",
    "subject_id", "subject_name", "subject_type",
    "relation",
    "object_id", "object_name", "object_type",
    "fact_count", "document_count",
    "full_text_support_count",
    "max_source_score", "average_source_score",
    "quality_status", "frontier_eligible", "expansion_relevant",
    "direction_explicit",
    "frontier_reason", "frontier_rejection_reason",
    "has_supporting_conflict",
    "updated_at",
]


def edge_key(subject_id: str, relation: str, object_id: str) -> str:
    payload = "|".join([subject_id, relation, object_id])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _group_evidence(facts: pd.DataFrame) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {}
    for row in facts.to_dict(orient="records"):
        # Ignore legacy verification-injected pseudo-facts if any remain on disk.
        doc_id = str(row.get("doc_id") or "")
        if doc_id.startswith("verify:"):
            continue
        subject_id = str(row.get("subject_id") or "")
        object_id = str(row.get("object_id") or "")
        relation = str(row.get("relation") or "")
        if not subject_id or not object_id or not relation:
            continue
        key = edge_key(subject_id, relation, object_id)
        buckets.setdefault(key, []).append(row)
    return buckets


def _as_bool(value: Any, default: bool = False) -> bool:
    """Read nullable persisted flags without treating strings/NaN as true."""
    if value is None or value is pd.NA:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return default


def contextualize_facts(
    facts: pd.DataFrame, origins: dict[str, set[str]],
) -> pd.DataFrame:
    """Reuse accepted evidence for an explicitly associated document origin.

    This is an in-memory projection. Original retrieval provenance and Parquet facts
    are unchanged. Merely mentioning an entity never creates an association.
    """
    if facts is None or facts.empty:
        return facts
    frame = facts.copy()
    contexts: list[list[str]] = []
    relevant: list[bool] = []
    for row in frame.to_dict(orient="records"):
        matches = sorted(
            origin for origin in origins.get(str(row.get("doc_id") or ""), set())
            if origin == row.get("object_id")
            and row.get("relation") == "suppliesTo"
            and row.get("subject_type") in ("Company", "Mine")
        )
        contexts.append(matches)
        relevant.append(_as_bool(row.get("expansion_relevant")) or bool(matches))
    frame["expansion_origin_ids"] = contexts
    # Preserve unknown legacy flags when no new origin association exists.
    if "expansion_relevant" in frame:
        frame["expansion_relevant"] = [
            True if flag else original
            for flag, original in zip(relevant, frame["expansion_relevant"].tolist())
        ]
    elif any(relevant):
        frame["expansion_relevant"] = [True if flag else None for flag in relevant]
    return frame


def load_document_origins(config: dict[str, Any]) -> dict[str, set[str]]:
    """Read optional context associations without creating or modifying state."""
    path = storage_path(config, "state_db")
    if not path.exists():
        return {}
    connection = None
    try:
        from .. import state

        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return state.document_origins(connection)
    except sqlite3.Error as error:
        logger.warning("could not load document origins: %s", error)
        return {}
    finally:
        if connection is not None:
            connection.close()


def frontier_witness(row: dict[str, Any]) -> tuple[dict[str, Any], Optional[str], bool]:
    """All semantic gates must hold on the same accepted evidence witness."""
    from .validate import is_generic_company_phrase

    generic = row.get("subject_type") == "Company" and is_generic_company_phrase(
        str(row.get("subject_name") or row.get("subject_mention") or "")
    )
    reason, rejection, eligible = build_frontier_reason(
        relation=str(row.get("relation") or ""),
        subject_type=str(row.get("subject_type") or ""),
        ontology_valid=_as_bool(row.get("ontology_valid")),
        evidence_validated=_as_bool(row.get("evidence_validated")),
        expansion_relevant=_as_bool(row.get("expansion_relevant")),
        direction_explicit=_as_bool(row.get("direction_explicit")),
        generic_entity=generic,
    )
    grounded = row.get("evidence_quote_grounded")
    grounding_checked = _as_bool(grounded)
    grounding_failed = grounded is False or (isinstance(grounded, (int, float)) and grounded == 0)
    if not grounding_checked:
        eligible = False
        rejection = "evidence_quote_not_grounded" if grounding_failed else "evidence_quote_not_checked"
    reason.update({
        "evidence_quote_grounded": False if grounding_failed else (
            True if grounding_checked else None
        ),
        "parent_tier_valid": None,
        "parent_tier_check": "checked_when_attaching_to_frontier",
        "witness_fact_id": row.get("fact_id"),
        "witness_doc_id": row.get("doc_id"),
        "origin_entity": row.get("origin_entity"),
        "customer_id": row.get("object_id"),
    })
    return reason, rejection, eligible


def build_frontier_reason(
    *,
    relation: str,
    subject_type: str,
    ontology_valid: bool,
    evidence_validated: bool,
    expansion_relevant: bool,
    direction_explicit: bool,
    generic_entity: bool = False,
    parent_tier_valid: bool = True,
) -> tuple[dict[str, Any], Optional[str], bool]:
    """Return (frontier_reason, rejection_reason, frontier_eligible)."""
    origin_match = bool(expansion_relevant)
    named_supplier = subject_type in ("Company", "Mine") and not generic_entity
    relation_ok = relation == "suppliesTo"
    reason = {
        "origin_match": origin_match,
        "relation_explicit": relation_ok and ontology_valid and evidence_validated,
        "direction_explicit": bool(direction_explicit),
        "named_supplier": named_supplier,
        "generic_entity": bool(generic_entity),
        "parent_tier_valid": bool(parent_tier_valid),
        "ontology_valid": bool(ontology_valid),
        "evidence_validated": bool(evidence_validated),
        "expansion_relevant": origin_match,
    }
    eligible = (
        relation_ok
        and ontology_valid
        and evidence_validated
        and direction_explicit
        and origin_match
        and named_supplier
        and not generic_entity
        and parent_tier_valid
    )
    rejection: Optional[str] = None
    if not eligible:
        if not relation_ok:
            rejection = "not_supplies_to"
        elif generic_entity:
            rejection = "generic_supplier"
        elif not named_supplier:
            rejection = "unnamed_supplier"
        elif not ontology_valid:
            rejection = "ontology_invalid"
        elif not evidence_validated:
            rejection = "weak_relation_evidence"
        elif not direction_explicit:
            rejection = "ambiguous_supply_direction"
        elif not origin_match:
            rejection = "downstream_relation" if relation_ok else "origin_mismatch"
        elif not parent_tier_valid:
            rejection = "invalid_parent_tier"
        else:
            rejection = "frontier_gates_failed"
    return reason, rejection, eligible


def assign_quality_status(
    *,
    ontology_valid: bool,
    evidence_validated: bool,
    relation: str,
    subject_type: str,
    max_source_score: float = 0.0,
    has_conflict: bool = False,
    expansion_relevant: bool = False,
    direction_explicit: bool = True,
) -> tuple[str, bool, dict[str, Any], Optional[str]]:
    """Return (quality_status, frontier_eligible, frontier_reason, rejection).

    One ontology-valid + evidence-supported expansion-relevant suppliesTo edge with
    explicit direction is enough to advance the frontier. Source reputation never gates.
    """
    if has_conflict:
        reason, rejection, _ = build_frontier_reason(
            relation=relation,
            subject_type=subject_type,
            ontology_valid=ontology_valid,
            evidence_validated=evidence_validated,
            expansion_relevant=expansion_relevant,
            direction_explicit=direction_explicit,
        )
        return "contested", False, reason, rejection or "contested"

    status = "accepted"
    if max_source_score >= 85:
        status = "accepted_high_reputation"

    reason, rejection, eligible = build_frontier_reason(
        relation=relation,
        subject_type=subject_type,
        ontology_valid=ontology_valid,
        evidence_validated=evidence_validated,
        expansion_relevant=expansion_relevant,
        direction_explicit=direction_explicit,
    )
    return status, eligible, reason, rejection


def aggregate_edge_support(
    facts: pd.DataFrame,
    *,
    config: dict[str, Any],
    contradicting_edge_keys: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Build/refresh edge_support rows from accepted facts."""
    contradicting_edge_keys = contradicting_edge_keys or set()
    if facts is None or facts.empty:
        return pd.DataFrame(columns=EDGE_SUPPORT_COLUMNS)

    rows_out: list[dict[str, Any]] = []
    for key, group_rows in _group_evidence(facts).items():
        sample = group_rows[0]
        doc_ids = {
            str(r.get("doc_id"))
            for r in group_rows
            if r.get("doc_id") and not str(r.get("doc_id")).startswith("verify:")
        }
        # Counts and reputation summaries are per document, not per repeated quote.
        scores_by_doc: dict[str, float] = {}
        full_text_docs: set[str] = set()
        for item in group_rows:
            doc = str(item.get("doc_id") or "")
            if doc and item.get("content_source") == "full_text":
                full_text_docs.add(doc)
            try:
                score = float(item.get("source_effective_score"))
                if math.isfinite(score) and doc:
                    scores_by_doc[doc] = max(scores_by_doc.get(doc, score), score)
            except (TypeError, ValueError):
                pass
        scores = list(scores_by_doc.values())
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        full_text = len(full_text_docs)
        evaluations = [(row, *frontier_witness(row)) for row in group_rows]
        valid = [entry for entry in evaluations if entry[3]]
        # Pick one actual witness for the reason, never a union of unrelated gates.
        chosen = (valid or sorted(evaluations, key=lambda entry: sum(
            entry[1].get(field) is True for field in (
                "ontology_valid", "evidence_validated", "direction_explicit", "origin_match"
            )
        ), reverse=True))[0]
        _, frontier_reason, rejection, eligible = chosen
        frontier_reason = dict(frontier_reason)
        frontier_reason["eligible_witness_fact_ids"] = sorted({
            str(row.get("fact_id")) for row, *_ in valid if row.get("fact_id")
        })
        frontier_reason["eligible_witness_doc_ids"] = sorted({
            str(row.get("doc_id")) for row, *_ in valid if row.get("doc_id")
        })
        expansion_relevant = any(_as_bool(r.get("expansion_relevant")) for r in group_rows)
        direction_explicit = any(_as_bool(r.get("direction_explicit")) for r in group_rows)
        status = "accepted_high_reputation" if max_score >= 85 else "accepted"
        if key in contradicting_edge_keys:
            status, eligible, rejection = "contested", False, "contested"
        if eligible:
            logger.info(
                "frontier-eligible edge: %s --%s--> %s docs=%d status=%s",
                sample.get("subject_name"), sample.get("relation"),
                sample.get("object_name"), len(doc_ids), status,
            )
        rows_out.append(
            {
                "edge_key": key,
                "subject_id": sample.get("subject_id"),
                "subject_name": sample.get("subject_name"),
                "subject_type": sample.get("subject_type"),
                "relation": sample.get("relation"),
                "object_id": sample.get("object_id"),
                "object_name": sample.get("object_name"),
                "object_type": sample.get("object_type"),
                "fact_count": len(group_rows),
                "document_count": len(doc_ids),
                "full_text_support_count": full_text,
                "max_source_score": max_score,
                "average_source_score": avg_score,
                "quality_status": status,
                "frontier_eligible": bool(eligible),
                "expansion_relevant": bool(expansion_relevant),
                "direction_explicit": bool(direction_explicit),
                "frontier_reason": frontier_reason,
                "frontier_rejection_reason": rejection,
                "has_supporting_conflict": key in contradicting_edge_keys,
                "updated_at": _utcnow(),
            }
        )
    frame = pd.DataFrame(rows_out, columns=EDGE_SUPPORT_COLUMNS)
    return frame.sort_values(["relation", "subject_id", "object_id"]).reset_index(drop=True)


def upsert_edge_support(frame: pd.DataFrame, path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty and not path.exists():
        atomic_write_parquet(frame, path)
        return 0
    if path.exists() and not frame.empty:
        existing = pd.read_parquet(path)
        keep = existing[~existing.edge_key.isin(frame.edge_key)] if "edge_key" in existing.columns else existing
        frame = pd.concat([keep, frame], ignore_index=True)
    elif path.exists() and frame.empty:
        return len(pd.read_parquet(path))
    for column in EDGE_SUPPORT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[EDGE_SUPPORT_COLUMNS]
    frame = frame.drop_duplicates(subset="edge_key", keep="last").reset_index(drop=True)
    atomic_write_parquet(frame, path)
    return len(frame)


def load_edge_support(config: dict[str, Any]) -> pd.DataFrame:
    path = storage_path(config, "edge_support")
    if not path.exists():
        return pd.DataFrame(columns=EDGE_SUPPORT_COLUMNS)
    frame = pd.read_parquet(path)
    for column in EDGE_SUPPORT_COLUMNS:
        if column not in frame.columns:
            if column == "quality_status" and "verification_status" in frame.columns:
                frame[column] = frame["verification_status"].replace(
                    {
                        "candidate": "accepted",
                        "strong_candidate": "accepted_high_reputation",
                        "supported": "accepted",
                        "verified": "accepted_high_reputation",
                    }
                )
            else:
                frame[column] = None
    return frame


def refresh_edge_support(config: dict[str, Any]) -> pd.DataFrame:
    """Rebuild edge_support from accepted facts only (no verification evidence)."""
    facts_path = storage_path(config, "extracted_facts")
    if facts_path.exists():
        facts = pd.read_parquet(facts_path)
    else:
        facts = pd.DataFrame()
    facts = contextualize_facts(facts, load_document_origins(config))
    support = aggregate_edge_support(facts, config=config)
    path = storage_path(config, "edge_support")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(support, path)
    return support


def frontier_supplier_records(
    edge_support: pd.DataFrame,
    customer_ids: set[str],
    tier_relations: list[str],
) -> list[dict[str, Any]]:
    """Typed canonical upstream candidates, with interpretable expansion priority."""
    if edge_support is None or edge_support.empty or "suppliesTo" not in tier_relations:
        return []
    matches = edge_support[
        (edge_support.relation == "suppliesTo")
        & edge_support.object_id.isin(list(customer_ids))
        & edge_support.subject_type.isin(["Company", "Mine"])
        & (edge_support.frontier_eligible == True)  # noqa: E712
    ]
    result: dict[str, dict[str, Any]] = {}
    for row in matches.to_dict(orient="records"):
        docs = int(row.get("document_count") or 0)
        full = int(row.get("full_text_support_count") or 0)
        # One explicit valid source receives full base priority. Further evidence
        # is only a bounded tie-break; reputation never affects scheduling.
        priority = 100.0 + (10.0 if full else 0.0) + min(max(docs - 1, 0), 3)
        record = {
            "entity_id": str(row["subject_id"]),
            "entity_name": str(row["subject_name"]),
            "entity_type": str(row["subject_type"]),
            "customer_id": str(row["object_id"]),
            "edge_key": str(row["edge_key"]),
            "frontier_reason": row.get("frontier_reason"),
            "priority": priority,
            "priority_reason": {
                "explicit_valid_witness": True,
                "full_text_available": bool(full),
                "supporting_documents": docs,
                "additional_document_bonus": min(max(docs - 1, 0), 3),
            },
        }
        previous = result.get(record["entity_id"])
        if previous is None or record["priority"] > previous["priority"]:
            result[record["entity_id"]] = record
    return sorted(result.values(), key=lambda item: (-item["priority"], item["entity_id"]))


def frontier_eligible_suppliers(
    edge_support: pd.DataFrame,
    customer_ids: set[str],
    tier_relations: list[str],
) -> list[str]:
    """Legacy name-only API; expansion itself uses typed supplier records."""
    return list(dict.fromkeys(
        item["entity_name"] for item in frontier_supplier_records(
            edge_support, customer_ids, tier_relations
        )
    ))
