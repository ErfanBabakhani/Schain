"""Candidate-fact validation: ontology gate, optional repair, selective evidence checks.

Order: type plausibility → ontology → evidence and direction.
Source reputation is informational and never gates acceptance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from json_repair import repair_json

from ..audit import llm_call_context, prompt_hash
from ..config import storage_path
from ..checkpoint import atomic_write_parquet
from ..names import canonical_id, normalization_key
from .entity_types import company_type_hint
from .facts import FACT_COLUMNS, fact_id
from .relation_evidence import (
    check_supplies_to_direction,
    lexical_relation_gate,
    load_relation_evidence,
    render_relation_policy,
)

logger = logging.getLogger(__name__)

REJECTED_COLUMNS = [
    "candidate_fact_id", "doc_id", "chunk_index",
    "subject_id", "subject_name", "subject_type",
    "relation", "object_id", "object_name", "object_type",
    "evidence", "source_url", "canonical_source_url", "source_domain",
    "original_evidence", "evidence_repair_attempted",
    "evidence_repair_status", "evidence_repair_reason",
    "rejection_stage", "rejection_reason",
    "repair_attempted", "repair_result",
    "model", "created_at",
]

# FACT_COLUMNS already includes the additive quality/provenance fields.
QUALITY_FACT_COLUMNS = list(FACT_COLUMNS)


def _coalesce(value: Any, fallback: Any) -> Any:
    """Treat nullable Parquet/pandas scalars as absent without touching containers."""
    if value is None:
        return fallback
    try:
        if bool(pd.isna(value)):
            return fallback
    except (TypeError, ValueError):
        pass
    return value

ONTOLOGY_REPAIR_PROMPT = """\
You are repairing ONE ontology-invalid supply-chain relation.
Use ONLY the evidence below. Do not invent facts from outside knowledge.

Subject: {subject_name} ({subject_type})
Invalid relation: {relation}
Object: {object_name} ({object_type})
Evidence: {evidence}

Allowed patterns (source_type, relation, target_type):
{patterns}

Decide:
- "repair" with a different allowed relation if the evidence clearly supports it
  (example: Company suppliesTo Product may become Company produces Product)
- "reject" if the evidence does not support any allowed relation
- "keep" only if the original relation is actually allowed (should be rare)

Return JSON only:
{{"decision": "keep|repair|reject", "relation": "produces|null", "evidence_support": true|false}}
"""

EVIDENCE_BATCH_PROMPT = """\
You validate whether the quoted evidence entails EACH proposed supply-chain relation.
Use ONLY that candidate's evidence string (and treat subject/object names as the
exact endpoints). No outside knowledge. Do not approve because the broader document
is about a related topic or because another company in the same article fits better.

For each candidate, decide:
- "supports" only if the evidence EXPLICITLY supports THIS exact
  subject --relation--> object tuple
- "contradicts" if the evidence contradicts that exact tuple
- "unclear" if the evidence is insufficient for that exact tuple

Exact tuple entailment:
- A quote about Foryou Multimedia supplying Great Wall Motor does NOT support
  Crystal-Optech suppliesTo Great Wall Motor merely because Crystal-Optech appears
  elsewhere in the same document.
- Subject and object must be the actors in the evidence, not nearby names.

Relation meanings (do not stretch):
- suppliesTo: supplier → customer receiving supply inputs. NOT partnership,
  collaboration, investment, reseller/distributor/aftermarket sales of a brand's
  parts to shops/consumers, or "parts for Brand X vehicles" without Brand X as
  the receiving customer. Sample/evaluation/C-sample/prototype alone is unclear.
- owns: CURRENT ownership/control. Planned/announced/bid-to-acquire is unclear
  or contradicts owns.
- produces: actual manufacturing. Offering, selling, co-designing, or using is
  not produces unless manufacturing is established.
- locatedIn: the entity itself is based/has a facility there. Mentions of partner
  or project locations, or ambiguous "the company", are unclear.

If no ontology relation is supported for the pair: decision must be unclear or
contradicts — never invent a substitute relation.

Active relation-specific policy (positive cues are hints, never proof):
{relation_policy}

Candidates:
{candidates}

Return JSON only:
{{"results": [{{"candidate_fact_id": "...", "decision": "supports|contradicts|unclear"}}]}}
"""


def candidate_fact_id(doc_id: str, subject_id: str, relation: str, object_id: str, evidence: str) -> str:
    payload = "|".join([doc_id, subject_id, relation, object_id, evidence or ""])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def ontology_patterns(ontology: dict[str, Any]) -> set[tuple[str, str, str]]:
    patterns = set()
    for entry in ontology.get("patterns") or []:
        if len(entry) == 3:
            patterns.add((str(entry[0]), str(entry[1]), str(entry[2])))
    return patterns


# Lightweight generic-category phrases that must not become Company entities.
_GENERIC_COMPANY_EXACT = frozenset({
    "suppliers", "manufacturers", "vendors", "partners", "customers",
    "companies", "firms", "providers", "contractors",
    "oem", "oems", "automaker", "automakers", "chinese oem",
    "industry partner", "industry partners",
    "auto supplier", "major supplier", "battery supplier",
    "japanese supplier", "component manufacturer", "component supplier",
    "raw material supplier", "parts supplier", "key supplier",
})
_GENERIC_COMPANY_SUFFIXES = (
    " suppliers", " manufacturers", " vendors", " providers",
    " partners", " companies", " firms", " automakers",
    " supplier", " manufacturer", " vendor", " provider",
)
_GENERIC_COMPANY_PHRASES = (
    "chip suppliers", "battery suppliers", "component suppliers",
    "automotive suppliers", "semiconductor suppliers", "material suppliers",
    "raw material suppliers", "parts suppliers", "tier 1 suppliers",
    "tier one suppliers", "oem suppliers", "chinese oem",
    "auto supplier", "major supplier", "battery supplier",
    "japanese supplier", "component manufacturer",
)

# Legal / corporate tokens that indicate a real named organization.
_ORG_MARKERS = (
    "inc", "ltd", "llc", "corp", "corporation", "company", "co", "ag", "gmbh",
    "plc", "sa", "nv", "group", "holdings", "technologies", "systems",
)


def is_generic_company_phrase(name: Optional[str]) -> bool:
    """True for category phrases that are not a named organization.

    Blocks ``Auto supplier`` / ``major supplier`` while allowing legitimate names
    that contain Supplier/Manufacturing as part of a proper organization.
    """
    text = " ".join(str(name or "").split()).strip().casefold()
    if not text:
        return False
    if text in _GENERIC_COMPANY_EXACT or text in _GENERIC_COMPANY_PHRASES:
        return True
    if len(text) < 4:
        return False
    tokens = text.split()
    # Real orgs usually carry a legal/corporate marker or look like a proper name.
    has_org_marker = any(token.rstrip(".") in _ORG_MARKERS for token in tokens)
    for suffix in _GENERIC_COMPANY_SUFFIXES:
        if text.endswith(suffix) and " " in text and not has_org_marker:
            # "Bosch Manufacturing" keeps Manufacturing as a proper-name token with
            # capitalization historically; after casefold we rely on org markers /
            # multi-token proper nouns of length >= 3 that do not start with a
            # category adjective alone.
            head = text[: -len(suffix)].strip()
            if head in {
                "auto", "major", "battery", "japanese", "chinese", "korean",
                "component", "parts", "material", "raw material", "key",
                "tier 1", "tier one", "oem", "chip", "semiconductor",
            }:
                return True
            if len(head.split()) <= 2 and not has_org_marker and head in {
                "auto", "major", "battery", "japanese", "component",
            }:
                return True
            # Plural category forms without org markers remain generic.
            if suffix.strip() in {"suppliers", "manufacturers", "vendors", "providers"}:
                return True
    return False


def is_ontology_valid(
    subject_type: str, relation: str, object_type: str, ontology: dict[str, Any]
) -> bool:
    patterns = ontology_patterns(ontology)
    if not patterns:
        # Fall back to type lists only.
        return (
            subject_type in (ontology.get("entity_types") or [])
            and object_type in (ontology.get("entity_types") or [])
            and relation in (ontology.get("relation_types") or [])
        )
    return (subject_type, relation, object_type) in patterns


def _upsert_rejected(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty and not path.exists():
        atomic_write_parquet(pd.DataFrame(columns=REJECTED_COLUMNS), path)
        return
    if path.exists():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
    frame = frame.drop_duplicates(subset="candidate_fact_id", keep="last").reset_index(drop=True)
    atomic_write_parquet(frame, path)


def _reject_row(row: dict[str, Any], stage: str, reason: str, repair_attempted: bool = False,
                repair_result: Optional[str] = None) -> dict[str, Any]:
    return {
        "candidate_fact_id": row.get("candidate_fact_id") or candidate_fact_id(
            row.get("doc_id", ""), row.get("subject_id", ""), row.get("relation", ""),
            row.get("object_id", ""), row.get("evidence", ""),
        ),
        "doc_id": row.get("doc_id"),
        "chunk_index": row.get("chunk_index"),
        "subject_id": row.get("subject_id"),
        "subject_name": row.get("subject_name"),
        "subject_type": row.get("subject_type"),
        "relation": row.get("relation"),
        "object_id": row.get("object_id"),
        "object_name": row.get("object_name"),
        "object_type": row.get("object_type"),
        "evidence": row.get("evidence"),
        "original_evidence": _coalesce(row.get("original_evidence"), row.get("evidence")),
        "evidence_repair_attempted": bool(row.get("evidence_repair_attempted", False)),
        "evidence_repair_status": row.get("evidence_repair_status") or "not_needed",
        "evidence_repair_reason": row.get("evidence_repair_reason"),
        "source_url": row.get("source_url") or row.get("retrieval_url"),
        "canonical_source_url": row.get("canonical_source_url"),
        "source_domain": row.get("source_domain"),
        "rejection_stage": stage,
        "rejection_reason": reason,
        "repair_attempted": repair_attempted,
        "repair_result": repair_result,
        "model": row.get("model"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _repair_relation(llm, row: dict[str, Any], ontology: dict[str, Any]) -> Optional[str]:
    patterns = "\n".join(
        f"- {a} --{b}--> {c}" for a, b, c in sorted(ontology_patterns(ontology))
    )
    prompt = ONTOLOGY_REPAIR_PROMPT.format(
        subject_name=row.get("subject_name"),
        subject_type=row.get("subject_type"),
        relation=row.get("relation"),
        object_name=row.get("object_name"),
        object_type=row.get("object_type"),
        evidence=row.get("evidence") or "",
        patterns=patterns or "(none)",
    )
    try:
        with llm_call_context(
            prompt_template_id="ONTOLOGY_REPAIR_PROMPT",
            prompt_template_hash=prompt_hash(ONTOLOGY_REPAIR_PROMPT),
            candidate_fact_id=row.get("candidate_fact_id"),
            repair_type="ontology_relation",
        ):
            raw = llm.invoke(prompt).content
        payload = json.loads(repair_json(raw))
    except Exception as error:  # noqa: BLE001
        raise RuntimeError("ontology repair failed; document remains retryable") from error
    decision = str(payload.get("decision") or "").lower()
    if decision == "reject" or not payload.get("evidence_support", True):
        return None
    if decision == "keep":
        return row.get("relation")
    repaired = payload.get("relation")
    return str(repaired) if repaired else None


def _needs_evidence_check(row: dict[str, Any], config: dict[str, Any], repaired: bool) -> bool:
    evidence_cfg = (config.get("validation") or {}).get("evidence") or {}
    if evidence_cfg.get("enabled", True) is False:
        return False
    if repaired:
        return True
    relation = row.get("relation")
    if evidence_cfg.get("validate_tier_advancing_edges", True) and relation == "suppliesTo":
        if row.get("subject_type") in ("Company", "Mine") and row.get("object_type") == "Company":
            return True
    risky = set(
        evidence_cfg.get("validate_risky_relations")
        or ["owns", "locatedIn", "produces"]
    )
    return relation in risky


def _batch_evidence_validate(llm, rows: list[dict[str, Any]]) -> dict[str, str]:
    if not rows:
        return {}
    lines = []
    for row in rows:
        lines.append(
            json.dumps(
                {
                    "candidate_fact_id": row["candidate_fact_id"],
                    "subject": f"{row.get('subject_name')} ({row.get('subject_type')})",
                    "relation": row.get("relation"),
                    "object": f"{row.get('object_name')} ({row.get('object_type')})",
                    "evidence": row.get("evidence") or "",
                },
                ensure_ascii=False,
            )
        )
    policy = rows[0].get("_relation_policy") or load_relation_evidence()
    prompt = EVIDENCE_BATCH_PROMPT.format(
        candidates="\n".join(lines),
        relation_policy=render_relation_policy(policy, [str(row.get("relation") or "") for row in rows]),
    )
    try:
        with llm_call_context(
            prompt_template_id="EVIDENCE_BATCH_PROMPT",
            prompt_template_hash=prompt_hash(EVIDENCE_BATCH_PROMPT),
            candidate_fact_ids=[row["candidate_fact_id"] for row in rows],
        ):
            raw = llm.invoke(prompt).content
        payload = json.loads(repair_json(raw))
        results = payload.get("results", []) if isinstance(payload, dict) else []
    except Exception as error:  # noqa: BLE001
        # A service/parse failure is operational, not evidence against the fact.
        # Let the document checkpoint retry from EXTRACTED on resume.
        raise RuntimeError("evidence validation failed; document remains retryable") from error
    decisions: dict[str, str] = {}
    if not isinstance(results, list):
        raise ValueError("evidence validation response results must be a list")
    for item in results:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("candidate_fact_id") or "")
        decision = str(item.get("decision") or "unclear").lower()
        if decision not in ("supports", "contradicts", "unclear"):
            raise ValueError("invalid evidence decision; document must retry validation")
        if cid:
            decisions[cid] = decision
    missing = {row["candidate_fact_id"] for row in rows} - decisions.keys()
    if missing:
        raise ValueError("evidence validation response omitted candidate decisions; retry required")
    return decisions


def normalize_evidence_text(value: Any) -> str:
    """Match source quotes with Unicode NFKC and whitespace normalization only.

    Case, punctuation, negation and numbers remain significant. No fuzzy matching
    can turn a hallucinated paraphrase into source evidence.
    """
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _source_passages(source: str) -> list[str]:
    """Return source substrings at sentence/newline boundaries without rewriting them."""
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", source or "")
        if part.strip()
    ]


def _surface_present(passage: str, values: list[Any]) -> bool:
    normalized = normalize_evidence_text(passage).casefold()
    return any(
        normalize_evidence_text(value).casefold() in normalized
        for value in values
        if normalize_evidence_text(value)
    )


def _token_overlap(left: str, right: str) -> float:
    tokens_left = set(re.findall(r"\w+", normalize_evidence_text(left).casefold()))
    tokens_right = set(re.findall(r"\w+", normalize_evidence_text(right).casefold()))
    return len(tokens_left & tokens_right) / max(1, len(tokens_left | tokens_right))


def recover_evidence_quote(
    row: dict[str, Any],
    document_text: str,
    *,
    policy: dict[str, Any],
) -> tuple[Optional[str], str]:
    """Recover one exact source passage under strict endpoint/relation constraints.

    Similarity ranks already-semantic candidates only; it can never make a passage
    eligible by itself.  Ambiguous ties fail closed.
    """
    # Prefer the literal extracted mentions.  Falling back to a shorter canonical
    # name when a mention is present could confuse similarly named organizations
    # (for example, "BMW" versus "BMW Financial Services").
    subject_forms = [row.get("subject_mention") or row.get("subject_name")]
    object_forms = [row.get("object_mention") or row.get("object_name")]
    model_evidence = str(row.get("evidence") or "")
    if not (
        _surface_present(model_evidence, subject_forms)
        and _surface_present(model_evidence, object_forms)
    ):
        return None, "model_evidence_missing_endpoints"
    candidates: list[tuple[float, str]] = []
    for passage in _source_passages(document_text):
        if not _surface_present(passage, subject_forms):
            continue
        if not _surface_present(passage, object_forms):
            continue
        # Recovery is deliberately stricter than ordinary lexical screening.  A
        # passage containing an explicit negative must never be used to repair a
        # positive assertion, even if it also contains the expected verb/endpoints.
        if re.search(
            r"\b(?:not|never|neither|no longer|doesn't|does not|do not|did not|"
            r"isn't|is not|are not|was not|were not|won't|will not|cannot|can't)\b",
            normalize_evidence_text(passage).casefold(),
        ):
            continue
        lexical_ok, _reason = lexical_relation_gate(
            str(row.get("relation") or ""),
            passage,
            policy=policy,
        )
        if lexical_ok is not True:
            continue
        candidates.append((_token_overlap(str(row.get("evidence") or ""), passage), passage))
    if not candidates:
        return None, "no_source_passage_with_endpoints_and_relation_cue"
    candidates.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None, "ambiguous_source_passages"
    recovered = candidates[0][1]
    if normalize_evidence_text(recovered) not in normalize_evidence_text(document_text):
        return None, "recovered_passage_failed_exact_grounding"
    return recovered, "deterministic_endpoint_and_relation_match"


def validate_facts_frame(
    facts: pd.DataFrame,
    *,
    config: dict[str, Any],
    ontology: dict[str, Any],
    llm=None,
    document_text: Optional[str] = None,
    audit=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate candidate facts; return (accepted, rejected)."""
    if facts is None or facts.empty:
        empty_rej = pd.DataFrame(columns=REJECTED_COLUMNS)
        return pd.DataFrame(columns=QUALITY_FACT_COLUMNS), empty_rej

    validation = config.get("validation") or {}
    ontology_cfg = validation.get("ontology") or {}
    ontology_enabled = ontology_cfg.get("enabled", True)
    repair_enabled = ontology_cfg.get("repair_invalid_relation", True) and llm is not None
    max_repair = int(ontology_cfg.get("max_repair_attempts", 1))
    evidence_cfg = validation.get("evidence") or {}
    evidence_batch = int(evidence_cfg.get("batch_size", 10))
    policy = load_relation_evidence(evidence_cfg.get("policy_file"))

    # Normalize once per document; whitespace variation from PDF/text splitting is harmless.
    source_text = normalize_evidence_text(document_text) if document_text is not None else None
    require_quote = evidence_cfg.get("require_source_quote", True)
    recovery_enabled = evidence_cfg.get("recover_source_quote", True)

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    pending_evidence: list[dict[str, Any]] = []
    check_order = [
        "grounding", "entity_type", "generic_entity", "ontology",
        "lexical_evidence", "direction", "semantic_entailment",
    ]

    def trace_check(row: dict[str, Any], name: str, *, executed: bool,
                    decision: str, reason: str, details: Optional[dict[str, Any]] = None) -> None:
        if audit is None:
            return
        audit.emit(
            "validation_check",
            stage="validation",
            candidate_fact_id=row.get("candidate_fact_id"),
            fact_id=row.get("fact_id"),
            check=name,
            executed=executed,
            decision=decision,
            reason=reason,
            input_evidence=(row.get("evidence") if audit.capture_content else None),
            details=details or {},
            related_llm_call_ids=audit.contextual_llm_call_ids(
                "validation", "candidate_fact_id", row.get("candidate_fact_id"),
            ) + audit.contextual_llm_call_ids(
                "validation", "candidate_fact_ids", row.get("candidate_fact_id"),
            ),
        )

    def trace_skipped(row: dict[str, Any], after: str) -> None:
        start = check_order.index(after) + 1
        for name in check_order[start:]:
            trace_check(
                row, name, executed=False,
                decision="skipped_due_to_earlier_rejection",
                reason=f"rejected_at_{after}",
            )

    for raw in facts.to_dict(orient="records"):
        row = dict(raw)
        row["candidate_fact_id"] = candidate_fact_id(
            str(row.get("doc_id") or ""),
            str(row.get("subject_id") or ""),
            str(row.get("relation") or ""),
            str(row.get("object_id") or ""),
            str(row.get("evidence") or ""),
        )
        if audit is not None:
            audit.emit(
                "validation_candidate_started",
                stage="validation",
                candidate_fact_id=row["candidate_fact_id"],
                fact_id=row.get("fact_id"),
                subject_id=row.get("subject_id"), relation=row.get("relation"),
                object_id=row.get("object_id"),
                evidence=(row.get("evidence") if audit.capture_content else None),
            )
        row["original_evidence"] = _coalesce(
            row.get("original_evidence"), row.get("evidence") or "",
        )
        row["evidence_repair_attempted"] = False
        row["evidence_repair_status"] = "not_needed"
        row["evidence_repair_reason"] = None
        if row.get("validation_status") is None:
            row["validation_status"] = "accepted"
        if row.get("ontology_valid") is None:
            row["ontology_valid"] = True
        if row.get("evidence_validated") is None:
            row["evidence_validated"] = False
        if row.get("direction_explicit") is None:
            row["direction_explicit"] = None
        # Never inherit stale validation assertions from an input frame.
        row["ontology_valid"] = is_ontology_valid(
            str(row.get("subject_type")), str(row.get("relation")),
            str(row.get("object_type")), ontology,
        ) if ontology_enabled else False
        row["evidence_validated"] = False
        quote = normalize_evidence_text(row.get("evidence"))
        row["evidence_quote_grounded"] = bool(quote and quote in source_text) if source_text is not None else None
        if require_quote and source_text is not None and not row["evidence_quote_grounded"]:
            row["evidence_repair_attempted"] = bool(recovery_enabled and document_text)
            recovered = None
            recovery_reason = "recovery_disabled"
            if row["evidence_repair_attempted"]:
                recovered, recovery_reason = recover_evidence_quote(
                    row,
                    document_text or "",
                    policy=policy,
                )
            if recovered is None:
                row["evidence_repair_status"] = "failed"
                row["evidence_repair_reason"] = recovery_reason
                if audit is not None:
                    audit.emit(
                        "grounding_checked",
                        stage="grounding",
                        candidate_fact_id=row["candidate_fact_id"],
                        exact_match=False,
                        repair_attempted=row["evidence_repair_attempted"],
                        outcome="rejected",
                    )
                    audit.emit(
                        "evidence_repair_failed",
                        stage="grounding",
                        candidate_fact_id=row["candidate_fact_id"],
                        original_evidence=row["original_evidence"] if audit.capture_content else None,
                        reason=recovery_reason,
                    )
                rejected_rows.append(_reject_row(row, "grounding", "evidence_not_in_source"))
                trace_check(
                    row, "grounding", executed=True, decision="fail",
                    reason="evidence_not_in_source",
                    details={"repair_attempted": row["evidence_repair_attempted"],
                             "repair_reason": recovery_reason},
                )
                trace_skipped(row, "grounding")
                continue
            row["evidence"] = recovered
            row["evidence_quote_grounded"] = True
            row["evidence_repair_status"] = "succeeded"
            row["evidence_repair_reason"] = recovery_reason
            row["validation_status"] = "repaired"
            row["fact_id"] = fact_id(
                str(row.get("doc_id") or ""),
                str(row.get("subject_id") or ""),
                str(row.get("relation") or ""),
                str(row.get("object_id") or ""),
                recovered,
                row.get("chunk_index"),
            )
            if audit is not None:
                audit.emit(
                    "grounding_checked",
                    stage="grounding",
                    candidate_fact_id=row["candidate_fact_id"],
                    exact_match=False,
                    repair_attempted=True,
                    outcome="recovered_exact_source_passage",
                )
                audit.emit(
                    "evidence_repair_succeeded",
                    stage="grounding",
                    candidate_fact_id=row["candidate_fact_id"],
                    method="deterministic_source_extraction",
                    reason=recovery_reason,
                    original_evidence=row["original_evidence"] if audit.capture_content else None,
                    grounded_evidence=recovered if audit.capture_content else None,
                )
            trace_check(
                row, "grounding", executed=True, decision="pass",
                reason="recovered_exact_source_passage",
                details={"repair_attempted": True, "repair_reason": recovery_reason},
            )
        elif audit is not None:
            audit.emit(
                "grounding_checked",
                stage="grounding",
                candidate_fact_id=row["candidate_fact_id"],
                exact_match=bool(row["evidence_quote_grounded"]),
                repair_attempted=False,
                outcome=("grounded" if row["evidence_quote_grounded"] else "not_checked"),
            )
        if not (require_quote and source_text is not None and not quote):
            if row.get("evidence_repair_status") != "succeeded":
                trace_check(
                    row, "grounding",
                    executed=bool(require_quote and source_text is not None),
                    decision=("pass" if row.get("evidence_quote_grounded") else "not_applicable"),
                    reason=("exact_source_quote" if row.get("evidence_quote_grounded")
                            else "source_quote_check_not_required"),
                )
        evidence_repaired = row["evidence_repair_status"] == "succeeded"
        repaired = evidence_repaired

        # Defense for both the production resolver and direct/legacy fact callers.
        type_rejected = False
        for side in ("subject", "object"):
            surface = str(row.get(f"{side}_mention") or row.get(f"{side}_name") or "")
            target_type, type_reason = company_type_hint(surface)
            if row.get(f"{side}_type") == "Product" and target_type == "Product" and row.get("relation") == "suppliesTo":
                # Also covers a model already repaired by resolve_entities.
                rejected_rows.append(_reject_row(row, "entity_type", f"invalid_product_supply_role:{side}"))
                type_rejected = True
                trace_check(row, "entity_type", executed=True, decision="fail",
                            reason=f"invalid_product_supply_role:{side}")
                break
            if row.get(f"{side}_type") != "Company":
                continue
            if target_type:
                logger.info("entity type repair %s: Company -> %s (%s)", surface, target_type, type_reason)
                row[f"{side}_type"] = target_type
                row[f"{side}_name"] = surface
                row[f"{side}_id"] = canonical_id(target_type, surface)
                row["validation_status"] = "repaired"
                repaired = True
                # Type repair cannot turn supplying a model into manufacturing it.
                if row.get("relation") == "suppliesTo":
                    rejected_rows.append(_reject_row(row, "entity_type", f"{type_reason}:{side}",
                                                     repair_attempted=True, repair_result=target_type))
                    type_rejected = True
                    trace_check(row, "entity_type", executed=True, decision="fail",
                                reason=f"{type_reason}:{side}",
                                details={"repair_attempted": True,
                                         "proposed_type": target_type})
                    break
            elif type_reason:
                rejected_rows.append(_reject_row(row, "entity_type", f"{type_reason}:{side}"))
                type_rejected = True
                trace_check(row, "entity_type", executed=True, decision="fail",
                            reason=f"{type_reason}:{side}")
                break
        if type_rejected:
            trace_skipped(row, "entity_type")
            continue
        trace_check(row, "entity_type", executed=True, decision="pass",
                    reason="entity_types_compatible")
        if repaired:
            row["fact_id"] = fact_id(str(row.get("doc_id") or ""), row["subject_id"],
                row["relation"], row["object_id"], row.get("evidence"), row.get("chunk_index"))
            row["ontology_valid"] = is_ontology_valid(row["subject_type"], row["relation"],
                                                      row["object_type"], ontology)

        # Reject generic category phrases labelled as Company (auditable rejection).
        generic_side = None
        generic_name = None
        for side, etype, ename in (
            ("subject", row.get("subject_type"), row.get("subject_name")),
            ("object", row.get("object_type"), row.get("object_name")),
        ):
            if etype == "Company" and is_generic_company_phrase(ename):
                generic_side = side
                generic_name = ename
                break
        if generic_side is not None:
            rejected_rows.append(
                _reject_row(
                    row, "generic_entity",
                    f"generic_company_phrase:{generic_side}:{generic_name}",
                )
            )
            trace_check(row, "generic_entity", executed=True, decision="fail",
                        reason=f"generic_company_phrase:{generic_side}:{generic_name}")
            trace_skipped(row, "generic_entity")
            continue
        trace_check(row, "generic_entity", executed=True, decision="pass",
                    reason="named_entities")

        if ontology_enabled and not is_ontology_valid(
            str(row.get("subject_type")), str(row.get("relation")),
            str(row.get("object_type")), ontology,
        ):
            row["ontology_valid"] = False
            new_relation = None
            if repair_enabled and max_repair >= 1:
                calls_before = set(audit.llm_call_ids("validation")) if audit is not None else set()
                original_relation = row.get("relation")
                new_relation = _repair_relation(llm, row, ontology)
                repaired = True
                if audit is not None:
                    audit.emit(
                        "ontology_repair_attempted", stage="validation",
                        candidate_fact_id=row.get("candidate_fact_id"),
                        repair_trigger="ontology_invalid",
                        original_fact=(dict(row) if audit.capture_content else None),
                        original_relation=original_relation,
                        proposed_relation=new_relation,
                        attempt_number=1,
                        accepted=bool(new_relation and is_ontology_valid(
                            str(row.get("subject_type")), str(new_relation),
                            str(row.get("object_type")), ontology,
                        )),
                        related_llm_call_ids=[
                            call_id for call_id in audit.llm_call_ids("validation")
                            if call_id not in calls_before
                        ],
                    )
            if new_relation and is_ontology_valid(
                str(row.get("subject_type")), new_relation,
                str(row.get("object_type")), ontology,
            ):
                logger.debug(
                    "repaired %s --%s--> %s into %s",
                    row.get("subject_name"), row.get("relation"),
                    row.get("object_name"), new_relation,
                )
                row["relation"] = new_relation
                row["fact_id"] = fact_id(
                    str(row.get("doc_id") or ""), str(row.get("subject_id") or ""),
                    new_relation, str(row.get("object_id") or ""),
                    str(row.get("evidence") or ""), row.get("chunk_index"),
                )
                row["ontology_valid"] = True
                row["validation_status"] = "repaired"
                trace_check(
                    row, "ontology", executed=True, decision="pass",
                    reason="ontology_relation_repaired",
                    details={"repair_attempted": True, "repaired_relation": new_relation},
                )
            else:
                rejected_rows.append(
                    _reject_row(
                        row, "ontology", "ontology_invalid",
                        repair_attempted=repaired,
                        repair_result=new_relation,
                    )
                )
                trace_check(
                    row, "ontology", executed=True, decision="fail",
                    reason="ontology_invalid",
                    details={"repair_attempted": repaired, "repair_result": new_relation},
                )
                trace_skipped(row, "ontology")
                continue
        else:
            trace_check(
                row, "ontology", executed=ontology_enabled,
                decision="pass" if ontology_enabled else "not_applicable",
                reason="ontology_pattern_allowed" if ontology_enabled else "ontology_check_disabled",
            )

        # Deterministic relation-cue gate (assists; does not alone accept).
        lexical_ok, lexical_reason = lexical_relation_gate(
            str(row.get("relation") or ""),
            str(row.get("evidence") or ""),
            policy=policy,
        )
        trace_check(
            row, "lexical_evidence", executed=True,
            decision="pass" if lexical_ok is True else "fail" if lexical_ok is False else "indeterminate",
            reason=lexical_reason,
        )
        if lexical_ok is False:
            rejected_rows.append(_reject_row(row, "evidence", lexical_reason))
            trace_skipped(row, "lexical_evidence")
            continue

        # Directional suppliesTo: deterministic first; LLM adjudicates genuine ambiguity.
        if row.get("relation") == "suppliesTo":
            direction_ok, direction_reason = check_supplies_to_direction(
                str(row.get("subject_name") or ""),
                str(row.get("object_name") or ""),
                str(row.get("evidence") or ""),
                policy=policy,
                subject_aliases=[
                    str(row.get("subject_mention") or ""),
                    str(row.get("subject_name") or ""),
                ],
                object_aliases=[
                    str(row.get("object_mention") or ""),
                    str(row.get("object_name") or ""),
                ],
            )
            row["direction_explicit"] = bool(direction_ok)
            if not direction_ok and direction_reason == "ambiguous_supply_direction":
                retrieval_cfg = (config.get("expansion") or {}).get("retrieval") or {}
                if retrieval_cfg.get("semantic_direction_adjudication", True) and llm is not None:
                    from ..llm.adjudication import adjudicate_supply_direction
                    context = ""
                    if source_text and quote:
                        idx = source_text.find(quote)
                        if idx >= 0:
                            context = source_text[max(0, idx - 240): idx + len(quote) + 240]
                    verdict = adjudicate_supply_direction(
                        llm, config,
                        subject=str(row.get("subject_name") or ""),
                        obj=str(row.get("object_name") or ""),
                        evidence=str(row.get("evidence") or ""),
                        context=context,
                    )
                    if verdict.get("status") == "resolved" and verdict.get("supported"):
                        supplier = normalization_key(str(verdict.get("supplier") or ""))
                        customer = normalization_key(str(verdict.get("customer") or ""))
                        subj = normalization_key(str(row.get("subject_name") or ""))
                        objn = normalization_key(str(row.get("object_name") or ""))
                        # Accept only when adjudicated roles match the proposed edge.
                        if supplier and customer and supplier == subj and customer == objn:
                            direction_ok = True
                            direction_reason = "llm_direction_resolved"
                            row["direction_explicit"] = True
                        elif supplier and customer and supplier == objn and customer == subj:
                            # Flip subject/object to match adjudicated direction.
                            for a, b in (
                                ("subject_name", "object_name"),
                                ("subject_id", "object_id"),
                                ("subject_type", "object_type"),
                                ("subject_mention", "object_mention"),
                            ):
                                row[a], row[b] = row[b], row[a]
                            direction_ok = True
                            direction_reason = "llm_direction_flipped"
                            row["direction_explicit"] = True
                            row["validation_status"] = "repaired"
                    elif verdict.get("status") == "unsupported":
                        direction_reason = "ambiguous_supply_direction"
            trace_check(
                row, "direction", executed=True,
                decision="pass" if direction_ok else "fail", reason=direction_reason,
            )
            if not direction_ok:
                rejected_rows.append(_reject_row(row, "evidence", direction_reason))
                trace_skipped(row, "direction")
                continue
        else:
            row["direction_explicit"] = True
            trace_check(row, "direction", executed=False, decision="not_applicable",
                        reason="relation_is_not_suppliesTo")

        if _needs_evidence_check(row, config, repaired=repaired or row.get("validation_status") == "repaired"):
            row["_relation_policy"] = policy
            pending_evidence.append(row)
        else:
            trace_check(row, "semantic_entailment", executed=False,
                        decision="not_applicable", reason="selective_validation_not_required")
            row["evidence_validated"] = False
            row["validation_status"] = row.get("validation_status") or "accepted"
            accepted_rows.append(row)

    # Batched evidence checks.
    for start in range(0, len(pending_evidence), max(1, evidence_batch)):
        batch = pending_evidence[start:start + max(1, evidence_batch)]
        if llm is None:
            # Retain deterministic-only candidates for offline inspection without
            # asserting model entailment or allowing them to pass the frontier gate.
            for row in batch:
                if row.get("evidence_repair_status") == "succeeded":
                    row["evidence_repair_status"] = "failed"
                    row["evidence_repair_reason"] = "semantic_validation_unavailable"
                    rejected_rows.append(
                        _reject_row(
                            row,
                            "grounding",
                            "evidence_repair_semantic_unverified",
                        )
                    )
                    trace_check(
                        row, "semantic_entailment", executed=False, decision="fail",
                        reason="evidence_repair_semantic_unverified",
                    )
                    continue
                trace_check(row, "semantic_entailment", executed=False,
                            decision="not_instrumented",
                            reason="semantic_validator_unavailable")
                row["evidence_validated"] = False
                row["validation_status"] = "unchecked"
                accepted_rows.append(row)
            continue
        decisions = _batch_evidence_validate(llm, batch)
        for row in batch:
            decision = decisions.get(row["candidate_fact_id"], "unclear")
            if decision == "supports":
                trace_check(row, "semantic_entailment", executed=True,
                            decision="pass", reason="evidence_entailed")
                row["evidence_validated"] = True
                row["validation_status"] = row.get("validation_status") or "accepted"
                accepted_rows.append(row)
            elif decision == "unclear":
                # Bounded second-pass semantic judgment for grounded, named, ontology-valid facts.
                from ..llm.adjudication import second_pass_evidence
                second = second_pass_evidence(llm, config, row)
                if second == "YES":
                    trace_check(
                        row, "semantic_entailment", executed=True,
                        decision="pass", reason="second_pass_evidence_entailed",
                        details={"first_pass": "unclear", "second_pass": second},
                    )
                    row["evidence_validated"] = True
                    row["validation_status"] = row.get("validation_status") or "accepted"
                    accepted_rows.append(row)
                else:
                    rejection_reason = (
                        "evidence_not_entailed"
                        if second == "AMBIGUOUS" else "evidence_contradicts"
                    )
                    trace_check(
                        row, "semantic_entailment", executed=True,
                        decision="fail", reason=rejection_reason,
                        details={"first_pass": "unclear", "second_pass": second},
                    )
                    rejected_rows.append(
                        _reject_row(
                            row, "evidence", rejection_reason,
                        )
                    )
            else:
                trace_check(
                    row, "semantic_entailment", executed=True, decision="fail",
                    reason=("evidence_not_entailed" if decision == "unclear"
                            else "evidence_contradicts"),
                )
                rejected_rows.append(
                    _reject_row(row, "evidence", "evidence_contradicts")
                )

    accepted = pd.DataFrame(accepted_rows)
    if not accepted.empty:
        for column in QUALITY_FACT_COLUMNS:
            if column not in accepted.columns:
                accepted[column] = None
        accepted = accepted[QUALITY_FACT_COLUMNS]
    else:
        accepted = pd.DataFrame(columns=QUALITY_FACT_COLUMNS)
    rejected = pd.DataFrame(rejected_rows, columns=REJECTED_COLUMNS)
    repaired_n = (
        int((accepted.validation_status == "repaired").sum()) if not accepted.empty else 0
    )
    if len(accepted) or len(rejected):
        logger.info(
            "document validation accepted=%d rejected=%d repaired=%d",
            len(accepted), len(rejected), repaired_n,
        )
    return accepted, rejected


def persist_rejected(config: dict[str, Any], rejected: pd.DataFrame) -> None:
    if rejected is None or rejected.empty:
        return
    path = storage_path(config, "rejected_facts")
    _upsert_rejected(rejected, path)
