"""Expand-run diagnostics: funnel drops, fetch failures, growth, discovery yield.

Telemetry only — never gates frontier admission or fact acceptance.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Documents excluded after relevance / before Prompt-1 spend.
PROMPT1_DROP_REASONS = (
    "duplicate_url",
    "duplicate_content",
    "downstream_only",
    "weak_origin_match",
    "navigation_only",
    "ambiguous_low_rank",
    "target_reached",
    "entity_max_reached",
    "global_cap",
    "runtime_reserve",
    "diminishing_returns",
    "beam_excluded",
    "fetch_failure",
    "candidate_unusable",
    "already_processed",
    "zero_yield_rescue_skip",
    "other",
)

FETCH_FAILURE_REASONS = (
    "wrapper_decode_failed",
    "google_wrapper_decode_failed",  # alias retained for older manifests
    "publisher_url_missing",
    "publisher_domain_missing",
    "title_lookup_attempted",
    "title_lookup_no_result",
    "title_lookup_mismatch",
    "title_lookup_low_similarity",  # alias → title_lookup_mismatch
    "domain_mismatch",
    "title_lookup_domain_mismatch",  # alias → domain_mismatch
    "redirect_failed",
    "http_401",
    "http_403",
    "http_404",
    "http_429",
    "timeout",
    "tls_error",
    "unsupported_content_type",
    "body_too_short",
    "body_extraction_failed",
    "bot_block_or_challenge",
    "canonical_unresolved",
    "other",
    "unknown_fetch_error",
)

REJECTION_CATEGORIES = (
    "ontology_invalid",
    "entity_type_invalid",
    "entity_repair_failed",
    "generic_entity",
    "quote_not_grounded",
    "relation_not_supported",
    "ambiguous_supply_direction",
    "wrong_direction",
    "origin_mismatch",
    "evidence_false",
    "evidence_unclear",
    "insufficient_relation_language",
    "operational_error",
    "other",
)


def classify_fetch_error(
    error: Optional[str],
    *,
    http_status: Optional[int] = None,
    is_wrapper: bool = False,
    title_lookup_attempted: bool = False,
) -> str:
    """Map raw fetch/recovery codes onto stable telemetry categories."""
    raw = (error or "").strip()
    code = raw.upper()
    lowered = raw.casefold()
    if http_status == 401 or code in ("HTTP_401",):
        return "http_401"
    if http_status == 403 or code in ("HTTP_403",):
        return "http_403"
    if http_status == 404 or code in ("HTTP_404",):
        return "http_404"
    if http_status == 429 or code in ("HTTP_429",):
        return "http_429"
    if code in ("TIMEOUT",) or lowered == "timeout":
        return "timeout"
    if lowered in ("title_lookup_no_result",):
        return "title_lookup_no_result"
    if lowered in ("title_lookup_low_similarity", "title_lookup_mismatch"):
        return "title_lookup_mismatch"
    if lowered in ("title_lookup_domain_mismatch", "domain_mismatch"):
        return "domain_mismatch"
    if lowered in ("title_lookup_attempted",):
        return "title_lookup_attempted"
    if lowered in ("wrapper_decode_failed", "google_wrapper_decode_failed"):
        return "wrapper_decode_failed"
    if code in ("CANONICAL_UNRESOLVED",) or lowered == "canonical_unresolved":
        return "canonical_unresolved" if not title_lookup_attempted else "wrapper_decode_failed"
    if code in ("EMPTY_CONTENT",):
        return "body_too_short"
    if code in ("PARSE_FAILED",):
        return "body_extraction_failed"
    if "TLS" in code or "SSL" in code:
        return "tls_error"
    if "CHALLENGE" in code or "CAPTCHA" in code or "CLOUDFLARE" in code:
        return "bot_block_or_challenge"
    if is_wrapper and not code:
        return "wrapper_decode_failed"
    if code:
        return "other"
    return "other"


def map_rejection_reason(reason: Optional[str]) -> str:
    text = str(reason or "").strip()
    if not text:
        return "other"
    lowered = text.casefold()
    mapping = {
        "ontology_invalid": "ontology_invalid",
        "generic_entity": "generic_entity",
        "generic_company_phrase": "generic_entity",
        "ambiguous_supply_direction": "ambiguous_supply_direction",
        "insufficient_relation_language": "insufficient_relation_language",
        "evidence_not_entailed": "evidence_unclear",
        "evidence_contradicts": "evidence_false",
        "quote_not_grounded": "quote_not_grounded",
        "relation_not_supported": "relation_not_supported",
        "wrong_direction": "wrong_direction",
        "origin_mismatch": "origin_mismatch",
        "entity_type_invalid": "entity_type_invalid",
        "entity_repair_failed": "entity_repair_failed",
    }
    for key, category in mapping.items():
        if key in lowered:
            return category
    if "repair" in lowered:
        return "entity_repair_failed"
    if "ontology" in lowered:
        return "ontology_invalid"
    return "other"


def rejection_category_counts(rejected_frame) -> dict[str, int]:
    counts = Counter({name: 0 for name in REJECTION_CATEGORIES})
    if rejected_frame is None or getattr(rejected_frame, "empty", True):
        return dict(counts)
    column = "rejection_reason" if "rejection_reason" in rejected_frame.columns else None
    if not column:
        return dict(counts)
    for value in rejected_frame[column].tolist():
        counts[map_rejection_reason(value)] += 1
    return dict(counts)


def empty_prompt1_drop_counts() -> dict[str, int]:
    return {name: 0 for name in PROMPT1_DROP_REASONS}


def empty_fetch_failure_counts() -> dict[str, int]:
    return {name: 0 for name in FETCH_FAILURE_REASONS}


def graph_growth_snapshot(
    *,
    total_nodes: int,
    tiered_nodes: int,
    context_nodes: int,
    expansion_edges: int,
    new_supplier_entities: int,
    frontier_size: int,
    documents_processed: int,
    accepted_facts: int,
    new_suppliers_by_seed: Optional[dict[str, int]] = None,
    total_edges: int = 0,
    supply_chain_core_edges: int = 0,
    nodes_merged_neo4j: int = 0,
) -> dict[str, Any]:
    docs = max(1, documents_processed)
    scale = 10.0 / docs
    return {
        "total_nodes": total_nodes,
        "tiered_nodes": tiered_nodes,
        "context_nodes": context_nodes,
        "total_edges": total_edges,
        "supply_chain_core_edges": supply_chain_core_edges,
        "expansion_edges": expansion_edges,
        "expansion_relevant_edges": expansion_edges,
        "new_supplier_entities": new_supplier_entities,
        "new_suppliers_by_seed": new_suppliers_by_seed or {},
        "frontier_size": frontier_size,
        "documents_processed": documents_processed,
        "accepted_facts": accepted_facts,
        "nodes_merged_neo4j": nodes_merged_neo4j,
        "accepted_facts_per_10_docs": round(accepted_facts * scale, 3),
        "new_suppliers_per_10_docs": round(new_supplier_entities * scale, 3),
        "expansion_edges_per_10_docs": round(expansion_edges * scale, 3),
    }


def premature_discovery_exhaustion_warning(
    *,
    global_cap: int,
    documents_processed: int,
    runtime_max_minutes: float,
    elapsed_seconds: float,
    stop_reason: Optional[str],
) -> Optional[dict[str, Any]]:
    """Diagnostic only — never weakens evidence or auto-expands budgets."""
    if stop_reason != "frontier_empty":
        return None
    if global_cap < 250:
        return None
    if documents_processed >= 0.35 * global_cap:
        return None
    max_seconds = float(runtime_max_minutes or 0) * 60
    if max_seconds <= 0 or elapsed_seconds >= 0.30 * max_seconds:
        return None
    warning = {
        "code": "PREMATURE_DISCOVERY_EXHAUSTION",
        "message": (
            "Large experiment finished early with substantial document/runtime "
            "budget remaining while stop_reason=frontier_empty. Investigate "
            "escalation/rescue and candidate coverage — do not loosen validation."
        ),
        "global_cap": global_cap,
        "documents_processed": documents_processed,
        "runtime_max_minutes": runtime_max_minutes,
        "elapsed_seconds": elapsed_seconds,
        "stop_reason": stop_reason,
        "diagnostic_only": True,
    }
    logger.warning(
        "PREMATURE_DISCOVERY_EXHAUSTION docs=%s/%s elapsed=%ss stop=%s",
        documents_processed, global_cap, round(elapsed_seconds, 1), stop_reason,
    )
    return warning


def low_discovery_yield_warning(
    *,
    seed_count: int,
    documents_processed: int,
    tiered_nodes: int,
    expansion_edges: int,
    zero_yield_origins: int,
    heuristic_nodes: int = 60,
    min_docs_for_warning: int = 24,
) -> Optional[dict[str, Any]]:
    """Diagnostic-only. Never changes pipeline behavior."""
    if seed_count < 2 or documents_processed < min_docs_for_warning:
        return None
    if tiered_nodes >= heuristic_nodes // 2 and expansion_edges >= seed_count:
        return None
    if tiered_nodes >= heuristic_nodes:
        return None
    warning = {
        "code": "LOW_DISCOVERY_YIELD",
        "message": (
            "Processed many documents for several major seeds but produced few "
            "supply-chain entities/edges. Investigate retrieval recall, rescue, "
            "full-text recovery, and ranking — do not loosen validation."
        ),
        "heuristic_nodes": heuristic_nodes,
        "processed_docs": documents_processed,
        "tiered_nodes": tiered_nodes,
        "expansion_edges": expansion_edges,
        "zero_yield_origins": zero_yield_origins,
        "diagnostic_only": True,
    }
    logger.warning(
        "LOW_DISCOVERY_YIELD docs=%s tiered_nodes=%s expansion_edges=%s zero_yield_origins=%s",
        documents_processed, tiered_nodes, expansion_edges, zero_yield_origins,
    )
    return warning
