"""Search escalation stages for zero-yield / deepening expansion."""

from __future__ import annotations

import logging
from typing import Any, Optional

from .queries import generate_queries
from ..llm.query_planner import plan_upstream_queries

logger = logging.getLogger(__name__)

STAGE_A = "directional"
STAGE_B = "broad"
STAGE_C = "component"
STAGE_D = "llm_planned"
STAGE_E = "official_pdf"
STAGES = (STAGE_A, STAGE_B, STAGE_C, STAGE_D, STAGE_E)

COMPONENT_TEMPLATES = (
    '"{entity}" {category} supplier',
    '"{entity}" {category} sourcing',
    '"{entity}" purchases {category}',
)

OFFICIAL_TEMPLATES = (
    '"{entity}" suppliers filetype:pdf',
    '"{entity}" "annual report" suppliers',
    '"{entity}" "sustainability report" supply',
    '"{entity}" "supplier list"',
    '"{entity}" ESG suppliers',
    '"{entity}" procurement report suppliers',
)


def stage_for_query(template_or_query: str) -> str:
    text = (template_or_query or "").casefold()
    if any(token in text for token in ("filetype:pdf", "annual report", "sustainability", "esg", "procurement report")):
        return STAGE_E
    if any(token in text for token in ("supply chain", "supplier list", "key suppliers", "major suppliers", '" suppliers')):
        return STAGE_B
    if "supplies" in text or "supplier to" in text or "sources from" in text or "purchases from" in text or "procurement supplier" in text:
        return STAGE_A
    return STAGE_C


def build_escalation_queries(
    config: dict[str, Any],
    entity_name: str,
    tier: int,
    *,
    llm=None,
    stages: Optional[list[str]] = None,
    used_queries: Optional[list[str]] = None,
    entity_type: str = "Company",
    aliases: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Return query specs tagged with ``escalation_stage`` and ``selection_family``."""
    wanted = set(stages or STAGES)
    used = used_queries or []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(query: str, stage: str, providers: Optional[list[str]] = None) -> None:
        norm = " ".join(query.split()).casefold()
        if not norm or norm in seen:
            return
        seen.add(norm)
        defaults = ((config.get("expansion") or {}).get("search") or {}).get("defaults") or {}
        out.append({
            "query": query,
            "template": query,
            "escalation_stage": stage,
            "selection_family": stage,
            "providers": providers or list(defaults.get("providers") or ["google_news", "gdelt", "general_web", "wikipedia"]),
            "max_results": int(defaults.get("max_results") or 14),
            "max_results_by_provider": dict(defaults.get("max_results_by_provider") or {}),
        })

    if STAGE_A in wanted or STAGE_B in wanted:
        for spec in generate_queries(config, entity_name, tier):
            stage = stage_for_query(spec.get("template") or spec.get("query") or "")
            if stage not in wanted and not (
                (stage == STAGE_A and STAGE_A in wanted) or (stage == STAGE_B and STAGE_B in wanted)
            ):
                # Map unknown templates to A/B by content.
                stage = STAGE_B if "supplier" in (spec.get("query") or "").casefold() else STAGE_A
            if stage not in wanted:
                continue
            entry = {**spec, "escalation_stage": stage, "selection_family": stage}
            norm = " ".join(entry["query"].split()).casefold()
            if norm in seen:
                continue
            seen.add(norm)
            out.append(entry)
        retrieval = (config.get("expansion") or {}).get("retrieval") or {}
        alias_limit = max(0, int(retrieval.get("alias_query_max", 0)))
        initial_limit = max(0, int(retrieval.get("max_initial_query_variants", 0)))
        alias_queries = []
        for alias in aliases or []:
            if alias.casefold() == entity_name.casefold():
                continue
            alias_queries.extend((
                f'"supplier to" "{alias}"',
                f'"{alias}" procurement suppliers',
            ))
        for query in alias_queries[:alias_limit]:
            if initial_limit and len(out) >= initial_limit:
                break
            _add(query, STAGE_A)

    if STAGE_C in wanted or STAGE_D in wanted:
        plan = plan_upstream_queries(
            llm=llm if STAGE_D in wanted else None,
            config=config,
            entity_name=entity_name,
            entity_type=entity_type,
            tier=tier,
            used_queries=used + [q["query"] for q in out],
        )
        if STAGE_C in wanted:
            for category in plan.get("upstream_categories") or []:
                for template in COMPONENT_TEMPLATES:
                    _add(template.format(entity=entity_name, category=category), STAGE_C)
        if STAGE_D in wanted:
            for query in plan.get("queries") or []:
                _add(query, STAGE_D)

    if STAGE_E in wanted:
        for template in OFFICIAL_TEMPLATES:
            _add(template.format(entity=entity_name), STAGE_E, providers=["general_web", "google_news"])

    logger.info(
        "escalation queries entity=%s stages=%s count=%d",
        entity_name, sorted(wanted), len(out),
    )
    return out
