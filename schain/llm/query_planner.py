"""Bounded LLM upstream query planner — retrieval only, never graph facts."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

from json_repair import repair_json

from ..config import storage_path

logger = logging.getLogger(__name__)

QUERY_PLANNER_PROMPT = """\
You plan SEARCH QUERIES to discover upstream suppliers of one company.

Entity: {entity_name}
Entity type: {entity_type}
Discovery tier: {tier}

Already used query families (do not repeat verbatim):
{used_queries}

Return JSON only:
{{
  "upstream_categories": ["category1", "..."],
  "queries": ["search query 1", "..."]
}}

Rules:
- Propose at most {max_categories} upstream material/component/service categories
  that are plausible for THIS entity (OEM vs battery vs semiconductor vs mining).
- Propose at most {max_queries} concrete web search queries targeting SUPPLIERS OF
  this entity (components, materials, parts, services, feedstock).
- Include at least one high-yield document query when possible, such as:
  supplier list, key suppliers, annual/sustainability/procurement report, PDF suppliers.
- Queries must help find evidence that some X supplies this entity.
- Do NOT name specific supplier companies as facts. Do NOT invent supply relationships.
- Do NOT claim anything about the graph. Output is for search only.
- Prefer queries with the entity name and category / report words.
"""

# Deterministic fallback categories when the LLM is unavailable or returns junk.
DEFAULT_CATEGORIES = {
    "battery": [
        "cathode active material", "precursor", "anode", "graphite", "electrolyte",
        "separator", "lithium", "nickel", "cobalt", "manganese", "copper foil",
        "aluminum foil",
    ],
    "semiconductor": [
        "wafer", "foundry", "packaging", "substrates", "memory",
        "fabrication materials", "specialty chemicals", "equipment",
    ],
    "automotive": [
        "batteries", "battery cells", "semiconductor", "displays",
        "thermal management", "tires", "braking systems", "seating",
        "wiring harnesses", "steel", "aluminum", "rare earth", "motors",
        "inverters", "e-axles",
    ],
    "default": ["component", "raw material", "parts", "materials", "modules"],
}

HIGH_YIELD_QUERY_TEMPLATES = (
    '"{entity}" supplier list',
    '"{entity}" key suppliers',
    '"{entity}" annual report suppliers',
    '"{entity}" sustainability report suppliers',
    '"{entity}" procurement report',
    '"{entity}" sourcing report',
    '"{entity}" supplier award',
    '"{entity}" filetype:pdf suppliers',
    '"{entity}" supply chain report',
)


def _cache_key(entity: str, model: str, prompt_version: str, config_hash: str) -> str:
    payload = "|".join([entity, model, prompt_version, config_hash])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _heuristic_categories(entity_name: str) -> list[str]:
    lowered = entity_name.casefold()
    if any(token in lowered for token in (
        "battery", "catl", "panasonic", "samsung sdi", "lg energy", "l&f", "lf ",
        "envision", "eve energy", "calb",
    )):
        return list(DEFAULT_CATEGORIES["battery"])
    if any(token in lowered for token in (
        "qualcomm", "tsmc", "semiconductor", "chip", "nxp", "infineon", "stmicro",
        "renesas", "broadcom", "nvidia",
    )):
        return list(DEFAULT_CATEGORIES["semiconductor"])
    if any(token in lowered for token in (
        "bmw", "toyota", "tesla", "byd", "stellantis", "volkswagen", "ford",
        "gm", "hyundai", "mercedes", "volvo", "rivian", "lucid",
    )):
        return list(DEFAULT_CATEGORIES["automotive"])
    return list(DEFAULT_CATEGORIES["default"])


def _high_yield_queries(entity_name: str, limit: int) -> list[str]:
    return [template.format(entity=entity_name) for template in HIGH_YIELD_QUERY_TEMPLATES][:limit]


def plan_upstream_queries(
    *,
    llm,
    config: dict[str, Any],
    entity_name: str,
    entity_type: str = "Company",
    tier: int = 0,
    used_queries: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Return ``{upstream_categories, queries, source}`` for retrieval only."""
    retrieval = (config.get("expansion") or {}).get("retrieval") or {}
    if not retrieval.get("query_planner_enabled", True):
        return {"upstream_categories": [], "queries": [], "source": "disabled"}

    max_categories = int(retrieval.get("query_planner_max_categories", 8))
    max_queries = int(retrieval.get("query_planner_max_queries", 12))
    used = used_queries or []
    model = str((config.get("llm") or {}).get("model") or "unknown")
    prompt_version = "query_planner_v2"
    config_hash = hashlib.sha256(
        json.dumps(
            {"max_categories": max_categories, "max_queries": max_queries},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    key = _cache_key(entity_name, model, prompt_version, config_hash)
    cache_dir = storage_path(config, "state_db").parent / "query_plans"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["source"] = "cache"
            return cached
        except (ValueError, OSError):
            pass

    categories = _heuristic_categories(entity_name)[:max_categories]
    category_queries = [
        f'"{entity_name}" {category} supplier'
        for category in categories
    ]
    # Reserve slots for high-yield list/report/PDF queries inside the cap.
    reserve = min(3, max(1, max_queries // 4))
    queries = list(dict.fromkeys(
        category_queries[: max(0, max_queries - reserve)]
        + _high_yield_queries(entity_name, reserve)
    ))[:max_queries]
    plan = {
        "upstream_categories": categories,
        "queries": queries,
        "source": "heuristic",
    }

    if llm is not None:
        prompt = QUERY_PLANNER_PROMPT.format(
            entity_name=entity_name,
            entity_type=entity_type,
            tier=tier,
            used_queries="\n".join(f"- {q}" for q in used[:40]) or "(none)",
            max_categories=max_categories,
            max_queries=max_queries,
        )
        previous = getattr(llm, "stage", None)
        try:
            if hasattr(llm, "stage"):
                llm.stage = "query_planner"
            raw = llm.invoke(prompt).content
            payload = json.loads(repair_json(raw))
            cats = [str(c).strip() for c in (payload.get("upstream_categories") or []) if str(c).strip()]
            qlist = [str(q).strip() for q in (payload.get("queries") or []) if str(q).strip()]
            # Never accept queries that look like asserted facts about named suppliers.
            cleaned = []
            for query in qlist:
                lowered = query.casefold()
                if " supplies " in lowered and entity_name.casefold() not in lowered:
                    continue
                cleaned.append(query)
            if cats or cleaned:
                merged = list(dict.fromkeys(
                    (cleaned or queries) + _high_yield_queries(entity_name, 2)
                ))[:max_queries]
                plan = {
                    "upstream_categories": (cats or categories)[:max_categories],
                    "queries": merged,
                    "source": "llm",
                }
        except Exception as error:  # noqa: BLE001
            logger.warning("query planner failed for %s: %s; using heuristic", entity_name, error)
        finally:
            if hasattr(llm, "stage") and previous is not None:
                llm.stage = previous

    used_norm = {" ".join(q.split()).casefold() for q in used}
    plan["queries"] = [
        q for q in plan["queries"]
        if " ".join(q.split()).casefold() not in used_norm
    ][:max_queries]
    try:
        cache_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    except OSError:
        pass
    return plan
