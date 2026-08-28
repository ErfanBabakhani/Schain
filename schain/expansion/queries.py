"""Tier-aware query generation.

Expansion is node-centric: each frontier entity is searched with the small set of query
templates configured for its tier. No query is ever built from a pair of graph nodes, so
the number of searches grows as

    expanded entities x templates for that tier

rather than as the square of the graph size.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import resolve_tier_queries

logger = logging.getLogger(__name__)

ENTITY_PLACEHOLDER = "{entity}"


def generate_queries(config: dict[str, Any], entity_name: str, tier: int) -> list[dict[str, Any]]:
    """Build the concrete queries for one entity at one tier.

    Each returned entry carries the substituted query string plus the limits resolved by
    the config hierarchy (global defaults < tier defaults < per-query override).
    ``max_results`` applies to that single query.
    """
    queries = []
    for spec in resolve_tier_queries(config, tier):
        template = spec["query"]
        if ENTITY_PLACEHOLDER not in template:
            logger.warning(
                "tier %d template %r has no %s placeholder; used as a literal query",
                tier, template, ENTITY_PLACEHOLDER,
            )
        queries.append({**spec, "template": template,
                        "query": template.replace(ENTITY_PLACEHOLDER, entity_name)})
    return queries
