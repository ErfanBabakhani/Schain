"""Persist only witnessed upstream candidates, preserving canonical IDs and types."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .. import state
from ..config import seed_entities
from ..kg.support import aggregate_edge_support, contextualize_facts, frontier_supplier_records, load_edge_support
from ..names import canonical_id, mention_key

logger = logging.getLogger(__name__)


def seed_frontier(connection, config: dict[str, Any], ontology=None) -> int:
    """Configured seeds always enter at Tier 0, including on a shorter-path revisit."""
    aliases = ((ontology or {}).get("aliases") or {}).get("Company") or {}
    normalized_aliases = {mention_key("Company", key): value for key, value in aliases.items()}
    max_tier = int(config.get("expansion", {}).get("max_tier", 3))
    added = 0
    for name in seed_entities(config):
        canonical_name = normalized_aliases.get(mention_key("Company", name), name)
        added += state.enqueue_entity(
            connection, name, tier=0, entity_id=canonical_id("Company", canonical_name),
            expandable=max_tier > 0, stop_reason="terminal_tier" if max_tier == 0 else None,
            priority=1000, priority_reason={"configured_seed": True},
        )
    logger.info("seeded %d new Tier-0 entit(ies)", added)
    return added


def entity_identifiers(connection, entity_name: str, entity_type="Company", entity_id=None) -> set[str]:
    identifiers = {entity_id or canonical_id(entity_type, entity_name)}
    row = connection.execute(
        "SELECT canonical_id FROM entity_resolution WHERE mention_key=?",
        (mention_key(entity_type, entity_name),),
    ).fetchone()
    if row:
        identifiers.add(row["canonical_id"])
    return identifiers


def discover_upstream(connection, config: dict[str, Any], ontology: dict[str, Any],
                      entity_name: str, entity_key: str, tier: int,
                      facts: pd.DataFrame) -> list[str]:
    """Store eligible next-tier candidates; only enabled types below max_tier expand.

    Support eligibility already requires one complete, grounded, directional witness.
    A local fallback is aggregated through the same gate; it never trusts raw triples.
    """
    max_tier = int(config.get("expansion", {}).get("max_tier", 3))
    if tier >= max_tier:
        return []
    parent = connection.execute("SELECT * FROM frontier WHERE entity_key=?", (entity_key,)).fetchone()
    if parent is None or int(parent["tier"]) != int(tier):
        logger.warning("frontier rejected parent=%s reason=invalid_parent_tier", entity_key)
        return []
    identifiers = entity_identifiers(connection, entity_name, parent["entity_type"], entity_key)
    support = load_edge_support(config)
    if support.empty and facts is not None and not facts.empty:
        local = contextualize_facts(facts, state.document_origins(connection))
        support = aggregate_edge_support(local, config=config)
    candidates = frontier_supplier_records(support, identifiers, ["suppliesTo"])
    enabled_types = set(ontology.get("expandable_types") or [])
    next_tier = tier + 1
    added = []
    for candidate in candidates:
        # Self-supply is never a new supply-chain tier; cycles cannot lengthen a known tier.
        if candidate["entity_id"] in identifiers:
            continue
        entity_type = candidate["entity_type"]
        expandable = entity_type in enabled_types and next_tier < max_tier
        reason = dict(candidate.get("frontier_reason") or {})
        reason.update(parent_tier_valid=True, parent_id=entity_key, parent_tier=tier,
                      edge_key=candidate.get("edge_key"), entity_type=entity_type)
        terminal = "terminal_tier" if next_tier >= max_tier else "non_expandable_type"
        if state.enqueue_entity(
            connection, candidate["entity_name"], next_tier, discovered_from=entity_key,
            entity_id=candidate["entity_id"], entity_type=entity_type, expandable=expandable,
            priority=float(candidate.get("priority") or 0), priority_reason=reason,
            stop_reason=None if expandable else terminal,
        ):
            added.append(candidate["entity_name"])
            logger.info("frontier added id=%s type=%s tier=%d via=%s expandable=%s",
                        candidate["entity_id"], entity_type, next_tier, entity_key, expandable)
    return added
