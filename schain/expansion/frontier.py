"""The persistent expansion frontier.

Tier semantics (PROJECT_DECISIONS.md section 3):

* Tier 0 is the configured seeds.
* A company discovered as an upstream supplier of a Tier-N company enters Tier N+1.
* Only tier-advancing relations (``suppliesTo`` by default) move a company upstream;
  ``locatedIn``, ``owns`` and ``produces`` describe an entity without adding depth.
* Only expandable entity types (``Company`` in V1) may enter the frontier.
* A company is enqueued only when its tier is within ``max_tier``, so Tier-3 companies
  are stored in the graph but never expanded.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .. import state
from ..config import seed_entities
from ..kg.facts import upstream_companies
from ..names import canonical_id, mention_key

logger = logging.getLogger(__name__)


def seed_frontier(connection, config: dict[str, Any]) -> int:
    """Put the configured seeds into the frontier at Tier 0. Existing rows are untouched."""
    added = 0
    for name in seed_entities(config):
        if state.enqueue_entity(connection, name, tier=0):
            added += 1
    logger.info("seeded %d new Tier-0 entit(ies)", added)
    return added


def entity_identifiers(connection, entity_name: str) -> set[str]:
    """The canonical ids this frontier entity may appear under in the facts table.

    Normally that is just ``canonical_id(name)``, but if Prompt 2 merged the entity into
    a differently named canonical entity in an earlier run, the resolution cache knows the
    id actually used, and facts must be matched against that too.
    """
    identifiers = {canonical_id("Company", entity_name)}
    row = connection.execute(
        "SELECT canonical_id FROM entity_resolution WHERE mention_key = ?",
        (mention_key("Company", entity_name),),
    ).fetchone()
    if row:
        identifiers.add(row["canonical_id"])
    return identifiers


def discover_upstream(
    connection,
    config: dict[str, Any],
    ontology: dict[str, Any],
    entity_name: str,
    entity_key: str,
    tier: int,
    facts: pd.DataFrame,
) -> list[str]:
    """Enqueue the companies that supply this entity, one tier further upstream.

    Returns the names actually added to the frontier.
    """
    max_tier = config.get("expansion", {}).get("max_tier", 3)
    next_tier = tier + 1
    if next_tier > max_tier:
        logger.info(
            "entity %s is at tier %d, the configured boundary; not expanding further",
            entity_name, tier,
        )
        return []
    if "Company" not in (ontology.get("expandable_types") or []):
        logger.info("Company is not an expandable type; nothing enqueued")
        return []

    tier_relations = ontology.get("tier_advancing_relations") or ["suppliesTo"]
    identifiers = entity_identifiers(connection, entity_name)
    candidates = upstream_companies(facts, identifiers, tier_relations)

    added = [
        name for name in candidates
        if state.enqueue_entity(connection, name, next_tier, discovered_from=entity_key)
    ]
    logger.info(
        "entity %s: %d upstream candidate(s), %d new at tier %d",
        entity_name, len(candidates), len(added), next_tier,
    )
    return added
