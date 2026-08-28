"""Prompt 2 - entity disambiguation into canonical identifiers.

Four stages, cheapest first:

1. Cache. Any mention resolved in an earlier run is reused from the state database, so
   the same disambiguation is never paid for twice.
2. Deterministic grouping on a normalization key - casing, punctuation, whitespace and
   company legal suffixes. "BMW AG" and "bmw ag." collapse here with no LLM call.
3. Known aliases from config/ontology.yaml, also deterministic: a configured alias is
   grouped with its canonical name, so "CATL" needs no LLM call once it is listed.
4. One LLM call per entity type over whatever is left, which is what catches alias pairs
   nobody configured. Entity types are never mixed.

Two stability rules keep repeated runs from producing duplicate graph nodes:

* the canonical name of a group is the longest mention in it, not whatever the model
  chooses to call it, so ``canonical_id`` depends on group membership alone;
* if any member of a group is already a known canonical entity, that entity's existing
  id and name win, so an entity already written to Neo4j is never renamed.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any, Optional

from json_repair import repair_json
from neo4j_graphrag.components.types import Neo4jGraph
from neo4j_graphrag.llm import LLMInterface

from ..names import canonical_id, mention_key, normalization_key, pick_canonical_name

logger = logging.getLogger(__name__)

# Above this many known canonical names for one entity type, the known-candidate list is
# left out of the prompt rather than sending an unreadably long prompt.
MAX_KNOWN_CANDIDATES = 100

DISAMBIGUATION_PROMPT = """\
You are resolving entity names extracted from supply-chain text.

Below is a list of {entity_type} names. Group the names that refer to the same
real-world {entity_type}, for example an abbreviation and its full legal name.

Rules:
- Group only names you are confident denote the same real-world entity.
- Keep genuinely different entities in separate groups, including parent companies and
  their subsidiaries, and different plants or divisions.
- Every input name must appear in exactly one group.
- Do not invent names that are not in the list.
- Return JSON only, in this form, with no commentary:
{{"groups": [["name A", "name A full legal name"], ["name B"]]}}

Names:
{names}
"""


def _mentions_by_label(graph: Neo4jGraph) -> dict[str, dict[str, str]]:
    """Collect ``label -> node_id -> name`` from the extracted graph."""
    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    for node in graph.nodes:
        name = (node.properties or {}).get("name")
        if not name:
            logger.warning("node %s has no name property, skipping", node.id)
            continue
        grouped[node.label][node.id] = str(name).strip()
    return grouped


def _llm_groups(llm: LLMInterface, label: str, names: list[str]) -> list[list[str]]:
    """Ask Prompt 2 to group aliases for one entity type."""
    prompt = DISAMBIGUATION_PROMPT.format(
        entity_type=label, names="\n".join(f"- {name}" for name in names)
    )
    raw = llm.invoke(prompt).content
    try:
        payload = json.loads(repair_json(raw))
        groups = [
            [str(name) for name in group if str(name) in names]
            for group in payload.get("groups", [])
        ]
    except (ValueError, TypeError, AttributeError) as error:
        logger.warning(
            "Prompt 2 output unusable for %s (%s); keeping groups apart", label, error
        )
        return [[name] for name in names]

    # Any name the model dropped stays in a group of its own, so nothing is lost.
    covered = {name for group in groups for name in group}
    groups.extend([name] for name in names if name not in covered)
    return [group for group in groups if group]


def alias_lookup(aliases: Optional[dict[str, Any]], label: str) -> dict[str, str]:
    """Configured aliases for one entity type, keyed by the alias' normalization key."""
    for_label = ((aliases or {}).get(label) or {})
    return {normalization_key(alias): canonical for alias, canonical in for_label.items()}


def _known_canonical(cache: dict[str, dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    """Known canonical entities of one label, keyed by canonical name."""
    known: dict[str, dict[str, Any]] = {}
    for entry in cache.values():
        if entry["label"] == label:
            known[entry["canonical_name"]] = entry
    return known


def resolve_entities(
    graph: Neo4jGraph,
    llm: Optional[LLMInterface] = None,
    cache: Optional[dict[str, dict[str, Any]]] = None,
    aliases: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Resolve every extracted node to a canonical entity.

    Returns ``(resolution, new_entries)`` where ``resolution`` maps
    ``node_id -> {canonical_id, canonical_name, label, aliases, mention, method}`` and
    ``new_entries`` are the mentions that were not in the cache and should be persisted.

    Pass ``llm=None`` to run the cache, normalization and alias stages only.
    """
    cache = cache or {}
    resolution: dict[str, dict[str, Any]] = {}
    new_entries: list[dict[str, Any]] = []

    for label, nodes in _mentions_by_label(graph).items():
        cached_nodes: dict[str, str] = {}
        fresh_nodes: dict[str, str] = {}
        for node_id, name in nodes.items():
            if mention_key(label, name) in cache:
                cached_nodes[node_id] = name
            else:
                fresh_nodes[node_id] = name

        # Stage 1: mentions already resolved in an earlier run.
        for node_id, name in cached_nodes.items():
            entry = cache[mention_key(label, name)]
            resolution[node_id] = {
                "label": label,
                "canonical_id": entry["canonical_id"],
                "canonical_name": entry["canonical_name"],
                "aliases": [name],
                "mention": name,
                "method": "cache",
            }

        if not fresh_nodes:
            logger.info("%s: all %d mention(s) served from cache", label, len(nodes))
            continue

        # Stages 2 and 3: deterministic grouping, with configured aliases folded in by
        # grouping an alias under its canonical name's key.
        known_aliases = alias_lookup(aliases, label)
        by_key: dict[str, list[str]] = defaultdict(list)
        configured_name: dict[str, str] = {}
        for name in fresh_nodes.values():
            alias_target = known_aliases.get(normalization_key(name))
            key = normalization_key(alias_target or name)
            by_key[key].append(name)
            if alias_target:
                configured_name[key] = alias_target

        def representative_of(key: str) -> str:
            """A configured canonical name wins over the longest mention."""
            return configured_name.get(key) or pick_canonical_name(by_key[key])

        representatives = [representative_of(key) for key in by_key]

        # Stage 3: LLM grouping. Known canonical names are offered as candidates so a new
        # alias attaches to an existing entity instead of creating a second one.
        known = _known_canonical(cache, label)
        candidates = list(representatives)
        if known and len(known) <= MAX_KNOWN_CANDIDATES:
            candidates += [name for name in known if name not in candidates]

        if llm is not None and len(candidates) > 1:
            groups = _llm_groups(llm, label, candidates)
            method = "llm"
        else:
            groups = [[name] for name in candidates]
            method = "deterministic"

        group_of = {name: index for index, group in enumerate(groups) for name in group}
        members: dict[int, list[str]] = defaultdict(list)
        group_configured: dict[int, str] = {}
        for key, names in by_key.items():
            index = group_of.get(representative_of(key), -1)
            members[index].extend(names)
            if key in configured_name:
                group_configured[index] = configured_name[key]

        for index, names in members.items():
            # Reuse an existing canonical entity if this group reached one.
            group_names = groups[index] if 0 <= index < len(groups) else names
            existing = next((known[n] for n in group_names if n in known), None)
            if existing is not None:
                canonical_name = existing["canonical_name"]
                entity_id = existing["canonical_id"]
            else:
                canonical_name = group_configured.get(index) or pick_canonical_name(names)
                entity_id = canonical_id(label, canonical_name)

            entry = {
                "label": label,
                "canonical_name": canonical_name,
                "canonical_id": entity_id,
                "aliases": sorted(set(names)),
                "method": method,
            }
            for node_id, name in fresh_nodes.items():
                if name in names:
                    resolution[node_id] = {**entry, "mention": name}
            for name in set(names):
                new_entries.append({**entry, "mention": name})

        logger.info(
            "%s: %d cached, %d new mention(s) resolved into %d canonical entit(ies) via %s",
            label, len(cached_nodes), len(fresh_nodes), len(members), method,
        )

    return resolution, new_entries
