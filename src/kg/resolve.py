"""Prompt 2 - entity disambiguation into canonical identifiers.

Two stages, cheapest first (CLAUDE.md: "Deterministic normalization should happen
before LLM disambiguation for trivial cases"):

1. Deterministic grouping on a normalization key - casing, punctuation, whitespace and
   company legal suffixes. "BMW AG" and "bmw ag." collapse here, with no LLM call.
2. One LLM call per entity type over the remaining distinct groups, which is what
   catches "CATL" vs "Contemporary Amperex Technology Co. Limited".

The canonical name of a group is the longest mention in it, not whatever the model
chooses to call it. That keeps ``canonical_id`` a function of group membership alone, so
repeated runs produce the same id and Neo4j MERGE stays idempotent.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import defaultdict
from typing import Any, Iterable, Optional

from json_repair import repair_json
from neo4j_graphrag.components.types import Neo4jGraph
from neo4j_graphrag.llm import LLMInterface

logger = logging.getLogger(__name__)

# Company legal forms, stripped before comparing names.
LEGAL_SUFFIXES = (
    "incorporated", "corporation", "company", "holdings", "holding", "group",
    "limited", "inc", "ltd", "llc", "plc", "corp", "co", "ag", "gmbh", "sa",
    "nv", "bv", "ab", "as", "oyj", "kk", "pte", "spa", "srl", "sas", "kgaa",
)

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


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalization_key(name: str) -> str:
    """Comparison key: accent-free, lower-case, punctuation-free, no legal suffix."""
    text = strip_accents(name).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    while words and words[-1] in LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words) or " ".join(text.split())


def canonical_id(label: str, canonical_name: str) -> str:
    """Stable node identifier, e.g. ``company:catl``. Used as the Neo4j MERGE key."""
    slug = re.sub(r"\s+", "-", normalization_key(canonical_name)).strip("-")
    return f"{label.lower()}:{slug}"


def pick_canonical_name(mentions: Iterable[str]) -> str:
    """Longest mention wins; ties break alphabetically so the result is deterministic."""
    return sorted(set(mentions), key=lambda name: (-len(name), name))[0]


def _mentions_by_label(graph: Neo4jGraph) -> dict[str, dict[str, list[str]]]:
    """Collect ``label -> node_id -> name`` from the extracted graph."""
    grouped: dict[str, dict[str, list[str]]] = defaultdict(dict)
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
        logger.warning("Prompt 2 output unusable for %s (%s); keeping groups apart", label, error)
        return [[name] for name in names]

    # Any name the model dropped stays in a group of its own, so nothing is lost.
    covered = {name for group in groups for name in group}
    groups.extend([name] for name in names if name not in covered)
    return [group for group in groups if group]


def resolve_entities(
    graph: Neo4jGraph,
    llm: Optional[LLMInterface] = None,
) -> dict[str, dict[str, Any]]:
    """Resolve every extracted node to a canonical entity.

    Returns ``node_id -> {canonical_id, canonical_name, label, aliases, mention}``.
    Pass ``llm=None`` to run deterministic normalization only.
    """
    resolution: dict[str, dict[str, Any]] = {}

    for label, nodes in _mentions_by_label(graph).items():
        # Stage 1: deterministic grouping.
        by_key: dict[str, list[str]] = defaultdict(list)
        for name in nodes.values():
            by_key[normalization_key(name)].append(name)
        representatives = [pick_canonical_name(names) for names in by_key.values()]

        # Stage 2: LLM grouping over the deterministic representatives.
        if llm is not None and len(representatives) > 1:
            groups = _llm_groups(llm, label, representatives)
        else:
            groups = [[name] for name in representatives]

        # Map every mention of this label onto the group it landed in.
        group_of_representative = {
            name: index for index, group in enumerate(groups) for name in group
        }
        members: dict[int, list[str]] = defaultdict(list)
        for key, names in by_key.items():
            representative = pick_canonical_name(names)
            index = group_of_representative.get(representative, -1)
            members[index].extend(names)

        for index, names in members.items():
            canonical_name = pick_canonical_name(names)
            entry = {
                "label": label,
                "canonical_name": canonical_name,
                "canonical_id": canonical_id(label, canonical_name),
                "aliases": sorted(set(names)),
            }
            for node_id, name in nodes.items():
                if name in names:
                    resolution[node_id] = {**entry, "mention": name}

        logger.info(
            "resolved %d %s mention(s) into %d canonical entit(ies)",
            len(nodes),
            label,
            len(members),
        )

    return resolution
