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
import re
from collections import defaultdict
from typing import Any, Optional

from json_repair import repair_json
from neo4j_graphrag.components.types import Neo4jGraph
from neo4j_graphrag.llm import LLMInterface

from ..audit import llm_call_context, prompt_hash
from ..names import canonical_id, mention_key, normalization_key, pick_canonical_name
from .entity_types import company_type_hint

logger = logging.getLogger(__name__)

# Above this many known canonical names for one entity type, the known-candidate list is
# left out of the prompt rather than sending an unreadably long prompt.
MAX_KNOWN_CANDIDATES = 100

DISAMBIGUATION_PROMPT = """\
You are resolving entity names extracted from supply-chain text.

Below is a list of {entity_type} names. Group the names that refer to the same
real-world {entity_type}, for example an abbreviation and its full legal name.

Rules:
- WHEN UNCERTAIN, DO NOT MERGE. False split is better than false merge.
- Group only names you are HIGHLY CONFIDENT denote the identical real-world entity.
- Valid merges: legal-suffix variation, capitalization, clear acronym/abbreviation,
  configured aliases, formally equivalent trade/legal names.
- Do NOT merge merely because entities appear in the same article, operate in the
  same industry, supply similar products, supply one another, are competitors,
  share a parent/customer, or have vaguely similar names.
- Do NOT merge different products or materials that are merely in the same category
  (example: Blade Batteries is not Ultium battery pack; ventilated rotors are not
  carbon-sleeved rotors).
- Do NOT merge a plant/factory/facility with its owning company
  (example: BMW Group Plant Landshut is not BMW).
- Do NOT merge distinct companies such as BYD and CATL.
- Example of incorrect merge: Volkswagen is not Stellantis.
- Keep genuinely different entities in separate groups, including parent companies and
  their subsidiaries, and different plants or divisions.
- Do NOT merge a parent company with a named subsidiary, division, brand, business
  unit, or affiliate unless the evidence explicitly says they are the same legal
  entity (example: Thyssenkrupp is not Thyssenkrupp Steel / nucera / Aerospace).
- Every input name must appear in exactly one group.
- Do not invent names that are not in the list.
- Return JSON only, in this form, with no commentary:
{{"groups": [["name A", "name A full legal name"], ["name B"]]}}

Names:
{names}
"""

_FACILITY_TOKENS = frozenset({
    "plant", "factory", "facility", "works", "campus", "mill", "smelter",
    "refinery", "mine site", "warehouse",
})


def looks_like_facility(name: str) -> bool:
    """True for plant/factory/facility mentions that must not alias to the company."""
    tokens = set(normalization_key(name).split())
    return bool(tokens & _FACILITY_TOKENS) or "plant" in normalization_key(name)


def _is_acronym_of(short: str, long_key: str, long_name: str = "") -> bool:
    """Require whole-name initials; legal Limited may supply the final L in CATL."""
    if not short or " " in short or " " not in long_key or len(short) < 2:
        return False
    initials = "".join(word[0] for word in long_key.split())
    if short == initials or short == long_key.replace(" ", ""):
        return True
    legal_limited = long_name.casefold().rstrip(". ").endswith((" limited", " ltd"))
    return bool(legal_limited and short == initials + "l")


def _document_supports_shorthand(
    left: str,
    right: str,
    document_text: Optional[str],
    candidate_names: Optional[list[str]],
) -> bool:
    """Return true only for explicit or collision-free in-document shorthand."""
    if not document_text:
        return False
    ka, kb = normalization_key(left), normalization_key(right)
    short, long_name = (left, right) if len(ka.split()) < len(kb.split()) else (right, left)
    short_key, long_key = normalization_key(short), normalization_key(long_name)
    if not short_key or not long_key.startswith(short_key + " "):
        return False

    text = document_text.casefold()
    # A shared leading token is not an alias when the document describes a
    # corporate relationship.  This check deliberately precedes the generic
    # shorthand rules: subsidiaries and divisions remain distinct entities.
    corporate_separation_patterns = (
        rf"{re.escape(long_name.casefold())}\s+is\s+(?:a|an|the)\s+"
        rf"(?:subsidiary|division|affiliate|business unit|brand)\s+of\s+"
        rf"{re.escape(short.casefold())}",
        rf"{re.escape(short.casefold())}(?:'s|\s+owned)\s+"
        rf"(?:subsidiary|division|affiliate|business unit|brand)\s+"
        rf"{re.escape(long_name.casefold())}",
        rf"{re.escape(short.casefold())}\s+and\s+{re.escape(long_name.casefold())}"
        rf"\s+are\s+(?:different|distinct|separate)",
    )
    if any(re.search(pattern, text) for pattern in corporate_separation_patterns):
        return False

    explicit_patterns = (
        rf"{re.escape(long_name.casefold())}\s+is\s+also\s+abbreviated\s+as\s+{re.escape(short.casefold())}",
        rf"{re.escape(short.casefold())}\s+and\s+{re.escape(long_name.casefold())}\s+refer\s+to\s+the\s+same",
        rf"{re.escape(long_name.casefold())}\s+and\s+{re.escape(short.casefold())}\s+refer\s+to\s+the\s+same",
    )
    if any(re.search(pattern, text) for pattern in explicit_patterns):
        return True

    collisions = {
        normalization_key(name)
        for name in (candidate_names or [])
        if normalization_key(name).startswith(short_key + " ")
    }
    if len(collisions) != 1:
        return False
    remainder = long_key[len(short_key):].strip()
    # The short form must occur independently, not only as the prefix of the full name.
    independent = re.search(
        rf"(?<!\w){re.escape(short.casefold())}(?!\s+{re.escape(remainder)})(?!\w)",
        text,
    )
    return bool(independent and long_name.casefold() in text)


def identity_plausible(
    left: str,
    right: str,
    *,
    document_text: Optional[str] = None,
    candidate_names: Optional[list[str]] = None,
) -> bool:
    """Conservative check: refuse merges that are almost certainly false.

    Used to sanitize LLM groups. WHEN UNCERTAIN for this heuristic, returns False
    only for clearly unsafe pairs; acronym/alias pairs remain allowed.
    """
    if left == right:
        return True
    ka, kb = normalization_key(left), normalization_key(right)
    if not ka or not kb:
        return False
    if ka == kb:
        return True

    fa, fb = looks_like_facility(left), looks_like_facility(right)
    if fa or fb:
        # Facilities only collapse under identical normalization keys.
        return False

    wa, wb = set(ka.split()), set(kb.split())
    if _is_acronym_of(ka, kb, right) or _is_acronym_of(kb, ka, left):
        return True

    if wa.isdisjoint(wb):
        return False

    if len(wa) == 1 and len(wb) == 1 and ka != kb:
        return False

    # Parent / subsidiary / business-unit style nesting: shared stem but extra
    # distinctive tokens (Steel, Aerospace, Nucera, Rothe Erde, …) → do not merge.
    if wa != wb and (wa.issubset(wb) or wb.issubset(wa)):
        smaller, larger = (wa, wb) if len(wa) < len(wb) else (wb, wa)
        extras = larger - smaller
        # Shared corporate stem plus unit/product qualifier is not identity.
        if extras and len(smaller) >= 1:
            return _document_supports_shorthand(
                left, right, document_text, candidate_names,
            )

    if len(wa) >= 2 and len(wb) >= 2:
        jaccard = len(wa & wb) / len(wa | wb)
        if jaccard < 0.5:
            return False
        # High overlap but different longer forms with distinct trailing tokens.
        if wa != wb and jaccard < 0.85:
            return False
    return True


def sanitize_llm_groups(
    groups: list[list[str]],
    *,
    document_text: Optional[str] = None,
    candidate_names: Optional[list[str]] = None,
    audit=None,
    label: Optional[str] = None,
) -> list[list[str]]:
    """Complete-link groups: every pair must be safe, with no acronym bridge."""
    sanitized: list[list[str]] = []
    seen: set[str] = set()
    for group in groups:
        buckets: list[list[str]] = []
        for name in sorted({str(n) for n in group if n} - seen):
            seen.add(name)
            for bucket in buckets:
                if all(identity_plausible(
                    name,
                    other,
                    document_text=document_text,
                    candidate_names=candidate_names,
                ) for other in bucket):
                    bucket.append(name)
                    break
            else:
                buckets.append([name])
        sanitized.extend(buckets)
        if audit is not None and len(group) > 1:
            accepted = len(buckets) == 1
            audit.emit(
                "alias_merge_evaluated",
                stage="resolution",
                entity_type=label,
                proposed_names=sorted({str(name) for name in group if name}),
                accepted_groups=buckets,
                accepted=accepted,
                reason=(
                    "complete_link_identity_supported"
                    if accepted else "unsafe_or_ambiguous_identity_pair"
                ),
                related_llm_call_ids=audit.contextual_llm_call_ids(
                    "resolution", "entity_type", label or "",
                ),
            )
    return sanitized


def _mentions_by_label(graph: Neo4jGraph) -> dict[str, dict[str, str]]:
    """Collect ``label -> node_id -> name`` from the extracted graph."""
    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    for node in graph.nodes:
        name = (node.properties or {}).get("name")
        if not name:
            logger.warning("node %s has no name property, skipping", node.id)
            continue
        label = node.label
        if label == "Company":
            repaired_type, reason = company_type_hint(str(name))
            if repaired_type:
                label = repaired_type
                logger.info("entity type repair %s: Company -> %s (%s)", name, label, reason)
        grouped[label][node.id] = str(name).strip()
    return grouped


def _llm_groups(
    llm: LLMInterface,
    label: str,
    names: list[str],
    *,
    document_text: Optional[str] = None,
    audit=None,
) -> list[list[str]]:
    """Ask Prompt 2 to group aliases for one entity type."""
    prompt = DISAMBIGUATION_PROMPT.format(
        entity_type=label, names="\n".join(f"- {name}" for name in names)
    )
    with llm_call_context(
        prompt_template_id="DISAMBIGUATION_PROMPT",
        prompt_template_hash=prompt_hash(DISAMBIGUATION_PROMPT),
        entity_type=label,
    ):
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
    groups = [group for group in groups if group]
    # Conservative post-filter: refuse clearly false merges even if the model emitted them.
    return sanitize_llm_groups(
        groups,
        document_text=document_text,
        candidate_names=names,
        audit=audit,
        label=label,
    )


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
    document_text: Optional[str] = None,
    audit=None,
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
        known_aliases = alias_lookup(aliases, label)
        cached_nodes: dict[str, str] = {}
        fresh_nodes: dict[str, str] = {}
        for node_id, name in nodes.items():
            cached = cache.get(mention_key(label, name))
            configured = known_aliases.get(normalization_key(name))
            safe_alias = bool(configured and cached and
                normalization_key(configured) == normalization_key(cached["canonical_name"]) and
                not (looks_like_facility(name) and not looks_like_facility(configured)))
            if cached and (safe_alias or identity_plausible(name, cached["canonical_name"])):
                cached_nodes[node_id] = name
            else:
                if cached:
                    logger.warning("ignoring unsafe cached identity %s -> %s", name, cached["canonical_name"])
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
        # Never apply aliases that would collapse a facility mention into a company.
        known_aliases = alias_lookup(aliases, label)
        by_key: dict[str, list[str]] = defaultdict(list)
        configured_name: dict[str, str] = {}
        for name in fresh_nodes.values():
            alias_target = known_aliases.get(normalization_key(name))
            if alias_target and looks_like_facility(name) and not looks_like_facility(alias_target):
                alias_target = None
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
            groups = _llm_groups(
                llm,
                label,
                candidates,
                document_text=document_text,
                audit=audit,
            )
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

    original_labels = {node.id: node.label for node in graph.nodes}
    repaired_mentions = set()
    for node_id, entry in resolution.items():
        if original_labels.get(node_id) != entry["label"]:
            entry["method"] = "type_repair:branded_model_structure/" + entry["method"]
            repaired_mentions.add((entry["label"], entry["mention"]))
    for entry in new_entries:
        if (entry["label"], entry["mention"]) in repaired_mentions:
            entry["method"] = "type_repair:branded_model_structure/" + entry["method"]
    return resolution, new_entries
