"""Deterministic, origin-relative discovery ranking; never a fact acceptance gate.

An explicit supplier OF the frontier entity takes priority over a bilateral agreement
or a downstream customer. Reputation remains informational metadata. No LLM calls.
"""

from __future__ import annotations

import re
from typing import Any, Optional
from urllib.parse import urlsplit

from ..kg.relation_evidence import (
    DIRECTION_AMBIGUOUS,
    DOWNSTREAM_EXPLICIT,
    UPSTREAM_EXPLICIT,
    classify_supply_direction,
    has_relation_cues,
)
from ..names import normalization_key
from .base import FULL_TEXT, SearchResult

# Broad discovery cues preserve recall. These are NOT sufficient relation evidence;
# the executable YAML policy and downstream semantic validation decide that separately.
RELATION_KEYWORDS = (
    "supplier", "suppliers", "supplies", "supply", "sourcing", "sources",
    "procurement", "purchases", "contract", "agreement", "offtake",
)

# Broad supplier-list pages are high-value discovery even without an explicit
# "X supplies Y" title. Validation still decides acceptance later.
SUPPLIER_LIST_CUES = (
    "main suppliers", "key suppliers", "major suppliers", "top suppliers",
    "who supplies", "supplier list", "list of suppliers", "automotive suppliers",
    "component suppliers", "parts suppliers",
)

# Prefer direct/full-text over Google News wrappers when scores are close.
DIRECT_SOURCE_MARGIN = 15.0

# Penalize navigation / registration results, not legitimate annual reports or an
# incidental privacy/cookie footer in otherwise useful publisher full text.
NEGATIVE_PHRASES = (
    "login", "log in", "sign in", "careers", "job opening", "vacancies",
    "supplier portal", "supplier login", "e-sourcing", "esourcing",
    "procurement portal", "sourcing portal", "register as a supplier",
    "become a supplier", "vendor registration", "code of conduct",
    "privacy policy", "cookie policy", "terms of use",
)

# Direction dominates ALL secondary scores; the remaining score is in [-55, 57].
DIRECTION_PRIORITY = {
    UPSTREAM_EXPLICIT: 300.0,
    DIRECTION_AMBIGUOUS: 150.0,
    DOWNSTREAM_EXPLICIT: 0.0,
}


def mentions_entity_name(
    text: str,
    entity_name: str,
    aliases: Optional[list[str]] = None,
) -> bool:
    """Boundary-aware mention of a name or explicitly configured alias.

    Legal-suffix normalization is shared with canonical identity; partial tokens such
    as BYD in BYDauto are not accepted, and no overlap-derived aliases are invented.
    """
    normalized = normalization_key(text or "")
    for name in [entity_name, *(aliases or [])]:
        key = normalization_key(name or "")
        if key and re.search(rf"(?<!\w){re.escape(key)}(?!\w)", normalized):
            return True
    return False


def has_explicit_relation_language(
    text: str,
    *,
    policy: Optional[dict[str, Any]] = None,
) -> bool:
    """Compatibility name for topical discovery eligibility, NOT explicit direction.

    Configured relation cues and generic supply/supplier wording can identify a
    useful short search result. Only ``classify_supply_direction`` gives it a
    directional ranking, and only grounded semantic validation accepts an edge.
    """
    if not text:
        return False
    return has_relation_cues(text, policy=policy) or bool(
        re.search(r"\bsuppl(?:y|ies|ier|iers)\b", text.casefold())
    )


def _has_negative_portal_language(text: str) -> bool:
    lowered = (text or "").casefold()
    return any(phrase in lowered for phrase in NEGATIVE_PHRASES)


def _query_is_relationship_specific(query: str) -> bool:
    lowered = (query or "").casefold()
    return any(re.search(rf"\b{re.escape(marker)}\b", lowered) for marker in RELATION_KEYWORDS)


def _has_direct_article_url(result: SearchResult) -> bool:
    """Article-shaped URLs cost less to fetch than unresolved Google wrappers."""
    try:
        parts = urlsplit(result.canonical_source_url or result.url or "")
    except ValueError:
        return False
    host = (parts.hostname or "").casefold()
    return bool(
        parts.scheme in ("http", "https") and host
        and host != "google.com" and not host.endswith(".google.com")
        and (parts.path.strip("/") or parts.query)
    )


def discovery_candidate_is_meaningful(
    result: SearchResult,
    entity_name: str,
    *,
    aliases: Optional[list[str]] = None,
    policy: Optional[dict[str, Any]] = None,
) -> bool:
    """Does substantive supply language survive removing navigation-only clauses?"""
    text = result.document_text()
    if not text:
        return False
    direction = classify_supply_direction(entity_name, text, aliases=aliases, policy=policy)
    if direction != UPSTREAM_EXPLICIT and _has_negative_portal_language(result.title or ""):
        return False
    # Broad EV vocabulary is useful for discovery, but a company update does not
    # become supply evidence merely because its footer links to a supplier portal.
    # Only navigation clauses are ignored; ordinary bilateral supply agreements
    # remain eligible for extraction even when deterministic direction is unknown.
    substantive_cues = any(
        has_explicit_relation_language(clause, policy=policy)
        for clause in re.split(r"[.!?;\n]+", text)
        if clause.strip() and not _has_negative_portal_language(clause)
    )
    if direction != UPSTREAM_EXPLICIT and not substantive_cues:
        return False
    return True


def is_usable_discovery_result(
    result: SearchResult,
    entity_name: str,
    *,
    min_chars_full: int = 200,
    min_chars_snippet: int = 40,
    aliases: Optional[list[str]] = None,
    policy: Optional[dict[str, Any]] = None,
) -> bool:
    """Keep useful evidence, never fill a document target with navigation pages.

    Soft ranking signals (ambiguous direction, generic supplier-list titles) do
    NOT hard-reject here. Only hard junk is excluded before the LLM.
    """
    text = result.document_text()
    if not text:
        return False
    mentioned = mentions_entity_name(text, entity_name, aliases)
    # GDELT/title-only: entity may appear only after full-text fetch. When the
    # query already names the entity and the title has supply language, keep the
    # candidate for selection so fetch/extraction can inspect it.
    query_mentions = mentions_entity_name(result.query or "", entity_name, aliases)
    if not mentioned:
        if not (
            result.provider == "gdelt"
            and query_mentions
            and has_explicit_relation_language(text, policy=policy)
        ):
            return False
    if not discovery_candidate_is_meaningful(result, entity_name, aliases=aliases, policy=policy):
        # Soft exception: query-aligned GDELT titles with supply cues.
        if not (
            result.provider == "gdelt"
            and query_mentions
            and has_explicit_relation_language(text, policy=policy)
        ):
            return False
    if result.content_source == FULL_TEXT:
        return len(text) >= min(min_chars_full, 80) or has_explicit_relation_language(text, policy=policy)
    if len(text) < min_chars_snippet:
        # Short titles that name the entity and a supplier cue remain usable.
        if mentioned and has_explicit_relation_language(text, policy=policy):
            return True
        return False
    return has_explicit_relation_language(text, policy=policy) or (
        mentioned and any(
            token in text.casefold()
            for token in ("supplier", "supply", "sourcing", "procurement", "agreement")
        )
    )


def rank_discovery_candidate(
    result: SearchResult,
    entity_name: str,
    *,
    source_effective_score: Optional[float] = None,
    aliases: Optional[list[str]] = None,
    policy: Optional[dict[str, Any]] = None,
) -> float:
    """Higher is better; upstream > ambiguous > downstream regardless of publisher.

    ``source_effective_score`` remains accepted for caller compatibility, but never
    affects ranking or acceptance. Explicit downstream documents remain candidates.
    """
    title = result.title or ""
    snippet = result.snippet or ""
    text = result.document_text()
    # Preserve sentence boundaries between headline/snippet/body: a repeated headline
    # must not connect an actor in one field to an unrelated actor in another.
    blob = "\n".join(part for part in (title, snippet, text) if part)
    direction = classify_supply_direction(entity_name, blob, aliases=aliases, policy=policy)
    score = DIRECTION_PRIORITY.get(direction, DIRECTION_PRIORITY[DIRECTION_AMBIGUOUS])

    if mentions_entity_name(title, entity_name, aliases):
        score += 25.0
    elif mentions_entity_name(blob, entity_name, aliases):
        score += 8.0

    if has_explicit_relation_language(title, policy=policy):
        score += 12.0
    elif has_explicit_relation_language(snippet, policy=policy):
        score += 8.0
    elif has_explicit_relation_language(text, policy=policy):
        score += 4.0

    blob_cf = blob.casefold()
    if any(cue in blob_cf for cue in SUPPLIER_LIST_CUES):
        # Keep below UPSTREAM_EXPLICIT priority; still above bare ambiguous noise.
        score += 18.0 if direction != DOWNSTREAM_EXPLICIT else 0.0

    if _query_is_relationship_specific(result.query or ""):
        score += 5.0
    if result.content_source == FULL_TEXT:
        score += 10.0
    # Before full-text fetching, equivalent direct article candidates should not
    # lose every shortlist slot to unresolved wrappers solely by provider order.
    # This is a URL/fetch-cost signal, not a publisher reputation or provider quota.
    if _has_direct_article_url(result):
        score += 3.0

    if _has_negative_portal_language(f"{title}\n{snippet}"):
        score -= 40.0
    if _has_negative_portal_language(title):
        score -= 15.0

    result._discovery_direction = direction  # type: ignore[attr-defined]
    result._discovery_rank = score  # type: ignore[attr-defined]
    # New dataclass fields are additive; older SearchResult shapes still work.
    if "discovery_direction" in result.__dataclass_fields__:
        result.discovery_direction = direction
    if "discovery_rank" in result.__dataclass_fields__:
        result.discovery_rank = score
    return score


def sort_discovery_results(
    results: list[SearchResult],
    entity_name: str,
    *,
    aliases: Optional[list[str]] = None,
    policy: Optional[dict[str, Any]] = None,
) -> list[SearchResult]:
    """Descending directional priority; stable ties, soft direct/full-text preference."""
    scored = [
        (rank_discovery_candidate(result, entity_name, aliases=aliases, policy=policy), index, result)
        for index, result in enumerate(results)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    ordered = [result for _, _, result in scored]
    return prefer_direct_full_text(ordered, margin=DIRECT_SOURCE_MARGIN)


def prefer_direct_full_text(
    results: list[SearchResult],
    *,
    margin: float = DIRECT_SOURCE_MARGIN,
    max_promotions: int = 2,
) -> list[SearchResult]:
    """Within a near-tie margin, promote direct/full-text over wrappers.

    Does not invent poor candidates — only reorders competitive ones that already
    cleared ranking. At most ``max_promotions`` non-wrapper slots are protected
    near the top when such candidates exist.
    """
    if len(results) < 2:
        return results
    out = list(results)
    promoted = 0
    i = 0
    while i < len(out) and promoted < max_promotions:
        candidate = out[i]
        is_direct = _has_direct_article_url(candidate) or candidate.content_source == FULL_TEXT
        if not is_direct:
            i += 1
            continue
        displaced = False
        for j in range(i):
            ahead = out[j]
            ahead_host = (urlsplit(ahead.url or "").hostname or "").casefold()
            ahead_is_wrapper = "google.com" in ahead_host and ahead.content_source != FULL_TEXT
            if not ahead_is_wrapper:
                continue
            score_c = float(getattr(candidate, "_discovery_rank", 0.0) or 0.0)
            score_a = float(getattr(ahead, "_discovery_rank", 0.0) or 0.0)
            if score_a - score_c <= margin:
                out.pop(i)
                out.insert(j, candidate)
                promoted += 1
                displaced = True
                break
        if not displaced:
            i += 1
    return out
