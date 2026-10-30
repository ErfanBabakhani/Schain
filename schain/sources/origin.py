"""Origin-aware relevance for EXPAND retrieval.

A document retrieved for frontier entity E must meaningfully discuss E.
Search-engine noise about other OEMs must not drive E's expansion.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ..names import normalization_key
from ..kg.relation_evidence import UPSTREAM_EXPLICIT, classify_supply_direction
from .base import FULL_TEXT, SearchResult
from .ranking import has_explicit_relation_language, mentions_entity_name

# Titles that are mostly about other automotive brands often indicate drift.
_OTHER_OEM_MARKERS = (
    "tesla", "byd", "lucid", "rivian", "norton", "stellantis", "volkswagen",
    "toyota", "ford", "gm ", "general motors", "hyundai", "kia ", "mercedes",
    "audi", "porsche", "volvo", "polestar", "nio", "xpeng", "li auto",
)


def origin_name_variants(entity_name: str, aliases: Optional[dict[str, str]] = None) -> list[str]:
    """Entity name plus configured aliases that resolve to it (or from it)."""
    names = [entity_name]
    key = normalization_key(entity_name)
    for alias, canonical in (aliases or {}).items():
        if normalization_key(canonical) == key or normalization_key(alias) == key:
            names.append(alias)
            names.append(canonical)
    # Unique, preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        nk = normalization_key(name)
        if nk and nk not in seen:
            seen.add(nk)
            out.append(name)
    return out


def mentions_any(text: str, names: list[str]) -> bool:
    return any(mentions_entity_name(text, name) for name in names if name)


def _mention_count(text: str, entity_name: str) -> int:
    key = normalization_key(entity_name)
    if not key:
        return 0
    norm = normalization_key(text or "")
    if not norm:
        return 0
    # Word-boundary-ish count on normalized whitespace text.
    return len(re.findall(rf"(?<!\w){re.escape(key)}(?!\w)", norm))


def _entity_near_supply_language(
    text: str, entity_name: str, window: int = 120,
    *, policy: Optional[dict[str, Any]] = None,
) -> bool:
    """True when supply-chain language appears near an origin-entity mention."""
    key = normalization_key(entity_name)
    norm = normalization_key(text or "")
    if not key or key not in norm:
        return False
    for match in re.finditer(rf"(?<!\w){re.escape(key)}(?!\w)", norm):
        start = max(0, match.start() - window)
        end = min(len(norm), match.end() + window)
        blob = norm[start:end]
        if has_explicit_relation_language(blob, policy=policy):
            return True
    return False


def _title_dominated_by_other_oems(title: str, entity_name: str) -> bool:
    """Heuristic: title names other OEMs and does not name the origin entity."""
    title_norm = normalization_key(title or "")
    if not title_norm:
        return False
    if mentions_entity_name(title, entity_name):
        return False
    hits = sum(1 for oem in _OTHER_OEM_MARKERS if mentions_entity_name(title, oem.strip()))
    return hits >= 2


def classify_origin_relevance(
    result: SearchResult,
    entity_name: str,
    *,
    aliases: Optional[dict[str, str]] = None,
    policy: Optional[dict[str, Any]] = None,
) -> tuple[bool, str]:
    """Return (keep, reason). reason is ok / no_entity_mention / weak_context / generic_result."""
    names = origin_name_variants(entity_name, aliases)
    title = result.title or ""
    snippet = result.snippet or ""
    text = result.document_text() or ""

    in_title = mentions_any(title, names)
    in_snippet = mentions_any(snippet, names)
    in_body = mentions_any(text, names)

    if not (in_title or in_snippet or in_body):
        return False, "no_entity_mention"

    if _title_dominated_by_other_oems(title, entity_name) and not in_title:
        # A headline about several OEMs does not invalidate a named upstream
        # relationship in its body. Require explicit local direction for this
        # exception; mere mentions in unrelated product comparisons remain noise.
        if classify_supply_direction(entity_name, text, aliases=names, policy=policy) == UPSTREAM_EXPLICIT:
            return True, "ok"
        return False, "generic_result"

    # Title/snippet path: entity must appear in title or snippet (not only buried later).
    if result.content_source != FULL_TEXT:
        if not (in_title or in_snippet):
            # Soft path for GDELT: query already named the origin; title has supply cues.
            # Full-text fetch still decides Prompt-1 usefulness. Avoid wiping the
            # entire GDELT pool before bodies can be inspected.
            query_has = mentions_any(result.query or "", names)
            if (
                result.provider == "gdelt"
                and query_has
                and has_explicit_relation_language(f"{title}\n{snippet}", policy=policy)
            ):
                return True, "ok"
            return False, "no_entity_mention"
        if has_explicit_relation_language(f"{title}\n{snippet}", policy=policy):
            return True, "ok"
        if in_title and has_explicit_relation_language(text, policy=policy):
            return True, "ok"
        # Soft keep when entity is in the headline even without strong relation words;
        # ranking will demote weak ones. Still require some supply keyword presence.
        blob = f"{title}\n{snippet}".casefold()
        if any(k in blob for k in ("supplier", "supply", "sourcing", "procurement", "agreement")):
            return True, "ok"
        return False, "weak_context"

    # Full text: require meaningful origin presence.
    count = max(_mention_count(text, name) for name in names)
    if count < 1 and not (in_title or in_snippet):
        return False, "no_entity_mention"

    if in_title or in_snippet:
        return True, "ok"

    # Entity only in body: demand supply language near the mention or repeated mentions.
    if any(_entity_near_supply_language(text, name, policy=policy) for name in names):
        return True, "ok"
    if count >= 3 and has_explicit_relation_language(text, policy=policy):
        return True, "ok"
    if count >= 2 and len(text) < 4000:
        return True, "ok"
    return False, "weak_context"


def filter_origin_relevant(
    results: list[SearchResult],
    entity_name: str,
    *,
    aliases: Optional[dict[str, str]] = None,
    policy: Optional[dict[str, Any]] = None,
) -> tuple[list[SearchResult], dict[str, int]]:
    """Keep origin-relevant results; return (kept, reason_counts)."""
    kept: list[SearchResult] = []
    reasons: dict[str, int] = {}
    for result in results:
        ok, reason = classify_origin_relevance(result, entity_name, aliases=aliases, policy=policy)
        reasons[reason] = reasons.get(reason, 0) + 1
        if ok:
            kept.append(result)
            result._origin_relevant = True  # type: ignore[attr-defined]
        else:
            result._origin_relevant = False  # type: ignore[attr-defined]
    return kept, reasons


def title_near_duplicate_key(title: str) -> str:
    """Lightweight title key for near-duplicate diversity (not independence verification)."""
    text = normalization_key(title or "")
    text = re.sub(r"\b(20\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", " ", text)
    return " ".join(text.split())
