"""Deterministic entity-name normalization.

Kept in its own module because the state layer, the resolver and the expansion frontier
all need the same rules, and none of them should import each other to get them.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# Company legal forms, stripped before comparing names.
LEGAL_SUFFIXES = (
    "incorporated", "corporation", "company", "holdings", "holding", "group",
    "limited", "inc", "ltd", "llc", "plc", "corp", "co", "ag", "gmbh", "sa",
    "nv", "bv", "ab", "as", "oyj", "kk", "pte", "spa", "srl", "sas", "kgaa",
)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalization_key(name: str) -> str:
    """Comparison key: accent-free, lower-case, punctuation-free, no legal suffix.

    "BMW AG", "bmw ag." and "BMW" all reduce to "bmw", so they never become separate
    entities and never reach the LLM.
    """
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


def mention_key(label: str, mention: str) -> str:
    """Cache key for a resolved mention."""
    return f"{label.lower()}:{normalization_key(mention)}"
