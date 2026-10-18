"""Legacy evidence-group metadata, retained for existing Parquet/JSON compatibility.

No independent-source acceptance or verification uses this module. ``source_group_id``
is informational only; its domain-level grouping must NOT be used as document dedup,
because a publisher can report many different valid supplier relationships.
"""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

from .reputation import normalize_host, score_source


def normalize_canonical_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(str(url).strip())
    if not parsed.scheme or not parsed.netloc:
        return str(url).strip().rstrip("/") or None
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/") or ""
    return urlunparse((parsed.scheme.lower(), host, path, "", "", ""))


def _normalize_title(title: Optional[str]) -> str:
    text = (title or "").casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def titles_near_duplicate(a: Optional[str], b: Optional[str], threshold: float = 0.92) -> bool:
    na, nb = _normalize_title(a), _normalize_title(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def source_group_id(
    *,
    canonical_source_url: Optional[str] = None,
    retrieval_url: Optional[str] = None,
    source_domain: Optional[str] = None,
    content_hash: Optional[str] = None,
    title: Optional[str] = None,
    reputation: Optional[dict[str, Any]] = None,
) -> str:
    """Stable legacy metadata group, not a support or frontier-acceptance rule.

    Existing values remain readable; URL/content dedup is handled separately.
    """
    reputation = reputation or {}
    independence = reputation.get("independence") or {}

    canonical = normalize_canonical_url(canonical_source_url)
    host = normalize_host(source_domain or canonical or retrieval_url) or "unknown"
    scored = score_source(domain=host, url=canonical or retrieval_url, reputation=reputation)

    # Exact content-hash duplicates are always one group (mirrors / syndication).
    if independence.get("content_hash_exact_match_is_same_group", True) and content_hash:
        return f"hash:{content_hash}"

    # Retrieval wrappers never form their own publisher group.
    if scored.source_class == "retrieval_wrapper":
        basis = _normalize_title(title) or normalize_canonical_url(retrieval_url) or host
        return "wrapper:" + hashlib.sha256(str(basis).encode("utf-8")).hexdigest()[:16]

    if scored.syndicated and independence.get("collapse_press_release_syndication", True):
        basis = _normalize_title(title) or host
        return "syndicated:" + hashlib.sha256(str(basis).encode("utf-8")).hexdigest()[:16]

    # Same publisher domain normally contributes at most one independent group.
    if host != "unknown" and independence.get("one_group_per_registered_domain", True):
        return f"domain:{host}"

    if canonical:
        return f"url:{canonical}"
    basis = _normalize_title(title) or (retrieval_url or "unknown")
    return "opaque:" + hashlib.sha256(str(basis).encode("utf-8")).hexdigest()[:16]


def merge_group_ids(
    existing: dict[str, str],
    *,
    content_hash: Optional[str],
    title: Optional[str],
    group_id: str,
    reputation: Optional[dict[str, Any]] = None,
) -> str:
    """Collapse a new group into an existing one when hash/title say they are the same."""
    reputation = reputation or {}
    threshold = float(
        (reputation.get("independence") or {}).get("near_duplicate_similarity_threshold", 0.92)
    )
    if content_hash and content_hash in existing:
        return existing[content_hash]
    for key, other_id in list(existing.items()):
        if key.startswith("title:") and titles_near_duplicate(key[6:], title, threshold):
            return other_id
    if content_hash:
        existing[content_hash] = group_id
    if title:
        existing[f"title:{_normalize_title(title)}"] = group_id
    return group_id
