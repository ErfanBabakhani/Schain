"""Informational source reputation: domain matching and content-quality metadata.

Reputation never gates acceptance, frontier promotion, or retrieval rank. Unknown
domains get policy.unknown.base_score and remain usable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import yaml

from ..config import REPO_ROOT, resolve_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceReputation:
    domain: str
    known: bool
    source_class: str
    base_score: float
    effective_score: float
    count_for_verification: bool
    resolve_original_source: bool
    syndicated: bool


def normalize_host(url_or_host: Optional[str]) -> Optional[str]:
    """Extract a lowercase hostname without a leading ``www.``."""
    if not url_or_host:
        return None
    text = str(url_or_host).strip().lower()
    if not text:
        return None
    if "://" not in text:
        host = text.split("/")[0].split(":")[0]
    else:
        host = (urlparse(text).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def clamp_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def load_source_reputation(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Load and lightly validate ``source_reputation.yaml``.

    When ``source_reputation.enabled`` is false, returns an empty dict and callers
    treat every domain as unknown with the default fallback score of 50.
    """
    section = (config or {}).get("source_reputation") or {}
    if section.get("enabled", True) is False:
        return {}
    path = resolve_path(section.get("path", "config/source_reputation.yaml"))
    if not Path(path).exists():
        if section.get("enabled"):
            raise FileNotFoundError(
                f"source_reputation.enabled is true but file not found: {path}"
            )
        logger.warning("source reputation file missing at %s; using unknown defaults", path)
        return {}
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    _validate_reputation_config(data, path)
    return data


def _validate_reputation_config(data: dict[str, Any], path: Path) -> None:
    policy = data.get("policy") or {}
    if "unknown" not in policy:
        raise ValueError(f"{path}: policy.unknown is required")
    for key in ("minimum_countable_score", "strong_source_score", "trusted_source_score",
                "hard_reject_below"):
        if key in policy:
            score = float(policy[key])
            if not 0 <= score <= 100:
                raise ValueError(f"{path}: policy.{key} must be in 0..100, got {score}")
    # Legacy verification / independence blocks may remain in YAML as documentation
    # only. They no longer drive runtime decisions; skip strict cross-checks on them.
    threshold = (data.get("independence") or {}).get("near_duplicate_similarity_threshold")
    if threshold is not None and not 0 <= float(threshold) <= 1:
        raise ValueError(f"{path}: independence.near_duplicate_similarity_threshold must be 0..1")
    classes = set((data.get("source_classes") or {}).keys())
    for domain, entry in (data.get("domains") or {}).items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: domains.{domain} must be a mapping")
        score = entry.get("score")
        if score is not None and not 0 <= float(score) <= 100:
            raise ValueError(f"{path}: domains.{domain}.score must be in 0..100")
        cls = entry.get("class")
        if cls and classes and cls not in classes:
            raise ValueError(f"{path}: domains.{domain}.class {cls!r} is not in source_classes")


def match_domain(host: Optional[str], reputation: dict[str, Any]) -> Optional[tuple[str, dict]]:
    """Return the most specific configured domain matching ``host``, or None."""
    if not host or not reputation:
        return None
    domains = reputation.get("domains") or {}
    if host in domains:
        return host, domains[host]
    # Prefer the longest configured domain that is a proper suffix match.
    candidates = [
        domain for domain in domains
        if host == domain or host.endswith("." + domain)
    ]
    if not candidates:
        return None
    best = max(candidates, key=len)
    return best, domains[best]


def score_source(
    *,
    url: Optional[str] = None,
    domain: Optional[str] = None,
    content_source: str = "full_text",
    canonical_resolved: bool = False,
    reputation: Optional[dict[str, Any]] = None,
) -> SourceReputation:
    """Compute base and effective scores for one URL/domain."""
    reputation = reputation or {}
    host = normalize_host(domain or url)
    policy = reputation.get("policy") or {}
    unknown = policy.get("unknown") or {"base_score": 50, "tier": "unknown"}
    classes = reputation.get("source_classes") or {}
    content = reputation.get("content_quality") or {}

    matched = match_domain(host, reputation)
    if matched is None:
        base = float(unknown.get("base_score", 50))
        source_class = str(unknown.get("tier", "unknown"))
        class_meta = classes.get(source_class) or classes.get("unknown") or {}
        known = False
        resolve_original = False
        syndicated = False
        count = bool(class_meta.get("count_for_verification", True))
        matched_domain = host or "unknown"
    else:
        matched_domain, entry = matched
        source_class = str(entry.get("class") or "unknown")
        class_meta = classes.get(source_class) or {}
        base = float(entry.get("score", class_meta.get("default_score", unknown.get("base_score", 50))))
        known = True
        resolve_original = bool(entry.get("resolve_original_source", False))
        syndicated = bool(entry.get("syndicated", False))
        count = bool(entry.get(
            "count_for_verification",
            class_meta.get("count_for_verification", True),
        ))

    effective = base
    if content_source == "full_text":
        effective += float(content.get("full_text_bonus", 0))
    elif content_source == "title_snippet":
        effective -= float(content.get("title_snippet_penalty", 0))
        cap = content.get("title_snippet_score_cap")
        if cap is not None:
            effective = min(effective, float(cap))
    if canonical_resolved:
        effective += float(content.get("canonical_url_resolved_bonus", 0))

    return SourceReputation(
        domain=matched_domain,
        known=known,
        source_class=source_class,
        base_score=clamp_score(base),
        effective_score=clamp_score(effective),
        count_for_verification=count and source_class != "retrieval_wrapper",
        resolve_original_source=resolve_original,
        syndicated=syndicated,
    )


def is_wrapper_domain(domain: Optional[str], reputation: Optional[dict[str, Any]] = None) -> bool:
    scored = score_source(domain=domain, reputation=reputation or {})
    return scored.source_class == "retrieval_wrapper"
