"""Versioned discovery-profile validation shared by workers and maintenance."""

from __future__ import annotations

from typing import Any


DEEP_DISCOVERY_PROFILE = "deep_discovery_v1"
DEEP_TARGETS = {0: 80, 1: 80, 2: 60, 3: 50, 4: 40, 5: 0}
DEEP_MAXIMA = {0: 200, 1: 160, 2: 120, 3: 100, 4: 80, 5: 0}


def profile_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    profile = config.get("discovery_profile") or {}
    expansion = config.get("expansion") or {}
    retrieval = expansion.get("retrieval") or {}
    return {
        "name": str(profile.get("name") or "default"),
        "version": str(profile.get("version") or "unversioned"),
        "max_tier": int(expansion.get("max_tier", 0)),
        "document_limit": int(expansion.get("max_documents_per_experiment") or 0),
        "target_documents_by_tier": dict(retrieval.get("target_documents_by_tier") or {}),
        "max_documents_by_tier": dict(retrieval.get("max_documents_by_tier") or {}),
        "max_full_text_fetches_per_entity": int(
            retrieval.get("max_full_text_fetches_per_entity") or 0
        ),
        "max_title_lookups_per_entity": int(
            retrieval.get("max_title_lookups_per_entity") or 0
        ),
        "max_candidates_per_entity": int(retrieval.get("max_candidates_per_entity") or 0),
        "max_full_text_fetches_per_domain": int(
            retrieval.get("max_full_text_fetches_per_domain") or 0
        ),
        "max_candidates_per_domain": int(retrieval.get("max_candidates_per_domain") or 0),
        "zero_gain_patience": int(retrieval.get("zero_gain_patience") or 0),
        "min_query_families_before_stop": int(
            retrieval.get("min_query_families_before_stop") or 0
        ),
        "min_source_domains_before_stop": int(
            retrieval.get("min_source_domains_before_stop") or 0
        ),
        "recovery_rounds": int(retrieval.get("recovery_rounds") or 0),
    }


def validate_deep_discovery(config: dict[str, Any]) -> dict[str, Any]:
    """Reject silent low-budget overrides of the approved profile."""
    snapshot = profile_snapshot(config)
    if snapshot["version"] != DEEP_DISCOVERY_PROFILE:
        raise ValueError(
            f"deep-discovery resume requires profile {DEEP_DISCOVERY_PROFILE!r}"
        )
    targets = {int(key): int(value) for key, value in snapshot["target_documents_by_tier"].items()}
    maxima = {int(key): int(value) for key, value in snapshot["max_documents_by_tier"].items()}
    if targets != DEEP_TARGETS or maxima != DEEP_MAXIMA:
        raise ValueError("deep-discovery tier document targets/maxima do not match the approved profile")
    checks = {
        "document_limit": 20000,
        "max_full_text_fetches_per_entity": 250,
        "max_title_lookups_per_entity": 100,
        "max_candidates_per_entity": 200,
        "min_query_families_before_stop": 4,
        "min_source_domains_before_stop": 4,
        "recovery_rounds": 4,
    }
    for key, minimum in checks.items():
        if int(snapshot[key]) < minimum:
            raise ValueError(f"deep-discovery {key} must be at least {minimum}")
    retrieval = (config.get("expansion") or {}).get("retrieval") or {}
    if int(retrieval.get("max_documents_per_entity") or 0) < max(DEEP_MAXIMA.values()):
        raise ValueError("max_documents_per_entity would silently cap a tier below its approved maximum")
    return snapshot
