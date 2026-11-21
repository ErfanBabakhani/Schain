"""Configuration loading and the search-limit precedence rules.

Three concerns are kept apart:

* ``config/*.yaml``       behaviour. Tracked in Git, no secrets.
* ``.env`` / environment  secrets (Neo4j password, API keys). Never tracked.
* ``config/ontology.yaml`` the static ontology, so the entity and relation types are
  stated in one place rather than in the code.

A config file may list ``include:`` paths. Those are loaded first and the including
file's own keys are merged over them, so ``config/ingest.yaml`` and
``config/expansion.yaml`` share one copy of the LLM, Neo4j and storage settings.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` onto ``base``. Dicts merge key by key; anything else replaces."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(value: str | Path) -> Path:
    """Resolve a config path: absolute stays as is, relative is repo-root relative."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path, _seen: Optional[set[Path]] = None) -> dict[str, Any]:
    """Read a YAML config, apply its ``include:`` chain, and load .env into the environment."""
    load_dotenv(REPO_ROOT / ".env")  # does nothing if .env is absent

    path = resolve_path(path).resolve()
    seen = _seen if _seen is not None else set()
    if path in seen:
        raise ValueError(f"circular config include detected at {path}")
    seen.add(path)

    try:
        raw = _read_yaml(path)
        includes = raw.pop("include", []) or []
        runtime_overlay = raw.pop("runtime_overlay", False)
        if runtime_overlay and (
            "seeds" in raw or "max_tier" in (raw.get("expansion") or {})
        ):
            raise ValueError(
                f"Runtime overlay {path.name} cannot replace seeds or max_tier; "
                "edit expansion.yaml or use explicit CLI --seed/--seeds/--max-tier."
            )
        config: dict[str, Any] = {}
        for included in includes:
            config = _deep_merge(config, load_config(included, seen))
        config = _deep_merge(config, raw)
    finally:
        # Track the recursion stack, not every file ever included (diamond includes).
        seen.remove(path)

    if _seen is None:
        logger.info("loaded config %s (includes: %s)", path.name, includes or "none")
    return config


def storage_path(config: dict[str, Any], key: str) -> Path:
    """Resolve one entry of the ``storage:`` block, creating its parent directory."""
    storage = config.get("storage", {})
    if key not in storage:
        raise KeyError(f"storage.{key} is not set in the config file")
    path = resolve_path(storage[key])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def storage_dir(config: dict[str, Any], key: str) -> Path:
    """Resolve a ``storage:`` entry that names a directory, creating it."""
    path = resolve_path(config.get("storage", {})[key])
    path.mkdir(parents=True, exist_ok=True)
    return path


def neo4j_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Neo4j connection settings: credentials from the environment, database from config."""
    return {
        "uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "username": os.environ.get("NEO4J_USERNAME", "neo4j"),
        "password": os.environ.get("NEO4J_PASSWORD", ""),
        "database": config.get("neo4j", {}).get("database", "neo4j"),
        "batch_size": config.get("neo4j", {}).get("batch_size", 1000),
    }


def neo4j_is_configured() -> bool:
    """True when a Neo4j password is present; used to skip integration tests cleanly."""
    return bool(os.environ.get("NEO4J_PASSWORD"))


def save_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Return and validate the explicit graph-database save policy."""
    raw = config.get("save") or {}
    settings = {
        "enabled": bool(raw.get("enabled", False)),
        "backend": str(raw.get("backend", "neo4j")),
        "mode": str(raw.get("mode", "incremental")),
        "max_attempts": max(1, int(raw.get("max_attempts", 3))),
        "retry_backoff_seconds": max(0.0, float(raw.get("retry_backoff_seconds", 1.0))),
    }
    if settings["backend"] != "neo4j":
        raise ValueError("save.backend must be 'neo4j'")
    if settings["mode"] != "incremental":
        raise ValueError("save.mode must be 'incremental'")
    return settings


# --- search limits: global defaults < tier defaults < per-query override ------------

# Fields resolved independently through the hierarchy. Only these are inheritable.
INHERITABLE_QUERY_FIELDS = (
    "max_results", "max_results_cap", "max_results_by_provider", "providers",
    "min_snippet_chars",
)


def resolve_tier_queries(config: dict[str, Any], tier: int) -> list[dict[str, Any]]:
    """Build the query specifications for one tier.

    Each returned entry has a ``query`` template plus every inheritable field resolved
    by "closest value wins": a per-query override beats the tier default, which beats
    the global default. Fields are resolved one by one, so a tier that overrides only
    ``max_results`` still inherits the global ``providers``.

    ``max_results`` is a limit per provider per query, not a cumulative document
    target. ``max_results_by_provider`` overrides it for the providers it names.

    A tier marked ``expand: false`` yields no queries at all: it is terminal, so its
    entities are stored but never searched.
    """
    expansion = config.get("expansion", {})
    if "max_tier" in expansion and tier >= int(expansion["max_tier"]):
        return []
    search = expansion.get("search", {})
    global_defaults = search.get("defaults", {}) or {}
    tiers = search.get("tiers", {}) or {}
    # YAML may key tiers as ints or strings; accept both.
    tier_config = tiers.get(tier, tiers.get(str(tier), {})) or {}
    if tier_config.get("expand", True) is False:
        return []
    tier_defaults = tier_config.get("defaults", {}) or {}

    resolved: list[dict[str, Any]] = []
    for entry in tier_config.get("queries", []) or []:
        entry = dict(entry)
        spec: dict[str, Any] = {"query": entry.pop("query"), "tier": tier}
        for field in INHERITABLE_QUERY_FIELDS:
            if field == "max_results_by_provider":
                inherited: dict[str, Any] = {}
                for source in (global_defaults, tier_defaults, entry):
                    inherited.update(source.get(field) or {})
                if inherited:
                    spec[field] = inherited
                continue
            for source in (entry, tier_defaults, global_defaults):  # closest first
                if field in source and source[field] is not None:
                    spec[field] = source[field]
                    break
        resolved.append(spec)
    return resolved


def query_limit(spec: dict[str, Any], provider: str) -> int:
    """Candidate ceiling for one provider on one query.

    A per-provider entry wins over the plain ``max_results``, so news providers can be
    given a high limit while Wikipedia stays small. ``max_results_cap`` is an optional
    user-selected ceiling that also applies to provider-specific values.
    """
    by_provider = spec.get("max_results_by_provider") or {}
    result_limit = int(by_provider.get(provider, spec.get("max_results", 20)))
    if spec.get("max_results_cap") is not None:
        result_limit = min(result_limit, int(spec["max_results_cap"]))
    return result_limit


def seed_entities(config: dict[str, Any]) -> list[str]:
    """Flatten the ``seeds:`` block into one ordered, de-duplicated list of names.

    Only active (uncommented) YAML entries are present after ``yaml.safe_load``.
    Commented seed groups never appear in the config dict and are not loaded.
    """
    seeds = config.get("seeds", {}) or {}
    if isinstance(seeds, list):
        groups: list[Any] = [seeds]
    else:
        groups = list(seeds.values())
    names: list[str] = []
    for group in groups:
        for name in group or []:
            if name and name not in names:
                names.append(str(name))
    return names


def seed_groups(config: dict[str, Any]) -> dict[str, list[str]]:
    """Return active seed groups as ``{group_name: [names...]}``."""
    seeds = config.get("seeds", {}) or {}
    if isinstance(seeds, list):
        return {"seeds": [str(name) for name in seeds if name]}
    result: dict[str, list[str]] = {}
    for group_name, names in seeds.items():
        cleaned = [str(name) for name in (names or []) if name]
        if cleaned:
            result[str(group_name)] = cleaned
    return result


def apply_runtime_overrides(
    config: dict[str, Any],
    overrides: Optional[dict[str, Any]] = None,
    *,
    protect_seeds: bool = True,
    protect_max_tier: bool = True,
) -> dict[str, Any]:
    """Merge runtime overrides onto config.

    By default ``seeds`` from expansion.yaml are preserved. Runtime overlays must not
    silently replace seeds. Pass ``protect_seeds=False`` only for an explicit user
    seed override (CLI ``--seeds``); in that case ``seeds`` are replaced wholesale.
    """
    if not overrides:
        return config
    overrides = copy.deepcopy(overrides)
    if protect_max_tier:
        overrides.get("expansion", {}).pop("max_tier", None)
    if protect_seeds:
        overrides.pop("seeds", None)
        return _deep_merge(config, overrides)
    # Explicit seed override: replace seeds entirely, then merge the rest.
    seed_override = overrides.pop("seeds", None)
    merged = _deep_merge(config, overrides)
    if seed_override is not None:
        merged["seeds"] = copy.deepcopy(seed_override)
    return merged


def log_effective_expansion_config(config: dict[str, Any]) -> None:
    """Log the effective EXPAND settings so seed/policy mistakes are visible."""
    expansion = config.get("expansion") or {}
    retrieval = expansion.get("retrieval") or {}
    seeds = seed_entities(config)
    groups = seed_groups(config)
    providers = [name for name in expansion.get("search", {}).get("defaults", {}).get("providers", [])
                 if expansion.get("sources", {}).get(name, {}).get("enabled", True)]
    logger.info(
        "expansion config: seeds=%s seed_count=%s seed_groups=%s max_tier=%s max_documents=%s "
        "target_documents_by_tier=%s max_documents_by_tier=%s max_documents_per_entity=%s "
        "frontier_caps=%s providers=%s model=%s runtime_minutes=%s",
        seeds,
        len(seeds),
        {name: len(members) for name, members in groups.items()},
        expansion.get("max_tier"),
        expansion.get("max_documents"),
        retrieval.get("target_documents_by_tier", retrieval.get("target_documents_per_entity")),
        retrieval.get("max_documents_by_tier"),
        retrieval.get("max_documents_per_entity"),
        expansion.get("max_expandable_entities_by_tier"),
        providers,
        config.get("llm", {}).get("model"),
        config.get("runtime", {}).get("max_minutes"),
    )


def validate_expansion_config(config: dict[str, Any]) -> None:
    """Catch invalid ceilings before a restart can clear any checkpoint files."""
    expansion = config.get("expansion", {})
    for key in ("max_tier", "max_documents"):
        if expansion.get(key) is not None and int(expansion.get(key, 0)) < 0:
            raise ValueError(f"expansion.{key} must be nonnegative")
    if float(config.get("runtime", {}).get("max_minutes", 110)) < 0:
        raise ValueError("runtime.max_minutes must be nonnegative")
    retrieval = expansion.get("retrieval", {})
    for key in ("target_documents_per_entity", "max_documents_per_entity"):
        if key in retrieval and retrieval[key] is not None and int(retrieval[key]) < 0:
            raise ValueError(f"expansion.retrieval.{key} must be nonnegative")
    for key in ("target_documents_by_tier", "max_documents_by_tier"):
        if any(int(v) < 0 for v in (retrieval.get(key) or {}).values()):
            raise ValueError(f"expansion.retrieval.{key} values must be nonnegative")
    selection = expansion.get("selection") or {}
    if selection.get("policy", "legacy") not in {"legacy", "adaptive_linucb"}:
        raise ValueError("expansion.selection.policy must be legacy or adaptive_linucb")
    if selection.get("candidate_pool_policy") not in (None, "title_snippet"):
        raise ValueError("expansion.selection.candidate_pool_policy must be title_snippet when set")
    allowed_actions = selection.get("allowed_actions")
    if allowed_actions is not None:
        if (not isinstance(allowed_actions, list) or not allowed_actions or
                any(action not in ("snippet", "full_text", "both") for action in allowed_actions) or
                len(set(allowed_actions)) != len(allowed_actions)):
            raise ValueError(
                "expansion.selection.allowed_actions must be a nonempty unique list "
                "of snippet, full_text, and both"
            )
    if (selection.get("total_input_tokens") is not None and
            int(selection["total_input_tokens"]) <= 0):
        raise ValueError("expansion.selection.total_input_tokens must be positive")
    if selection.get("policy") == "adaptive_linucb":
        if int(retrieval.get("max_candidates_per_entity", 0) or 0) <= 0:
            raise ValueError("adaptive_linucb requires a positive max_candidates_per_entity")
        if selection.get("total_input_tokens") is None:
            raise ValueError("adaptive_linucb requires expansion.selection.total_input_tokens")
        if selection.get("target_tokens_per_candidate") is None:
            raise ValueError("adaptive_linucb requires expansion.selection.target_tokens_per_candidate")
        if int(selection["total_input_tokens"]) <= 0 or int(selection["target_tokens_per_candidate"]) <= 0:
            raise ValueError("adaptive token budget and target must be positive")
        if not 1 <= int(selection.get("snippet_batch_size", 10)) <= 10:
            raise ValueError("adaptive snippet_batch_size must be between 1 and 10")
        priors = selection.get("initial_action_prior") or {}
        if priors and (float(priors.get("skip", 0)) <= 0 or
                       any(float(value) < 0 for value in priors.values())):
            raise ValueError("adaptive action priors must be nonnegative and skip must be positive")
        if float(selection.get("cost_reference_tokens", 1000)) <= 0:
            raise ValueError("expansion.selection.cost_reference_tokens must be positive")
        if float(selection.get("warmup_temperature", 1.0)) <= 0:
            raise ValueError("expansion.selection.warmup_temperature must be positive")
        if float(selection.get("ridge_lambda", 1.0)) <= 0 or float(selection.get("linucb_alpha", .5)) < 0:
            raise ValueError("adaptive ridge_lambda must be positive and linucb_alpha nonnegative")
        rewards = selection.get("reward_weights") or {}
        if float(rewards.get("clip_max", 5.0)) <= 0 or float(rewards.get("new_supply_edge", 1.0)) < 0:
            raise ValueError("adaptive new_supply_edge reward must be nonnegative and clip_max positive")
