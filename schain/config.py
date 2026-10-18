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


# --- search limits: global defaults < tier defaults < per-query override ------------

# Fields resolved independently through the hierarchy. Only these are inheritable.
INHERITABLE_QUERY_FIELDS = (
    "max_results", "max_results_by_provider", "providers", "min_snippet_chars",
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
    given a high limit while Wikipedia stays small, without duplicating every query.
    """
    by_provider = spec.get("max_results_by_provider") or {}
    return int(by_provider.get(provider, spec.get("max_results", 20)))


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
        if int(expansion.get(key, 0)) < 0:
            raise ValueError(f"expansion.{key} must be nonnegative")
    if float(config.get("runtime", {}).get("max_minutes", 110)) < 0:
        raise ValueError("runtime.max_minutes must be nonnegative")
    retrieval = expansion.get("retrieval", {})
    for key in ("target_documents_per_entity", "max_documents_per_entity"):
        if key in retrieval and int(retrieval[key]) < 0:
            raise ValueError(f"expansion.retrieval.{key} must be nonnegative")
    for key in ("target_documents_by_tier", "max_documents_by_tier"):
        if any(int(v) < 0 for v in (retrieval.get(key) or {}).values()):
            raise ValueError(f"expansion.retrieval.{key} values must be nonnegative")
