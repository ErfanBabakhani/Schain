"""Configuration loading and the search-limit precedence rules.

Three concerns are kept apart:

* ``config/*.yaml``       behaviour. Tracked in Git, no secrets.
* ``.env`` / environment  secrets (Neo4j password, API keys). Never tracked.
* ``config/ontology.yaml`` the static ontology, so the entity and relation types are
  stated in one place rather than in the code.

A config file may list ``include:`` paths (a single path string or a YAML list).
Those are loaded first and the including file's own keys are merged over them, so
``config/ingest.yaml`` and ``config/expansion.yaml`` share one copy of the LLM, Neo4j
and storage settings.
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


def _normalize_includes(value: Any) -> list[str]:
    """Normalize ``include:`` to a list of path strings.

    YAML accepts either a single path or a list. A bare string must not be iterated as
    characters (``include: ingest.yaml`` would otherwise try to open ``i``, ``n``, …).
    """
    if value is None or value == [] or value == "":
        return []
    if isinstance(value, Path):
        return [str(value)]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        paths: list[str] = []
        for entry in value:
            if isinstance(entry, (str, Path)):
                paths.append(str(entry))
            else:
                raise ValueError(
                    "include must be a path string or a list of path strings"
                )
        return paths
    raise ValueError("include must be a path string or a list of path strings")


def load_config(path: str | Path, _seen: Optional[set[Path]] = None) -> dict[str, Any]:
    """Read a YAML config, apply its ``include:`` chain, and load .env into the environment."""
    load_dotenv(REPO_ROOT / ".env")  # does nothing if .env is absent

    path = resolve_path(path).resolve()
    seen = _seen or set()
    if path in seen:
        raise ValueError(f"circular config include detected at {path}")
    seen.add(path)

    raw = _read_yaml(path)
    includes = _normalize_includes(raw.pop("include", None))
    config: dict[str, Any] = {}
    for included in includes:
        config = _deep_merge(config, load_config(included, seen))
    config = _deep_merge(config, raw)

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
    search = config.get("expansion", {}).get("search", {})
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
    """Flatten the ``seeds:`` block into one ordered, de-duplicated list of names."""
    seeds = config.get("seeds", {}) or {}
    if isinstance(seeds, list):
        groups: list[Any] = [seeds]
    else:
        groups = list(seeds.values())
    names: list[str] = []
    for group in groups:
        for name in group or []:
            if name not in names:
                names.append(name)
    return names
