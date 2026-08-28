"""Configuration loading.

Two concerns are deliberately kept apart:

* ``config/config.yaml``  - behaviour (paths, model, chunking). Safe to track in Git.
* ``.env`` / environment  - secrets (Neo4j password, API keys). Never tracked.

If ``config/config.yaml`` does not exist, ``config/config.example.yaml`` is used and
the choice is logged, so a fresh clone runs without a copy step.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
EXAMPLE_CONFIG_PATH = REPO_ROOT / "config" / "config.example.yaml"


def load_config(path: Optional[Path] = None) -> dict[str, Any]:
    """Read the YAML config and load .env into the process environment."""
    load_dotenv(REPO_ROOT / ".env")  # silently does nothing if .env is absent

    if path is None:
        path = DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else EXAMPLE_CONFIG_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    try:
        shown = path.relative_to(REPO_ROOT)
    except ValueError:
        shown = path  # a config outside the repository is fine, just log it in full
    logger.info("loaded config from %s", shown)
    return config


def storage_path(config: dict[str, Any], key: str) -> Path:
    """Resolve one entry of the ``storage:`` config block to an absolute path.

    Parent directories are created so callers can just open the file.
    """
    storage = config.get("storage", {})
    if key not in storage:
        raise KeyError(f"storage.{key} is not set in the config file")
    path = REPO_ROOT / storage[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def neo4j_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Neo4j connection settings: credentials from the environment, database from config."""
    return {
        "uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "username": os.environ.get("NEO4J_USERNAME", "neo4j"),
        "password": os.environ.get("NEO4J_PASSWORD", ""),
        "database": config.get("neo4j", {}).get("database", "neo4j"),
    }
