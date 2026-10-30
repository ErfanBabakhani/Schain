"""Compatibility-manifest construction for distributed workers and maintenance."""

from __future__ import annotations

import hashlib
from typing import Any

from ..config import REPO_ROOT
from ..durable import critical_configuration
from .schema import SCHEMA_VERSION


def _source_hash(relative: str) -> str:
    return hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()


def build_distributed_manifest(config: dict[str, Any]) -> dict[str, Any]:
    manifest = critical_configuration(config)
    manifest["state_schema_version"] = SCHEMA_VERSION
    manifest["distributed_source_sha256"] = {
        relative: _source_hash(relative)
        for relative in (
            "schain/names.py",
            "schain/kg/extract.py",
            "schain/kg/integrity.py",
            "schain/expansion/frontier.py",
        )
    }
    return manifest


def immutable_manifest_mismatches(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Fields an approved depth/retrieval migration is never allowed to alter."""
    keys = (
        "seeds", "llm", "ingest", "validation", "save", "neo4j_database",
        "ontology_sha256", "prompt_source_sha256",
    )
    return [key for key in keys if old.get(key) != new.get(key)]
