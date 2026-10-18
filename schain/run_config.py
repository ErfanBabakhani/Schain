"""Persist the expansion experiment identity independently of later export config.

Operational ceilings may change on resume and are recorded as revisions. Seeds,
depth, extraction/validation policy and model may not silently change underneath
completed checkpoints. Legacy artifacts are readable but cannot acquire historical
provenance retroactively.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from . import __version__
from .checkpoint import atomic_write_json
from .config import REPO_ROOT, resolve_path, seed_entities, seed_groups
from .ontology import load_ontology

logger = logging.getLogger(__name__)
_PRIVATE = re.compile(r"password|secret|api[_-]?key|access[_-]?token|authorization", re.I)


def snapshot_path(config: dict[str, Any]) -> Path:
    storage = config.get("storage", {})
    return resolve_path(storage.get("run_config", str(
        Path(storage.get("state_db", "data/state/state.db")).parent / "run_config.json"
    )))


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items() if not _PRIVATE.search(str(k))}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        parts = urlsplit(value)
        return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))
    return value


def _policy_hash(config: dict[str, Any]) -> str | None:
    path = resolve_path(config.get("validation", {}).get("evidence", {}).get(
        "policy_file", "config/relation_evidence.yaml"))
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _safe_config(config: dict[str, Any]) -> dict[str, Any]:
    safe = _safe({key: config[key] for key in (
        "seeds", "expansion", "runtime", "llm", "concurrency", "ingest",
        "validation", "source_reputation", "verification",
    ) if key in config})
    safe["ontology"] = _safe(load_ontology(config))
    if config.get("llm", {}).get("provider") == "local" and os.environ.get("SCHAIN_VLLM_MODEL"):
        safe.setdefault("llm", {})["local_model_weights_override"] = os.environ["SCHAIN_VLLM_MODEL"]
    return safe


def _identity(safe: dict[str, Any], policy_hash: str | None) -> dict[str, Any]:
    llm = dict(safe.get("llm", {}))
    # Endpoint location / transport timeout are operational; generation settings are not.
    for key in ("base_url", "request_timeout_seconds", "max_retries"):
        llm.pop(key, None)
    return {
        "seeds": safe.get("seeds", {}),
        "max_tier": safe.get("expansion", {}).get("max_tier", 5),
        "llm": llm, "ingest": safe.get("ingest", {}),
        "validation": safe.get("validation", {}), "ontology": safe.get("ontology", {}),
        "relation_policy_sha256": policy_hash, "code_version": __version__,
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _has_legacy_documents(config: dict[str, Any]) -> bool:
    path = resolve_path(config.get("storage", {}).get("state_db", "data/state/state.db"))
    if not path.exists():
        return False
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            return bool(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    except sqlite3.OperationalError:
        return False


def ensure_run_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """Create/validate an expansion snapshot after explicit CLI overrides.

The caller must clear state for ``--restart`` first. Never overwrite a conflicting
snapshot merely because a config still contains restart_from_scratch=true.
"""
    path = snapshot_path(config)
    safe = _safe_config(config)
    policy_hash = _policy_hash(config)
    identity = _identity(safe, policy_hash)
    fingerprint = _digest(identity)
    now = datetime.now(timezone.utc).isoformat()
    if path.exists():
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if snapshot.get("identity_sha256") != fingerprint:
            previous = snapshot.get("experiment_identity", {})
            changed = [key for key, value in identity.items() if previous.get(key) != value]
            raise ValueError("Expansion checkpoint configuration changed (" + ", ".join(changed)
                             + "). Use --restart or a separate storage directory for a new experiment.")
        if snapshot.get("effective_config") != safe:
            snapshot.setdefault("config_revisions", []).append({
                "time": now, "effective_config": safe,
                "reason": "operational settings changed on resume",
            })
            snapshot["effective_config"] = safe
            snapshot["last_config_update"] = now
            _add_summary(snapshot, config)
            atomic_write_json(path, snapshot)
        return snapshot
    commit = None
    if (REPO_ROOT / ".git").exists():
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                    capture_output=True, text=True, timeout=3, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    legacy = _has_legacy_documents(config)
    if legacy:
        logger.warning("Existing documents have no run snapshot; earlier configuration is unknown. "
                       "Recording legacy provenance as unverified; use --restart for a clean experiment.")
    snapshot = {
        "schema_version": 1, "started_at": now, "code_version": __version__,
        "git_commit": commit, "legacy_artifacts_unverified": legacy,
        "effective_config": safe, "experiment_identity": identity,
        "identity_sha256": fingerprint, "relation_policy_sha256": policy_hash,
        "config_revisions": [],
    }
    _add_summary(snapshot, config)
    atomic_write_json(path, snapshot)
    return snapshot


def _add_summary(snapshot: dict[str, Any], config: dict[str, Any]) -> None:
    expansion = config.get("expansion") or {}
    retrieval = expansion.get("retrieval") or {}

    def _str_keys(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): _str_keys(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_str_keys(v) for v in value]
        return value

    snapshot.update({
        "effective_seed_groups": seed_groups(config), "seeds": seed_entities(config),
        "max_tier": expansion.get("max_tier", 5),
        "document_budgets": _safe(retrieval),
        "global_document_hard_cap": (
            expansion.get("max_documents_per_run")
            if expansion.get("max_documents_per_run") is not None
            else expansion.get("max_documents")
        ),
        "max_documents_per_run": (
            expansion.get("max_documents_per_run")
            if expansion.get("max_documents_per_run") is not None
            else expansion.get("max_documents")
        ),
        "scheduling_batch_size_by_tier": _str_keys(
            expansion.get("scheduling_batch_size_by_tier")
            or expansion.get("max_expandable_entities_by_tier")
        ),
        "target_documents_by_tier": _str_keys(retrieval.get("target_documents_by_tier")),
        "max_documents_by_tier": _str_keys(retrieval.get("max_documents_by_tier")),
        "max_expandable_entities_by_tier": _str_keys(expansion.get("max_expandable_entities_by_tier")),
        "query_planner_policy": {
            "enabled": retrieval.get("query_planner_enabled", True),
            "max_categories": retrieval.get("query_planner_max_categories"),
            "max_queries": retrieval.get("query_planner_max_queries"),
        },
        "search_escalation_policy": {
            "zero_yield_rescue": retrieval.get("zero_yield_rescue", True),
            "semantic_direction_adjudication": retrieval.get("semantic_direction_adjudication", True),
            "evidence_second_pass": retrieval.get("evidence_second_pass", True),
            "max_title_lookups_per_entity": retrieval.get("max_title_lookups_per_entity"),
        },
        "retrieval_settings": _safe(expansion),
        "model": config.get("llm", {}).get("model"),
        "local_model_weights_override": snapshot["effective_config"].get("llm", {}).get("local_model_weights_override"),
        "runtime_cap_minutes": config.get("runtime", {}).get("max_minutes"),
        "relation_policy_sha256": snapshot.get("relation_policy_sha256"),
    })


def effective_export_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Use saved experiment semantics; retain current artifact/output paths."""
    path = snapshot_path(config)
    if not path.exists():
        return config, None
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != 1 or not isinstance(snapshot.get("effective_config"), dict):
        raise ValueError(f"Invalid expansion run snapshot: {path}")
    effective = copy.deepcopy(config)
    effective.update(copy.deepcopy(snapshot["effective_config"]))
    return effective, snapshot
