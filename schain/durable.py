"""Durable experiment snapshots for ephemeral runtimes such as Google Colab.

The live SQLite database stays on local disk.  Snapshots use SQLite's backup API,
copy the file artifacts into immutable generations, and publish a small LATEST
pointer only after every copied file has been hashed.  Restoring an older snapshot
is safe because Neo4j writes are idempotent and unresolved document commits replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .checkpoint import atomic_write_json
from .config import REPO_ROOT, resolve_path, seed_groups


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _redact(value: Any, key: str = "") -> Any:
    lowered = key.casefold()
    if any(marker in lowered for marker in ("password", "secret", "token", "api_key")):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _file_hash(path: Path) -> Optional[str]:
    try:
        return _sha256(path) if path.is_file() else None
    except OSError:
        return None


def critical_configuration(config: dict[str, Any]) -> dict[str, Any]:
    """Configuration and source fingerprints that define experiment identity."""
    ontology_path = resolve_path(config.get("ontology_file", "config/ontology.yaml"))
    prompt_files = [
        REPO_ROOT / "schain" / "ontology.py",
        REPO_ROOT / "schain" / "kg" / "validate.py",
        REPO_ROOT / "schain" / "kg" / "resolve.py",
    ]
    llm = config.get("llm") or {}
    ingest = config.get("ingest") or {}
    return {
        "seeds": seed_groups(config),
        "expansion": _redact(config.get("expansion") or {}),
        "llm": {
            key: llm.get(key) for key in (
                "provider", "backend", "model", "enable_thinking", "temperature",
                "json_mode", "structured_output", "max_tokens",
            )
        },
        "ingest": {
            key: ingest.get(key) for key in (
                "chunk_size", "chunk_overlap", "max_chunks_per_document", "on_error",
            )
        },
        "validation": _redact(config.get("validation") or {}),
        "save": _redact(config.get("save") or {}),
        "neo4j_database": (config.get("neo4j") or {}).get("database", "neo4j"),
        "ontology_sha256": _file_hash(ontology_path),
        "prompt_source_sha256": {
            str(path.relative_to(REPO_ROOT)): _file_hash(path) for path in prompt_files
        },
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            text=True, timeout=3, check=True,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class DurableCheckpointManager:
    """Own one durable experiment, its lock, and immutable snapshot generations."""

    def __init__(self, config: dict[str, Any], *, mode: str) -> None:
        settings = config.get("checkpoint_store") or {}
        experiment_id = str(settings.get("experiment_id") or "").strip()
        directory = settings.get("directory")
        if not experiment_id or not directory:
            raise ValueError(
                "checkpoint_store.enabled requires experiment_id and directory"
            )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", experiment_id):
            raise ValueError("checkpoint_store.experiment_id contains unsafe characters")
        self.config = config
        self.mode = mode
        self.experiment_id = experiment_id
        self.root = resolve_path(directory) / experiment_id
        self.snapshots = self.root / "snapshots"
        self.metadata_path = self.root / "experiment.json"
        self.latest_path = self.root / "LATEST.json"
        self.lock_path = self.root / "ACTIVE.lock"
        self.every = max(1, int(settings.get("snapshot_every_documents", 1)))
        self.keep = max(1, int(settings.get("keep_snapshots", 3)))
        self.lock_stale_seconds = max(60, int(settings.get("lock_stale_seconds", 21600)))
        self.token = uuid.uuid4().hex
        self._documents_since_snapshot = 0
        self.last_snapshot: Optional[str] = None
        self.restored_snapshot: Optional[str] = None

    @classmethod
    def from_config(
        cls, config: dict[str, Any], *, mode: str,
    ) -> Optional["DurableCheckpointManager"]:
        if not bool((config.get("checkpoint_store") or {}).get("enabled", False)):
            return None
        return cls(config, mode=mode)

    def _acquire_lock(self, *, force_unlock: bool = False) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            try:
                existing = json.loads(self.lock_path.read_text(encoding="utf-8"))
                age = time.time() - float(existing.get("created_epoch", time.time()))
            except (OSError, ValueError, TypeError):
                age = 0
            if age >= self.lock_stale_seconds or force_unlock:
                self.lock_path.unlink(missing_ok=True)
            else:
                raise RuntimeError(
                    f"experiment {self.experiment_id!r} is locked by another session; "
                    "wait for the stale-lock interval or use --force-checkpoint-unlock"
                )
        payload = {
            "experiment_id": self.experiment_id,
            "token": self.token,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "created_at": _now(),
            "created_epoch": time.time(),
        }
        try:
            descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(
                f"experiment {self.experiment_id!r} was locked concurrently"
            ) from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())

    def release(self) -> None:
        try:
            payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
            if payload.get("token") == self.token:
                self.lock_path.unlink(missing_ok=True)
        except (OSError, ValueError):
            return

    def prepare(
        self, *, new_run: bool, force_unlock: bool = False,
    ) -> dict[str, Any]:
        self._acquire_lock(force_unlock=force_unlock)
        try:
            critical = critical_configuration(self.config)
            fingerprint = _fingerprint(critical)
            if new_run:
                if self.metadata_path.exists():
                    raise RuntimeError(
                        f"experiment {self.experiment_id!r} already exists; choose a new "
                        "EXPERIMENT_ID or resume it"
                    )
                self.snapshots.mkdir(parents=True, exist_ok=True)
                atomic_write_json(self.metadata_path, {
                    "schema_version": "1.0",
                    "experiment_id": self.experiment_id,
                    "mode": self.mode,
                    "created_at": _now(),
                    "git_commit": _git_commit(),
                    "critical_fingerprint": fingerprint,
                    "critical_configuration": critical,
                    "configuration_snapshot": _redact(self.config),
                })
                return {"mode": "new", "experiment_id": self.experiment_id}

            if not self.metadata_path.exists():
                raise RuntimeError(
                    f"experiment {self.experiment_id!r} does not exist; start it with --new-run"
                )
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if metadata.get("mode") != self.mode:
                raise RuntimeError("checkpoint mode is incompatible with this command")
            if metadata.get("critical_fingerprint") != fingerprint:
                raise RuntimeError(
                    "resume compatibility check failed: seeds, tier/expansion policy, model, "
                    "ontology, prompts, validation, or save settings changed"
                )
            restored = self.restore_latest()
            return {
                "mode": "resume", "experiment_id": self.experiment_id,
                "restored_snapshot": restored,
            }
        except Exception:
            self.release()
            raise

    def _artifact_sources(self) -> dict[str, Path]:
        sources = {
            str(key): resolve_path(value)
            for key, value in (self.config.get("storage") or {}).items()
            if key != "state_db" and value
        }
        audit_path = (self.config.get("audit") or {}).get("path")
        if audit_path:
            sources["__audit__"] = resolve_path(audit_path)
        graph_value = (self.config.get("storage") or {}).get("graph_export")
        if graph_value:
            graph_path = resolve_path(graph_value)
            companion = graph_path.with_name(graph_path.stem + ".audit.json")
            sources["__audit_companion__"] = companion
            sources["__audit_journal__"] = companion.with_suffix(
                companion.suffix + ".journal.jsonl"
            )
        state_path = resolve_path((self.config.get("storage") or {})["state_db"])
        sources["__run_metrics__"] = state_path.with_name("run_metrics.json")
        sources["__run_config__"] = resolve_path(
            (self.config.get("storage") or {}).get(
                "run_config", str(state_path.with_name("run_config.json")),
            )
        )
        return sources

    def _copy_artifacts(
        self, destination: Path, connection: sqlite3.Connection,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        sources = self._artifact_sources()
        row = connection.execute(
            "SELECT value FROM expansion_run_state WHERE key='run_id'"
        ).fetchone()
        if row:
            run_id = json.loads(row[0])
            sources["__retrieval_manifest__"] = (
                REPO_ROOT / "logs" / f"retrieval_manifest_{run_id}.csv"
            )
        for key, source in sources.items():
            if not source.exists():
                continue
            target = destination / "artifacts" / key
            if source.is_dir():
                shutil.copytree(source, target)
                kind = "directory"
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                kind = "file"
            entries.append({"key": key, "kind": kind, "filename": source.name})
        return entries

    def snapshot(self, connection: sqlite3.Connection, *, reason: str) -> str:
        """Create and publish one complete generation using SQLite backup."""
        self.snapshots.mkdir(parents=True, exist_ok=True)
        generation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        building = self.snapshots / (".building-" + generation)
        final = self.snapshots / generation
        building.mkdir(parents=True)
        try:
            state_target = building / "state.sqlite3"
            destination = sqlite3.connect(state_target)
            try:
                connection.backup(destination)
                destination.commit()
            finally:
                destination.close()
            artifacts = self._copy_artifacts(building, connection)
            files = {
                str(path.relative_to(building)).replace("\\", "/"): _sha256(path)
                for path in building.rglob("*") if path.is_file()
            }
            manifest = {
                "schema_version": "1.0",
                "experiment_id": self.experiment_id,
                "generation": generation,
                "created_at": _now(),
                "reason": reason,
                "artifacts": artifacts,
                "files": files,
            }
            atomic_write_json(building / "manifest.json", manifest)
            building.replace(final)
            atomic_write_json(self.latest_path, {
                "experiment_id": self.experiment_id,
                "generation": generation,
                "published_at": _now(),
            })
            self.last_snapshot = generation
            self._documents_since_snapshot = 0
            self._prune()
            return generation
        finally:
            if building.exists():
                shutil.rmtree(building)

    def maybe_snapshot(self, connection: sqlite3.Connection, *, reason: str) -> Optional[str]:
        self._documents_since_snapshot += 1
        if self._documents_since_snapshot < self.every:
            return None
        return self.snapshot(connection, reason=reason)

    def _prune(self) -> None:
        generations = sorted(
            path for path in self.snapshots.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        for old in generations[:-self.keep]:
            shutil.rmtree(old)

    def restore_latest(self) -> str:
        if not self.latest_path.exists():
            raise RuntimeError("experiment exists but has no published checkpoint snapshot")
        latest = json.loads(self.latest_path.read_text(encoding="utf-8"))
        generation = str(latest.get("generation") or "")
        source = self.snapshots / generation
        manifest_path = source / "manifest.json"
        if not source.is_dir() or not manifest_path.is_file():
            raise RuntimeError("LATEST points to an incomplete checkpoint generation")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, expected in (manifest.get("files") or {}).items():
            candidate = source / relative
            if not candidate.is_file() or _sha256(candidate) != expected:
                raise RuntimeError(f"checkpoint integrity failure: {relative}")

        state_target = resolve_path((self.config.get("storage") or {})["state_db"])
        state_target.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(str(state_target) + suffix).unlink(missing_ok=True)
        shutil.copy2(source / "state.sqlite3", state_target)
        current_sources = self._artifact_sources()
        for entry in manifest.get("artifacts") or []:
            key = str(entry["key"])
            target = current_sources.get(key)
            if key == "__retrieval_manifest__":
                target = REPO_ROOT / "logs" / str(entry.get("filename"))
            if target is None:
                continue
            saved = source / "artifacts" / key
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            if entry.get("kind") == "directory":
                shutil.copytree(saved, target)
            else:
                shutil.copy2(saved, target)
        self.restored_snapshot = generation
        return generation

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "experiment_id": self.experiment_id,
            "root": str(self.root),
            "restored_snapshot": self.restored_snapshot,
            "last_snapshot": self.last_snapshot,
        }
