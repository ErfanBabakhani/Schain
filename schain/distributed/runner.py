"""High-level lifecycle used by the CLI and Colab notebook."""

from __future__ import annotations

import os
import socket
import subprocess
from typing import Any
import uuid

from ..config import save_settings, seed_groups
from ..names import canonical_id
from ..ontology import load_ontology
from ..pipeline import open_driver
from ..llm.provider import build_llm
from .config import distributed_settings
from .expand import DistributedExpandProcessor
from .outbox import Neo4jOutboxDispatcher
from ..state_backend import configured_state_backend
from .schema import SCHEMA_VERSION
from .worker import WorkerSession
from .manifest import build_distributed_manifest
from .profile import profile_snapshot, validate_deep_discovery


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=3,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def automatic_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def run_distributed_expand(
    config: dict[str, Any], *, tasks_this_session: int | None = None,
    worker_id: str | None = None,
) -> dict[str, Any]:
    settings = distributed_settings(config, require_credentials=True)
    if settings.backend != "postgres" or not settings.experiment_id or not settings.database_url:
        raise ValueError("distributed Expand requires PostgreSQL state settings")
    save = save_settings(config)
    repository = configured_state_backend(config).open()
    worker_id = worker_id or automatic_worker_id()
    session_id = uuid.uuid4().hex
    driver = None
    registered = False
    try:
        repository.migrate()
        if settings.resume_existing_only and not repository.experiment_exists(
            settings.experiment_id
        ):
            raise RuntimeError(
                f"experiment {settings.experiment_id!r} does not exist; "
                "resume_existing_only forbids creating a replacement"
            )
        seeds = []
        for group, names in seed_groups(config).items():
            seeds.extend({
                "group": group, "entity_name": name,
                "entity_key": canonical_id("Company", name), "entity_type": "Company",
            } for name in names)
        manifest = build_distributed_manifest(config)
        profile = profile_snapshot(config)
        if profile["version"] != "unversioned":
            profile = validate_deep_discovery(config)
        join_status = repository.initialize_or_join_experiment(
            settings.experiment_id, manifest,
            max_tier=int((config.get("expansion") or {}).get("max_tier", 3)),
            save_enabled=save["enabled"], seeds=seeds,
            max_task_attempts=settings.max_task_attempts,
            discovery_profile=profile,
            document_limit=int(profile.get("document_limit") or 0) or None,
        )
        commit = _git_commit()
        repository.register_worker(
            settings.experiment_id, worker_id, session_id,
            git_commit=commit,
            metadata={"schema_version": SCHEMA_VERSION, "mode": "distributed_expand"},
        )
        registered = True
        ontology = load_ontology(config)
        llm = build_llm(config)
        dispatcher = None
        if save["enabled"]:
            driver = open_driver(config)
            dispatcher = Neo4jOutboxDispatcher(
                repository, driver, config, ontology,
                experiment_id=settings.experiment_id, worker_id=worker_id,
            )
        processor = DistributedExpandProcessor(
            repository, config, llm, ontology, worker_id=worker_id,
        )
        session = WorkerSession(
            repository=repository, experiment_id=settings.experiment_id,
            worker_id=worker_id, processor=processor,
            lease_seconds=settings.task_lease_seconds,
            heartbeat_seconds=settings.heartbeat_seconds,
            poll_seconds=settings.poll_seconds,
            save_enabled=save["enabled"], graph_dispatcher=dispatcher,
        )
        task_limit = settings.tasks_per_session if tasks_this_session is None else int(tasks_this_session)
        result = session.run(max_tasks=max(0, task_limit))
        result.update({
            "experiment_join_status": join_status,
            "experiment_id": settings.experiment_id,
            "session_id": session_id,
            "git_commit": commit,
            "database_size_bytes": repository.database_size_bytes(),
            "save_enabled": save["enabled"],
        })
        return result
    finally:
        if registered:
            try:
                repository.stop_worker(settings.experiment_id, worker_id)
            except Exception:  # noqa: BLE001 - do not hide the run's original error
                pass
        if driver is not None:
            driver.close()
        repository.close()
