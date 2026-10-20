"""Validated settings for the optional PostgreSQL state backend."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass(frozen=True)
class DistributedSettings:
    backend: str
    enabled: bool
    experiment_id: str | None
    database_url_env: str
    database_url: str | None
    worker_concurrency: int
    task_lease_seconds: int
    heartbeat_seconds: int
    max_task_attempts: int
    poll_seconds: float
    pool_min: int
    pool_max: int
    event_retention_days: int


def distributed_settings(
    config: dict[str, Any], *, require_credentials: bool = False,
) -> DistributedSettings:
    state = config.get("state") or {}
    distributed = config.get("distributed") or {}
    backend = str(state.get("backend", "sqlite")).casefold()
    enabled = bool(distributed.get("enabled", False))
    if backend not in {"sqlite", "postgres"}:
        raise ValueError("state.backend must be 'sqlite' or 'postgres'")
    if enabled and backend != "postgres":
        raise ValueError("distributed.enabled requires state.backend: postgres")
    if backend == "postgres" and not enabled:
        raise ValueError("state.backend: postgres requires distributed.enabled: true")
    experiment_id = state.get("experiment_id")
    if backend == "postgres" and not experiment_id:
        raise ValueError("state.experiment_id is required for PostgreSQL state")
    env_name = str(state.get("database_url_env", "NEON_DATABASE_URL"))
    database_url = os.environ.get(env_name)
    if require_credentials and backend == "postgres" and not database_url:
        raise ValueError(f"{env_name} is required for PostgreSQL state")
    lease = int(distributed.get("task_lease_seconds", 600))
    heartbeat = int(distributed.get("heartbeat_seconds", 30))
    if lease < 10:
        raise ValueError("distributed.task_lease_seconds must be at least 10")
    if heartbeat < 1 or heartbeat * 2 >= lease:
        raise ValueError("distributed.heartbeat_seconds must be positive and less than half the lease")
    pool_min = int(distributed.get("connection_pool_min", 1))
    pool_max = int(distributed.get("connection_pool_max", 4))
    concurrency = int(distributed.get("worker_concurrency", 1))
    if concurrency < 1:
        raise ValueError("distributed.worker_concurrency must be at least 1")
    if concurrency != 1:
        raise ValueError(
            "distributed.worker_concurrency currently supports 1 per process; "
            "scale by starting additional workers/Colab runtimes"
        )
    if pool_min < 1 or pool_max < pool_min:
        raise ValueError("invalid distributed PostgreSQL connection pool bounds")
    return DistributedSettings(
        backend=backend,
        enabled=enabled,
        experiment_id=str(experiment_id) if experiment_id else None,
        database_url_env=env_name,
        database_url=database_url,
        worker_concurrency=concurrency,
        task_lease_seconds=lease,
        heartbeat_seconds=heartbeat,
        max_task_attempts=max(1, int(distributed.get("max_task_attempts", 3))),
        poll_seconds=max(0.25, float(distributed.get("poll_seconds", 3.0))),
        pool_min=pool_min,
        pool_max=pool_max,
        event_retention_days=max(1, int(distributed.get("event_retention_days", 30))),
    )
