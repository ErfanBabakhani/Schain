"""Credential-free checks for distributed settings and worker behavior."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from schain.distributed.config import distributed_settings
from schain.distributed.models import TaskLease, TaskResult
from schain.distributed.worker import WorkerSession
from schain.state_backend import configured_state_backend
from schain.config import load_config
from schain.durable import critical_configuration


def test_sqlite_remains_default_and_distributed_requires_postgres():
    settings = distributed_settings({})
    assert settings.backend == "sqlite"
    assert settings.enabled is False
    assert configured_state_backend({"storage": {"state_db": "data/state/state.db"}}).kind == "sqlite"
    with pytest.raises(ValueError, match="requires state.backend"):
        distributed_settings({"distributed": {"enabled": True}})


def test_distributed_example_config_loads():
    config = load_config("config/distributed_neon_expand.yaml")
    settings = distributed_settings(config)
    assert settings.backend == "postgres"
    assert settings.experiment_id == "bmw-supply-chain-v1"


def test_postgres_requires_experiment_and_credential(monkeypatch):
    with pytest.raises(ValueError, match="experiment_id"):
        distributed_settings({
            "state": {"backend": "postgres"},
            "distributed": {"enabled": True},
        })
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="NEON_DATABASE_URL"):
        distributed_settings({
            "state": {"backend": "postgres", "experiment_id": "test"},
            "distributed": {"enabled": True},
        }, require_credentials=True)


def test_invalid_lease_heartbeat_is_rejected():
    with pytest.raises(ValueError, match="less than half"):
        distributed_settings({
            "state": {"backend": "postgres", "experiment_id": "test"},
            "distributed": {
                "enabled": True, "task_lease_seconds": 20, "heartbeat_seconds": 10,
            },
        })


def test_save_disabled_worker_never_uses_graph_dispatcher():
    lease = TaskLease(
        task_id="task", experiment_id="exp", task_key="expand:company:a",
        worker_id="worker", lease_token=1, tier=0,
        entity_key="company:a", entity_name="A",
    )
    repository = Mock()
    repository.claim_task.side_effect = [lease, None]
    repository.progress.return_value = {"status": "completed"}
    processor = Mock(return_value=TaskResult())
    graph = Mock()
    result = WorkerSession(
        repository, "exp", "worker", processor,
        lease_seconds=100, heartbeat_seconds=1, poll_seconds=0,
        save_enabled=False, graph_dispatcher=graph,
    ).run()
    graph.drain.assert_not_called()
    repository.complete_task.assert_called_once_with(
        lease, processor.return_value, save_enabled=False,
    )
    assert result["tasks_completed_this_session"] == 1


def test_worker_generates_no_secret_output(monkeypatch):
    secret = "postgresql://secret-user:secret-password@example.invalid/db"
    monkeypatch.setenv("NEON_DATABASE_URL", secret)
    settings = distributed_settings({
        "state": {"backend": "postgres", "experiment_id": "safe"},
        "distributed": {"enabled": True},
    }, require_credentials=True)
    assert settings.database_url == secret
    assert secret not in repr({
        "backend": settings.backend,
        "experiment_id": settings.experiment_id,
    })
    manifest = critical_configuration({
        "state": {"backend": "postgres", "database_url": secret},
        "distributed": {"enabled": True},
    })
    assert secret not in repr(manifest)
