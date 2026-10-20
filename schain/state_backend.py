"""Explicit selector for local SQLite versus distributed PostgreSQL state.

The two modes intentionally expose different repositories: local code keeps its mature
SQLite API, while distributed orchestration uses a narrow transactional repository. This
selector prevents backend conditionals from leaking into extraction and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from . import state
from .config import storage_path
from .distributed.config import distributed_settings


class StateBackend(Protocol):
    kind: str

    def open(self) -> Any: ...


@dataclass
class SQLiteStateBackend:
    config: dict[str, Any]
    kind: str = "sqlite"

    def open(self) -> Any:
        return state.connect(storage_path(self.config, "state_db"))


@dataclass
class PostgresStateBackend:
    config: dict[str, Any]
    kind: str = "postgres"

    def open(self) -> Any:
        from .distributed.postgres import PostgresRepository

        settings = distributed_settings(self.config, require_credentials=True)
        return PostgresRepository(
            settings.database_url,
            pool_min=settings.pool_min,
            pool_max=settings.pool_max,
        )


def configured_state_backend(config: dict[str, Any]) -> StateBackend:
    settings = distributed_settings(config)
    if settings.backend == "sqlite":
        return SQLiteStateBackend(config)
    return PostgresStateBackend(config)
