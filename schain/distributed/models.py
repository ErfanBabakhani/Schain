"""Small transport types shared by the repository and worker loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class StaleLeaseError(RuntimeError):
    """The task/outbox lease was replaced and this worker is fenced out."""


class IdentityConflictError(RuntimeError):
    """An alias already maps to an incompatible canonical entity."""


@dataclass(frozen=True)
class TaskLease:
    task_id: str
    experiment_id: str
    task_key: str
    worker_id: str
    lease_token: int
    tier: int
    entity_key: str
    entity_name: str
    payload: dict[str, Any] = field(default_factory=dict)
    attempt: int = 1


@dataclass(frozen=True)
class OutboxLease:
    outbox_id: str
    experiment_id: str
    worker_id: str
    lease_token: int
    document_id: str
    tier: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    documents: list[dict[str, Any]] = field(default_factory=list)
    accepted_facts: list[dict[str, Any]] = field(default_factory=list)
    rejected_facts: list[dict[str, Any]] = field(default_factory=list)
    aliases: list[dict[str, Any]] = field(default_factory=list)
    discoveries: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
