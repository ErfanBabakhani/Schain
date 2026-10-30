"""Bounded distributed worker loop with independent lease renewal."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Callable, Protocol

from .models import StaleLeaseError, TaskLease, TaskResult

logger = logging.getLogger(__name__)


class TaskProcessor(Protocol):
    def __call__(self, lease: TaskLease) -> TaskResult: ...


class LeaseRenewer:
    """Renew with independent short transactions while inference is in flight."""

    def __init__(self, repository: Any, lease: TaskLease, *, lease_seconds: int, heartbeat_seconds: int):
        self.repository = repository
        self.lease = lease
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"schain-lease-{lease.task_id[:8]}",
            daemon=True,
        )

    def __enter__(self) -> "LeaseRenewer":
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.heartbeat_seconds + 1.0))

    def _run(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            try:
                if not self.repository.renew_task(
                    self.lease, lease_seconds=self.lease_seconds,
                ):
                    self.lost.set()
                    return
                self.repository.heartbeat_worker(
                    self.lease.experiment_id, self.lease.worker_id,
                    {"active_task": self.lease.task_id},
                )
            except Exception:  # noqa: BLE001 - loss is resolved by fencing at commit
                logger.exception("task lease renewal failed")
                self.lost.set()
                return


@dataclass
class WorkerSession:
    repository: Any
    experiment_id: str
    worker_id: str
    processor: TaskProcessor
    lease_seconds: int = 600
    heartbeat_seconds: int = 30
    poll_seconds: float = 3.0
    save_enabled: bool = False
    graph_dispatcher: Any | None = None

    def run(self, *, max_tasks: int = 0, idle_rounds: int = 2) -> dict[str, Any]:
        """Process up to ``max_tasks``; zero means until the shared queue is idle."""
        completed = failed = stale = graph_writes = llm_calls = 0
        idle = 0
        while max_tasks <= 0 or completed + failed < max_tasks:
            self.repository.heartbeat_worker(
                self.experiment_id,
                self.worker_id,
                {
                    "tasks_completed": completed, "tasks_failed": failed,
                    "stale": stale, "llm_calls": llm_calls,
                },
            )
            if self.save_enabled and self.graph_dispatcher is not None:
                graph_writes += self.graph_dispatcher.drain(max_items=1)
            lease = self.repository.claim_task(
                self.experiment_id, self.worker_id, lease_seconds=self.lease_seconds,
            )
            if lease is None:
                idle += 1
                progress = self.repository.progress(self.experiment_id)
                if progress["status"] in {"completed", "failed"} or idle >= idle_rounds:
                    break
                time.sleep(self.poll_seconds)
                continue
            idle = 0
            try:
                with LeaseRenewer(
                    self.repository, lease,
                    lease_seconds=self.lease_seconds,
                    heartbeat_seconds=self.heartbeat_seconds,
                ) as renewer:
                    result = self.processor(lease)
                    if renewer.lost.is_set():
                        raise StaleLeaseError("lease renewal failed while task was running")
                    self.repository.complete_task(
                        lease, result, save_enabled=self.save_enabled,
                    )
                    usage = result.metrics.get("usage") or {}
                    llm_calls += int(usage.get("llm_calls", result.metrics.get("llm_calls", 0)) or 0)
                completed += 1
            except StaleLeaseError:
                stale += 1
                self.repository.record_stale_completion(lease)
                logger.warning("stale completion fenced task=%s", lease.task_id)
            except Exception as error:  # noqa: BLE001 - retry policy is repository-owned
                failed += 1
                try:
                    self.repository.fail_task(lease, error)
                except StaleLeaseError:
                    stale += 1
                    self.repository.record_stale_completion(lease)
                logger.exception("distributed task failed task=%s", lease.task_id)
        if self.save_enabled and self.graph_dispatcher is not None:
            graph_writes += self.graph_dispatcher.drain(max_items=0)
        progress = self.repository.progress(self.experiment_id)
        return {
            "worker_id": self.worker_id,
            "tasks_completed_this_session": completed,
            "tasks_failed_this_session": failed,
            "stale_completions_rejected": stale,
            "graph_writes_this_session": graph_writes,
            "llm_calls_this_session": llm_calls,
            "experiment": progress,
        }
