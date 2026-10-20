"""Real PostgreSQL concurrency and recovery tests.

Set SCHAIN_TEST_POSTGRES_URL to a disposable database.  These tests use independent
pooled connections and PostgreSQL SKIP LOCKED; they are intentionally not emulated by
SQLite.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import time
import uuid

import pytest

from schain.distributed.models import (
    IdentityConflictError,
    StaleLeaseError,
    TaskResult,
)
from schain.distributed.postgres import PostgresRepository
from schain.distributed.outbox import Neo4jOutboxDispatcher


pytestmark = pytest.mark.postgres


@pytest.fixture(scope="module")
def repository():
    url = os.environ.get("SCHAIN_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("SCHAIN_TEST_POSTGRES_URL is not configured")
    repo = PostgresRepository(url, pool_min=1, pool_max=12)
    repo.migrate()
    yield repo
    repo.close()


def _experiment(repository, names=("A",), *, save=False, max_tier=3):
    experiment_id = "pytest-" + uuid.uuid4().hex
    seeds = [
        {"entity_key": f"company:{name.casefold()}", "entity_name": name, "group": "test"}
        for name in names
    ]
    manifest = {"test": experiment_id, "model": "deterministic"}
    assert repository.initialize_or_join_experiment(
        experiment_id, manifest, max_tier=max_tier, save_enabled=save,
        seeds=seeds, max_task_attempts=3,
    ) == "created"
    return experiment_id, manifest, seeds


def _experiment_with_attempts(repository, names, attempts):
    experiment_id = "pytest-" + uuid.uuid4().hex
    seeds = [
        {"entity_key": f"company:{name.casefold()}", "entity_name": name, "group": "test"}
        for name in names
    ]
    repository.initialize_or_join_experiment(
        experiment_id, {"test": experiment_id}, max_tier=3, save_enabled=False,
        seeds=seeds, max_task_attempts=attempts,
    )
    return experiment_id


def _register(repository, experiment_id, worker_id):
    repository.register_worker(experiment_id, worker_id, uuid.uuid4().hex)


def _document(name="doc"):
    return {
        "doc_id": name, "content_hash": "hash-" + name,
        "text": "Supplier A supplies Customer B.", "title": name,
    }


def _fact(doc_id="doc", suffix="1"):
    return {
        "fact_id": f"fact-{suffix}", "doc_id": doc_id,
        "subject_id": "company:supplier", "subject_type": "Company",
        "subject_name": "Supplier", "subject_mention": "Supplier",
        "relation": "suppliesTo", "object_id": "company:customer",
        "object_type": "Company", "object_name": "Customer",
        "object_mention": "Customer", "evidence": "Supplier supplies Customer",
    }


def test_concurrent_claims_are_distinct(repository):
    experiment_id, _, _ = _experiment(repository, ("A", "B", "C", "D"))
    workers = [f"worker-{i}" for i in range(8)]
    for worker in workers:
        _register(repository, experiment_id, worker)
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        leases = list(executor.map(
            lambda worker: repository.claim_task(
                experiment_id, worker, lease_seconds=30,
            ), workers,
        ))
    claimed = [lease for lease in leases if lease is not None]
    assert len(claimed) == 4
    assert len({lease.task_id for lease in claimed}) == 4
    assert len({lease.entity_key for lease in claimed}) == 4


def test_simultaneous_experiment_initialization_is_single_and_compatible(repository):
    experiment_id = "pytest-" + uuid.uuid4().hex
    manifest = {"model": "same", "ontology": "same"}
    seeds = [{"entity_key": "company:seed", "entity_name": "Seed", "group": "test"}]
    def initialize(_):
        return repository.initialize_or_join_experiment(
            experiment_id, manifest, max_tier=2, save_enabled=False,
            seeds=seeds, max_task_attempts=3,
        )
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(initialize, range(2)))
    assert outcomes.count("created") == 1
    for worker in ("init-a", "init-b"):
        _register(repository, experiment_id, worker)
    assert repository.claim_task(experiment_id, "init-a", lease_seconds=30) is not None
    assert repository.claim_task(experiment_id, "init-b", lease_seconds=30) is None
    with pytest.raises(ValueError, match="incompatible"):
        repository.initialize_or_join_experiment(
            experiment_id, {"model": "different"}, max_tier=2, save_enabled=False,
            seeds=seeds, max_task_attempts=3,
        )


def test_concurrent_duplicate_discovery_schedules_once(repository):
    experiment_id, _, _ = _experiment(repository, ("A", "B"))
    for worker in ("w1", "w2"):
        _register(repository, experiment_id, worker)
    first = repository.claim_task(experiment_id, "w1", lease_seconds=30)
    second = repository.claim_task(experiment_id, "w2", lease_seconds=30)
    discovery = {
        "entity_key": "company:shared", "entity_name": "Shared", "tier": 1,
        "entity_type": "Company",
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(
            lambda lease: repository.complete_task(
                lease, TaskResult(discoveries=[discovery]), save_enabled=False,
            ), (first, second),
        ))
    lease = repository.claim_task(experiment_id, "w1", lease_seconds=30)
    assert lease.entity_key == "company:shared"
    assert repository.claim_task(experiment_id, "w2", lease_seconds=30) is None


def test_lease_expiry_reclaim_and_stale_fencing(repository):
    experiment_id, _, _ = _experiment(repository)
    _register(repository, experiment_id, "old")
    _register(repository, experiment_id, "new")
    old = repository.claim_task(experiment_id, "old", lease_seconds=1)
    time.sleep(1.2)
    new = repository.claim_task(experiment_id, "new", lease_seconds=30)
    assert new.task_id == old.task_id
    assert new.lease_token > old.lease_token
    with pytest.raises(StaleLeaseError):
        repository.complete_task(old, TaskResult(), save_enabled=False)
    repository.complete_task(new, TaskResult(), save_enabled=False)


def test_terminal_failure_blocks_later_tiers(repository):
    experiment_id = _experiment_with_attempts(repository, ("A", "B"), attempts=1)
    _register(repository, experiment_id, "w1")
    _register(repository, experiment_id, "w2")
    first = repository.claim_task(experiment_id, "w1", lease_seconds=30)
    second = repository.claim_task(experiment_id, "w2", lease_seconds=1)
    repository.complete_task(
        first,
        TaskResult(discoveries=[{
            "entity_key": "company:tier1", "entity_name": "Tier 1", "tier": 1,
            "max_attempts": 1,
        }]),
        save_enabled=False,
    )
    time.sleep(1.2)
    assert repository.claim_task(experiment_id, "w1", lease_seconds=30) is None
    assert repository.progress(experiment_id)["status"] == "failed"


def test_failed_atomic_completion_leaves_no_partial_state(repository):
    experiment_id, _, _ = _experiment(repository)
    _register(repository, experiment_id, "worker")
    lease = repository.claim_task(experiment_id, "worker", lease_seconds=30)
    document = _document("atomic")
    repository.save_document_checkpoint(lease, document=document, stage="retrieved")
    alias = {
        "mention_key": "company:meridian", "entity_type": "Company",
        "mention": "Meridian", "canonical_id": "company:meridian-energy",
        "canonical_name": "Meridian Energy", "method": "deterministic", "evidence": [],
    }
    repository.complete_task(
        lease, TaskResult(documents=[document], aliases=[alias]), save_enabled=False,
    )

    # A separate experiment demonstrates rollback inside the completion transaction.
    second_id, _, _ = _experiment(repository, ("X", "Y"))
    _register(repository, second_id, "w1")
    _register(repository, second_id, "w2")
    one = repository.claim_task(second_id, "w1", lease_seconds=30)
    two = repository.claim_task(second_id, "w2", lease_seconds=30)
    repository.complete_task(
        one, TaskResult(aliases=[alias]), save_enabled=False,
    )
    conflict = dict(alias, canonical_id="company:unrelated", canonical_name="Unrelated")
    with pytest.raises(IdentityConflictError):
        repository.complete_task(
            two,
            TaskResult(documents=[_document("must-rollback")], aliases=[conflict]),
            save_enabled=False,
        )
    assert all(
        row["document"]["doc_id"] != "must-rollback"
        for row in repository.task_documents(two)
    )


def test_postgres_commit_survives_new_repository_and_repeat_join(repository):
    experiment_id, manifest, seeds = _experiment(repository, save=True)
    _register(repository, experiment_id, "worker")
    lease = repository.claim_task(experiment_id, "worker", lease_seconds=30)
    repository.complete_task(
        lease,
        TaskResult(documents=[_document()], accepted_facts=[_fact()]),
        save_enabled=True,
    )
    assert repository.initialize_or_join_experiment(
        experiment_id, manifest, max_tier=3, save_enabled=True,
        seeds=seeds, max_task_attempts=3,
    ) == "running"
    url = os.environ["SCHAIN_TEST_POSTGRES_URL"]
    resumed = PostgresRepository(url, pool_min=1, pool_max=2)
    try:
        assert resumed.export_facts(experiment_id) == [_fact()]
        assert resumed.progress(experiment_id)["pending_outbox_items"] == 1
    finally:
        resumed.close()


def test_tier_barrier_waits_for_required_outbox(repository):
    experiment_id, _, _ = _experiment(repository, save=True)
    _register(repository, experiment_id, "worker")
    lease = repository.claim_task(experiment_id, "worker", lease_seconds=30)
    repository.complete_task(
        lease,
        TaskResult(
            documents=[_document("barrier")], accepted_facts=[_fact("barrier", "barrier")],
            discoveries=[{"entity_key": "company:tier1", "entity_name": "Tier 1", "tier": 1}],
        ),
        save_enabled=True,
    )
    assert repository.claim_task(experiment_id, "worker", lease_seconds=30) is None
    outbox = repository.claim_outbox(experiment_id, "worker", lease_seconds=30)
    repository.acknowledge_outbox(outbox, nodes=2, relationships=1)
    assert repository.claim_task(experiment_id, "worker", lease_seconds=30).tier == 1


def test_save_disabled_completes_without_outbox(repository):
    experiment_id, _, _ = _experiment(repository, save=False)
    _register(repository, experiment_id, "worker")
    lease = repository.claim_task(experiment_id, "worker", lease_seconds=30)
    repository.complete_task(
        lease,
        TaskResult(documents=[_document("off")], accepted_facts=[_fact("off", "off")]),
        save_enabled=False,
    )
    progress = repository.progress(experiment_id)
    assert progress["pending_outbox_items"] == 0
    assert progress["status"] == "completed"


def test_extraction_checkpoint_survives_worker_disconnect(repository):
    experiment_id, _, _ = _experiment(repository)
    _register(repository, experiment_id, "first")
    lease = repository.claim_task(experiment_id, "first", lease_seconds=30)
    extraction = {"graph": {"nodes": [], "relationships": []}, "chunk_index": {}}
    repository.save_document_checkpoint(
        lease, document=_document("checkpoint"), stage="extracted", extraction=extraction,
    )
    rows = repository.task_documents(lease)
    assert rows[0]["stage"] == "extracted"
    assert rows[0]["extraction"] == extraction


def test_crash_after_neo4j_write_replays_idempotent_merge(repository):
    experiment_id, _, _ = _experiment(repository, save=True)
    _register(repository, experiment_id, "worker")
    lease = repository.claim_task(experiment_id, "worker", lease_seconds=30)
    repository.complete_task(
        lease,
        TaskResult(
            documents=[_document("replay")], accepted_facts=[_fact("replay", "replay")],
        ),
        save_enabled=True,
    )

    class FakeDriver:
        def __init__(self):
            self.queries = []

        def execute_query(self, query, **kwargs):
            self.queries.append(query)

    class CrashBeforeAck:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def acknowledge_outbox(self, *args, **kwargs):
            raise SystemExit("simulated process death after Neo4j commit")

    driver = FakeDriver()
    config = {"neo4j": {"database": "neo4j", "batch_size": 100}}
    ontology = {"entity_types": ["Company"], "relation_types": ["suppliesTo"]}
    first = Neo4jOutboxDispatcher(
        CrashBeforeAck(repository), driver, config, ontology,
        experiment_id=experiment_id, worker_id="graph-1", lease_seconds=1,
    )
    with pytest.raises(SystemExit):
        first.drain(max_items=1)
    assert repository.progress(experiment_id)["pending_outbox_items"] == 1
    time.sleep(1.2)
    second = Neo4jOutboxDispatcher(
        repository, driver, config, ontology,
        experiment_id=experiment_id, worker_id="graph-2", lease_seconds=30,
    )
    assert second.drain(max_items=1) == 1
    assert repository.progress(experiment_id)["pending_outbox_items"] == 0
    assert len(driver.queries) == 4
    assert all("MERGE" in query for query in driver.queries)
