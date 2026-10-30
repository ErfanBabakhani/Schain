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

from schain.config import load_config
from schain.distributed.maintenance import apply_extension, extension_plan
from schain.distributed.manifest import build_distributed_manifest
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


def test_safe_extension_preserves_history_and_schedules_ordered_idempotent_topups(repository):
    config = load_config("config/distributed_neon_deep_discovery.yaml")
    experiment_id = "pytest-" + uuid.uuid4().hex
    current_manifest = build_distributed_manifest(config)
    current_manifest["expansion"] = {**current_manifest["expansion"], "max_tier": 2}
    repository.initialize_or_join_experiment(
        experiment_id, current_manifest, max_tier=2, save_enabled=False,
        seeds=[{"entity_key": "company:toyota", "entity_name": "Toyota", "group": "test"}],
        max_task_attempts=3,
    )
    with repository._connection() as connection, connection.transaction():
        # Model the completed Tier-0/Tier-1 history and four terminal Tier-2 nodes.
        for index in range(11):
            repository._schedule_locked(
                connection, experiment_id, entity_key=f"company:tier1-{index}",
                entity_name=f"Tier One {index}", entity_type="Company", tier=1,
                discovered_from="company:toyota", provenance={"historical": True},
                priority=1, max_task_attempts=3, max_tier=2,
            )
        for index in range(4):
            repository._schedule_locked(
                connection, experiment_id, entity_key=f"company:tier2-{index}",
                entity_name=f"Tier Two {index}", entity_type="Company", tier=2,
                discovered_from="company:tier1-0", provenance={"historical": True},
                priority=1, max_task_attempts=3, max_tier=2,
            )
        connection.execute(
            "UPDATE schain_tasks SET status='completed',completed_at=clock_timestamp() "
            "WHERE experiment_id=%s", (experiment_id,),
        )
        connection.execute(
            "UPDATE schain_frontier SET status=CASE WHEN tier=2 THEN 'terminal' ELSE 'completed' END "
            "WHERE experiment_id=%s", (experiment_id,),
        )
        original_task_row = connection.execute(
            "SELECT task_id,entity_key,entity_name FROM schain_tasks "
            "WHERE experiment_id=%s AND tier=1 ORDER BY task_id LIMIT 1",
            (experiment_id,),
        ).fetchone()
        original_task = original_task_row["task_id"]
        connection.execute(
            "INSERT INTO schain_documents "
            "(experiment_id,document_id,content_hash,task_id,origin_entity,tier,stage,document) "
            "VALUES (%s,'historical-doc','historical-hash',%s,%s,1,'completed',%s)",
            (experiment_id, original_task, original_task_row["entity_key"],
             repository._json(_document("historical-doc"))),
        )
        connection.execute(
            "INSERT INTO schain_facts "
            "(experiment_id,fact_id,document_id,task_id,tier,accepted,payload) "
            "VALUES (%s,'historical-fact','historical-doc',%s,1,true,%s)",
            (experiment_id, original_task, repository._json({
                "fact_id": "historical-fact", "doc_id": "historical-doc",
                "subject_id": "company:tier2-0", "subject_name": "Tier Two 0",
                "subject_type": "Company", "relation": "suppliesTo",
                "object_id": original_task_row["entity_key"],
                "object_name": original_task_row["entity_name"],
                "object_type": "Company", "evidence": "Tier Two 0 supplies Toyota",
            })),
        )
    plan = extension_plan(
        repository, experiment_id, new_max_tier=5, topup_tier=1, config=config,
    )
    assert not plan["blockers"]
    assert plan["toyota_recovery"]["entity_key"] == "company:toyota"
    assert len(plan["tier_topups"]) == 11
    assert len(plan["terminal_nodes_to_reopen"]) == 4
    result = apply_extension(
        repository, experiment_id, new_max_tier=5, topup_tier=1, config=config,
    )
    assert result["scheduled"] == {"toyota": 1, "tier_topups": 11, "reopened": 4}
    assert result["historical_path_audit_rows_added"] == 1
    replay = apply_extension(
        repository, experiment_id, new_max_tier=5, topup_tier=1, config=config,
    )
    assert replay["idempotent_replay"] is True
    with repository._connection() as connection:
        assert connection.execute(
            "SELECT count(*) n FROM schain_documents WHERE experiment_id=%s AND document_id='historical-doc'",
            (experiment_id,),
        ).fetchone()["n"] == 1
        assert connection.execute(
            "SELECT count(*) n FROM schain_tasks WHERE experiment_id=%s AND status='completed'",
            (experiment_id,),
        ).fetchone()["n"] == 12
        assert connection.execute(
            "SELECT jsonb_array_length(manifest_history) n FROM schain_experiments WHERE experiment_id=%s",
            (experiment_id,),
        ).fetchone()["n"] == 1
    _register(repository, experiment_id, "resume-worker")
    assert repository.claim_task(experiment_id, "resume-worker", lease_seconds=30).tier == 0


def test_multi_path_relationships_do_not_duplicate_or_raise_global_tier(repository):
    experiment_id, _, _ = _experiment(repository, ("A", "B"), max_tier=3)
    for worker in ("path-a", "path-b"):
        _register(repository, experiment_id, worker)
    leases = [
        repository.claim_task(experiment_id, "path-a", lease_seconds=30),
        repository.claim_task(experiment_id, "path-b", lease_seconds=30),
    ]
    for index, lease in enumerate(leases):
        doc_id = f"path-{index}"
        customer = lease.entity_key
        fact = {
            **_fact(doc_id, str(index)), "object_id": customer,
            "object_name": lease.entity_name,
        }
        repository.complete_task(
            lease,
            TaskResult(
                documents=[_document(doc_id)], accepted_facts=[fact],
                discoveries=[{"entity_key": "company:supplier", "entity_name": "Supplier", "tier": 1}],
            ),
            save_enabled=False,
        )
    with repository._connection() as connection:
        supplier = connection.execute(
            "SELECT tier,count(*) OVER () total FROM schain_frontier "
            "WHERE experiment_id=%s AND entity_key='company:supplier'",
            (experiment_id,),
        ).fetchone()
        assert supplier["tier"] == 1 and supplier["total"] == 1
    paths = repository.frontier_audit(experiment_id)
    assert len(paths) == 2
    assert {row["customer_id"] for row in paths} == {"company:a", "company:b"}
    assert {row["path_depth"] for row in paths} == {1}


def test_toyota_and_toyota_motor_are_not_merged_without_evidenced_alias(repository):
    config = load_config("config/distributed_neon_deep_discovery.yaml")
    experiment_id = "pytest-" + uuid.uuid4().hex
    manifest = build_distributed_manifest(config)
    manifest["expansion"] = {**manifest["expansion"], "max_tier": 2}
    repository.initialize_or_join_experiment(
        experiment_id, manifest, max_tier=2, save_enabled=False,
        seeds=[{
            "entity_key": "company:toyota-motor", "entity_name": "Toyota Motor",
            "group": "test",
        }], max_task_attempts=3,
    )
    plan = extension_plan(
        repository, experiment_id, new_max_tier=5, topup_tier=1, config=config,
    )
    assert plan["toyota_recovery"] is None
    assert "toyota_tier0_not_found_or_not_evidenced" in plan["blockers"]
