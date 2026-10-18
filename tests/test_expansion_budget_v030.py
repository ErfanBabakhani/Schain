"""Offline regression coverage for persistent scheduling, typed frontier and boundaries."""
from pathlib import Path

import pandas as pd
import pytest

from schain import state
from schain.expansion.budget import entity_document_limits
from schain.expansion.frontier import discover_upstream, seed_frontier
from schain.expansion.loop import run_expand
from schain.kg.support import aggregate_edge_support
from schain.sources.base import SearchResult
from .conftest import DOCUMENT_TEXT, StubLLM
from .test_expand import facts_frame


def configure(config, seeds=("BMW", "Toyota"), cap=20, target=2, hard=3):
    config["seeds"] = {"oems": list(seeds)}
    config["expansion"]["max_tier"] = 1
    config["expansion"]["max_documents"] = cap
    config["expansion"]["max_documents_per_run"] = cap
    config["expansion"]["retrieval"].update(
        target_documents_by_tier={0: target}, max_documents_by_tier={0: hard},
        max_documents_per_entity=hard,
        # Existing fairness tests assume soft-target stop without rescue.
        zero_yield_rescue=False,
    )
    return config


def mocked_candidates(monkeypatch, count=7):
    calls = []
    def retrieve(config, name, key, tier, connection, manifest=None, **_kwargs):
        calls.append((name, tier))
        return [SearchResult(provider="test", query="supplies", title=f"CATL supplies {name}",
            url=f"https://example.test/{name}/{i}", text=DOCUMENT_TEXT + f"\nRelease {name} #{i}.",
            entity_key=key, tier=tier) for i in range(count)]
    monkeypatch.setattr("schain.expansion.loop.retrieve_for_entity", retrieve)
    return calls


def test_tier_targets_and_scalar_max_are_real_ceilings(expand_config):
    assert [entity_document_limits(expand_config["expansion"], tier) for tier in range(6)] == [
        (24, 36), (14, 24), (10, 18), (8, 14), (6, 10), (0, 0)]
    expand_config["expansion"]["retrieval"]["max_documents_per_entity"] = 4
    assert entity_document_limits(expand_config["expansion"], 0) == (4, 4)


def test_per_entity_budget_survives_fairness_and_resume(expand_config, connection, ontology, monkeypatch):
    configure(expand_config)
    calls = mocked_candidates(monkeypatch)
    first = StubLLM()
    summary = run_expand(expand_config, connection, first, ontology)
    assert summary["documents_processed"] <= 6
    assert state.document_usage_count(connection, "company:bmw") == 3
    assert state.document_usage_count(connection, "company:toyota") >= 2
    before = len(calls)
    again = StubLLM()
    second = run_expand(expand_config, connection, again, ontology)
    assert len(calls) == before
    assert second["documents_processed"] == 0
    assert again.extraction_calls == 0
    assert second["documents_selected_cumulative"] >= 5


def test_global_reservations_persist_and_last_doc_discovery_runs(expand_config, connection, ontology, monkeypatch):
    configure(expand_config, seeds=("BMW",), cap=1)
    expand_config["expansion"]["max_documents_per_run"] = 1
    calls = mocked_candidates(monkeypatch)
    first = run_expand(expand_config, connection, StubLLM(), ontology)
    assert first["documents_processed"] == 1
    assert first["discovered"]["BMW"] == ["Contemporary Amperex Technology Co. Limited"]
    assert state.frontier_tier_counts(connection) == {0: 1, 1: 1}
    assert first["completion"]["complete"] is False
    before = len(calls)
    # Fresh per-run allowance: resume may continue pending frontier work.
    second = run_expand(expand_config, connection, StubLLM(), ontology)
    assert second["stop_reason"] in ("document_budget", "max_documents", "frontier_empty", "entity_document_cap")
    assert second["lifetime_documents_processed"] >= first["lifetime_documents_processed"]
    assert len(calls) >= before


def test_failed_documents_use_budget_and_retries_are_bounded(expand_config, connection, ontology, monkeypatch):
    configure(expand_config, seeds=("BMW",), cap=1)
    mocked_candidates(monkeypatch, count=1)
    from .conftest import FailingLLM
    expand_config["runtime"]["max_consecutive_failures"] = 5
    summary = run_expand(expand_config, connection, FailingLLM(), ontology)
    rows = connection.execute("SELECT * FROM expansion_document_usage").fetchall()
    assert len(rows) == 1
    assert rows[0]["attempts"] == 2
    assert summary["documents_failed"] == 2
    assert state.document_usage_count(connection) == 1


def test_runtime_is_cumulative_and_uses_observed_document_duration(expand_config, connection, ontology, monkeypatch):
    configure(expand_config, seeds=("BMW",))
    mocked_candidates(monkeypatch)
    from schain.expansion import loop
    clock = [100.0]
    monkeypatch.setattr(loop.time, "monotonic", lambda: clock[0])
    real_process = loop.process_document
    def process(*args, **kwargs):
        outcome = real_process(*args, **kwargs)
        clock[0] += 10.0
        return outcome
    monkeypatch.setattr(loop, "process_document", process)
    expand_config["runtime"]["max_minutes"] = 0.25
    first = run_expand(expand_config, connection, StubLLM(), ontology)
    assert first["documents_processed"] == 1
    assert first["stop_reason"] == "runtime_reserve"
    assert state.run_value(connection, "elapsed_seconds") == 10
    second = run_expand(expand_config, connection, StubLLM(), ontology)
    assert second["documents_processed"] == 0
    assert second["stop_reason"] == "runtime_reserve"


def test_terminal_tier_five_is_never_claimed(connection):
    state.enqueue_entity(connection, "Tier Four", 4)
    state.enqueue_entity(connection, "Tier Five", 5)
    assert state.next_frontier_entity(connection, 5)["entity_name"] == "Tier Four"
    assert state.next_frontier_entity(connection, 5) is None
    assert state.pending_frontier_count(connection, 5) == 0


def test_mine_keeps_type_id_and_is_terminal_by_default(expand_config, connection, ontology):
    state.enqueue_entity(connection, "BMW", 0)
    facts = facts_frame(subject_type="Mine")
    facts["subject_id"], facts["subject_name"] = "mine:greenbushes", "Greenbushes"
    path = Path(expand_config["storage"]["edge_support"])
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_edge_support(facts, config=expand_config).to_parquet(path, index=False)
    assert discover_upstream(connection, expand_config, ontology, "BMW", "company:bmw", 0, facts) == ["Greenbushes"]
    mine = connection.execute("SELECT * FROM frontier WHERE entity_key='mine:greenbushes'").fetchone()
    assert mine["entity_type"] == "Mine"
    assert mine["tier"] == 1
    assert mine["status"] == state.DONE
    assert mine["expandable"] == 0
    assert mine["stop_reason"] == "non_expandable_type"
    assert not connection.execute("SELECT 1 FROM frontier WHERE entity_key='company:greenbushes'").fetchone()


def test_frontier_fallback_rejects_all_false_relevance(expand_config, connection, ontology):
    state.enqueue_entity(connection, "BMW", 0)
    facts = facts_frame()
    facts["expansion_relevant"] = False
    assert discover_upstream(connection, expand_config, ontology, "BMW", "company:bmw", 0, facts) == []
    assert state.frontier_tier_counts(connection) == {0: 1}


def test_missing_or_wrong_parent_tier_cannot_promote(expand_config, connection, ontology):
    assert discover_upstream(connection, expand_config, ontology, "BMW", "company:bmw", 0, facts_frame()) == []
    state.enqueue_entity(connection, "BMW", 0)
    assert discover_upstream(connection, expand_config, ontology, "BMW", "company:bmw", 2, facts_frame()) == []


def test_quality_beam_is_deterministic_preserves_seeds_and_admissions(connection):
    for name in ("Toyota", "Tesla", "BMW", "BYD"):
        state.enqueue_entity(connection, name, 0)
    state.enqueue_entity(connection, "Weak", 1, priority=100)
    state.enqueue_entity(connection, "Strong", 1, priority=110)
    caps = {0: 1, 1: 1}
    run_claims: dict = {}
    roots = [
        state.next_frontier_entity(
            connection, 5, max_expandable_entities_by_tier=caps, run_new_claims_by_tier=run_claims,
        )
        for _ in range(4)
    ]
    assert {r["tier"] for r in roots} == {0}
    for root in roots:
        if int(root["attempts"]) == 1:
            run_claims[0] = run_claims.get(0, 0) + 1
    winner = state.next_frontier_entity(
        connection, 5, max_expandable_entities_by_tier=caps, run_new_claims_by_tier=run_claims,
    )
    assert winner["entity_name"] == "Strong"
    run_claims[1] = run_claims.get(1, 0) + 1
    state.set_frontier_status(connection, winner["entity_key"], state.PENDING)
    # Re-claim Strong (already attempted) is allowed; Weak remains eligible afterward.
    again = state.next_frontier_entity(
        connection, 5, max_expandable_entities_by_tier=caps, run_new_claims_by_tier=run_claims,
    )
    assert again["entity_name"] == "Strong"
    state.set_frontier_status(connection, again["entity_key"], state.DONE)
    weak = state.next_frontier_entity(
        connection, 5, max_expandable_entities_by_tier=caps, run_new_claims_by_tier=run_claims,
    )
    assert weak is not None and weak["entity_name"] == "Weak"


def test_usage_and_multiple_document_origins_survive_reopen(tmp_path):
    path = tmp_path / "state.db"
    conn = state.connect(path)
    record = {"doc_id": "doc", "content_hash": "hash", "source_url": "https://example.test/a"}
    state.register_document(conn, record, "company:bmw", 0)
    state.register_document(conn, record, "company:toyota", 0)
    state.reserve_document(conn, "doc", "company:bmw")
    state.finish_document_usage(conn, "doc", "company:bmw", completed=True, elapsed_seconds=12)
    conn.close()
    conn = state.connect(path)
    assert state.document_origins(conn) == {"doc": {"company:bmw", "company:toyota"}}
    assert state.document_ids_for_entity(conn, "company:toyota") == ["doc"]
    assert state.document_usage_count(conn) == 1
    assert state.document_usage_count(conn, "company:bmw") == 1
    assert state.document_usage(conn, "doc", "company:bmw")["elapsed_seconds"] == 12
    conn.close()


def test_one_source_unknown_reputation_still_promotes(expand_config, connection, ontology):
    state.enqueue_entity(connection, "BMW", 0)
    facts = facts_frame()
    facts["source_effective_score"] = 0
    facts["source_known"] = False
    found = discover_upstream(connection, expand_config, ontology, "BMW", "company:bmw", 0, facts)
    assert found == ["Contemporary Amperex Technology Co. Limited"]


def test_all_configured_seeds_tier_zero(expand_config, connection, ontology):
    assert seed_frontier(connection, expand_config, ontology) == 4
    assert state.frontier_tier_counts(connection) == {0: 4}


def test_completed_document_reconciles_crashed_usage_ledger(expand_config, connection, ontology, monkeypatch):
    configure(expand_config, seeds=("BMW",), cap=1)
    mocked_candidates(monkeypatch, count=1)
    first = run_expand(expand_config, connection, StubLLM(), ontology)
    connection.execute("UPDATE expansion_document_usage SET completed=0")
    connection.commit()
    resumed = run_expand(expand_config, connection, StubLLM(), ontology)
    assert resumed["documents_processed"] == 0
    assert resumed["documents_processed_cumulative"] == 1
    assert not state.incomplete_usage_entities(connection, 2)


def test_failed_shared_article_has_global_retry_ceiling(expand_config, connection, ontology, monkeypatch):
    configure(expand_config)
    calls = []
    def retrieve(config, name, key, tier, connection, manifest=None, **_kwargs):
        return [SearchResult(provider="test", query="q", title="CATL supplies BMW and Toyota",
                text="CATL supplies battery cells to BMW and Toyota.", url="https://example.test/shared",
                entity_key=key, tier=tier)]
    monkeypatch.setattr("schain.expansion.loop.retrieve_for_entity", retrieve)
    from .conftest import FailingLLM
    summary = run_expand(expand_config, connection, FailingLLM(), ontology)
    doc_id = connection.execute("SELECT doc_id FROM documents").fetchone()[0]
    assert state.document_attempt_count(connection, doc_id) == 2
    assert summary["documents_failed"] == 2
    assert state.document_usage_count(connection) == 1


def test_high_yield_offline_scheduler_reaches_tier_four_within_budget(expand_config, connection, ontology, monkeypatch):
    """Adversarial abundant candidates: shallow maxima must not starve deep targets."""
    from schain.expansion import loop
    from schain.kg.facts import upsert_facts_parquet
    from schain.kg.support import refresh_edge_support
    from schain.names import canonical_id
    called_tiers = []
    def retrieve(config, name, key, tier, connection, manifest=None, **_kwargs):
        called_tiers.append(tier)
        return [SearchResult(provider="test", query="q", title=f"Supplier {tier} {key} {i} supplies {name}",
                    text=f"Supplier {tier} {key} {i} supplies components to {name}.",
                    url=f"https://example.test/{key}/{tier}/{i}", entity_key=key, tier=tier,
                    discovery_direction="UPSTREAM_EXPLICIT") for i in range(13)]
    def process(document, config, conn, llm, ontology, driver=None, *, origin_entity=None, tier=None, **kwargs):
        doc_id, _ = state.register_document(conn, document, origin_entity, tier)
        state.set_document_status(conn, doc_id, state.RESOLVED)
        supplier = "Supplier " + doc_id
        fact = facts_frame()
        fact["fact_id"], fact["doc_id"] = doc_id, doc_id
        fact["subject_id"], fact["subject_name"] = canonical_id("Company", supplier), supplier
        fact["object_id"], fact["origin_entity"] = origin_entity, origin_entity
        fact["object_name"] = origin_entity
        path = Path(config["storage"]["extracted_facts"])
        path.parent.mkdir(parents=True, exist_ok=True)
        upsert_facts_parquet(fact, path)
        refresh_edge_support(config)
        return {"doc_id":doc_id,"facts":1,"skipped":None,"nodes_merged":0,"relationships_merged":0}
    monkeypatch.setattr(loop, "retrieve_for_entity", retrieve)
    monkeypatch.setattr(loop, "process_document", process)
    summary = run_expand(expand_config, connection, None, ontology)
    assert max(called_tiers) == 4
    assert 5 not in called_tiers
    assert summary["documents_selected_cumulative"] <= 320
    assert summary["frontier_by_tier"][5] > 0
    assert connection.execute("SELECT COUNT(*) FROM frontier WHERE tier=0 AND attempts>0").fetchone()[0] == 4
    # Soft scheduling batches no longer permanently exclude suppliers: more than
    # the batch size may be attempted within a large offline run.
    for tier in (1, 2, 3, 4):
        started = connection.execute(
            "SELECT COUNT(*) FROM frontier WHERE tier=? AND attempts>0", (tier,),
        ).fetchone()[0]
        assert started >= 1
    # Tier5 nodes are stored but never searched (terminal).
    assert connection.execute(
        "SELECT COUNT(*) FROM frontier WHERE tier=5 AND expandable=1 AND status=?",
        (state.PENDING,),
    ).fetchone()[0] == 0


def test_only_two_strong_results_do_not_fill_target_with_portals(expand_config, connection, ontology, monkeypatch):
    configure(expand_config, seeds=("BMW",), target=8, hard=12)
    def retrieve(config, name, key, tier, connection, manifest=None, **_kwargs):
        explicit = [SearchResult(provider="test", query="q", title="CATL supplies BMW",
            text=DOCUMENT_TEXT + f" Release {i}", url=f"https://example.test/{i}", entity_key=key, tier=tier)
            for i in range(2)]
        portals = [SearchResult(provider="test", query="q", title="BMW supplier portal login and vendor registration",
            text=f"Register as a supplier to access the BMW sourcing portal. Region {i}",
            url=f"https://example.test/portal/{i}", entity_key=key, tier=tier) for i in range(6)]
        return explicit + portals
    monkeypatch.setattr("schain.expansion.loop.retrieve_for_entity", retrieve)
    summary = run_expand(expand_config, connection, StubLLM(), ontology)
    assert summary["documents_selected_cumulative"] == 2
    assert summary["documents_processed"] == 2


def test_same_batch_duplicate_content_does_not_consume_unique_budget(expand_config, connection, ontology, monkeypatch):
    configure(expand_config, seeds=("BMW",))
    def retrieve(config, name, key, tier, connection, manifest=None, **_kwargs):
        return [SearchResult(provider="test", query="q", title="CATL supplies BMW", text=DOCUMENT_TEXT,
            url=f"https://example.test/{i}", entity_key=key, tier=tier) for i in range(5)]
    monkeypatch.setattr("schain.expansion.loop.retrieve_for_entity", retrieve)
    llm = StubLLM()
    summary = run_expand(expand_config, connection, llm, ontology)
    assert summary["documents_selected_cumulative"] == 1
    assert summary["documents_processed"] == 1
    assert llm.extraction_calls == 1


def test_resume_repairs_frontier_after_final_document_crash_at_cap(expand_config, connection, ontology, monkeypatch):
    configure(expand_config, seeds=("BMW",), cap=1)
    mocked_candidates(monkeypatch, count=1)
    run_expand(expand_config, connection, StubLLM(), ontology)
    # Model a crash after the accepted document, before its frontier transaction.
    connection.execute("DELETE FROM frontier WHERE tier=1")
    connection.execute("UPDATE frontier SET status=? WHERE tier=0", (state.PROCESSING,))
    connection.commit()
    summary = run_expand(expand_config, connection, StubLLM(), ontology)
    assert summary["stop_reason"] in ("document_budget", "max_documents", "frontier_empty", "entity_document_cap")
    assert state.frontier_tier_counts(connection).get(0) == 1
    assert state.frontier_tier_counts(connection).get(1, 0) >= 1
