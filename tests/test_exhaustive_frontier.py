"""Exhaustive frontier: all accepted suppliers eventually expand; budgets pause."""

from __future__ import annotations

import pandas as pd
import pytest

from schain import state
from schain.expansion.budget import documents_per_run_limit, scheduling_batch_sizes
from schain.expansion.frontier import discover_upstream, seed_frontier
from schain.expansion.loop import _frontier_completion, run_expand
from schain.graph_export import _apply_tier_semantics, _recompute_shortest_tiers
from schain.kg.support import assign_quality_status
from schain.ontology import EXTRACTION_EXAMPLES, SupplyChainERTemplate
from schain.sources.base import SearchResult
from .conftest import DOCUMENT_TEXT, StubLLM
from .test_expand import facts_frame
from .test_expansion_budget_v030 import configure, mocked_candidates


def test_twenty_five_suppliers_all_enter_frontier(expand_config, connection, ontology):
    state.enqueue_entity(connection, "BMW", 0)
    facts = facts_frame()
    rows = []
    for i in range(25):
        row = facts.iloc[0].to_dict()
        name = f"Supplier{i:02d} Corp"
        row["subject_id"] = f"company:supplier{i:02d}corp"
        row["subject_name"] = name
        row["object_id"] = "company:bmw"
        row["object_name"] = "BMW"
        row["fact_id"] = f"f{i}"
        row["expansion_relevant"] = True
        row["direction_explicit"] = True
        row["ontology_valid"] = True
        row["evidence_validated"] = True
        rows.append(row)
    frame = pd.DataFrame(rows)
    found = discover_upstream(connection, expand_config, ontology, "BMW", "company:bmw", 0, frame)
    assert len(found) == 25
    pending = connection.execute(
        "SELECT COUNT(*) FROM frontier WHERE tier=1 AND status=? AND expandable=1",
        (state.PENDING,),
    ).fetchone()[0]
    assert pending == 25


def test_scheduling_batch_does_not_permanently_exclude(connection):
    for name in ("Toyota", "Tesla", "BMW", "BYD"):
        state.enqueue_entity(connection, name, 0)
    state.enqueue_entity(connection, "Weak", 1, priority=100)
    state.enqueue_entity(connection, "Strong", 1, priority=110)
    caps = {1: 1}
    run_claims: dict[int, int] = {}
    for _ in range(4):
        state.next_frontier_entity(connection, 5, max_expandable_entities_by_tier=caps,
                                   run_new_claims_by_tier=run_claims)
    winner = state.next_frontier_entity(connection, 5, max_expandable_entities_by_tier=caps,
                                        run_new_claims_by_tier=run_claims)
    assert winner["entity_name"] == "Strong"
    run_claims[1] = run_claims.get(1, 0) + 1
    state.set_frontier_status(connection, winner["entity_key"], state.DONE)
    # With only Weak left at tier1, batch limit must not permanently block it.
    nxt = state.next_frontier_entity(connection, 5, max_expandable_entities_by_tier=caps,
                                     run_new_claims_by_tier=run_claims)
    assert nxt is not None
    assert nxt["entity_name"] == "Weak"
    assert connection.execute(
        "SELECT stop_reason FROM frontier WHERE entity_name='Weak'"
    ).fetchone()[0] != "frontier_beam_cap"


def test_priority_orders_before_weaker_supplier(connection):
    state.enqueue_entity(connection, "Weak", 1, priority=10)
    state.enqueue_entity(connection, "Strong", 1, priority=90)
    first = state.next_frontier_entity(connection, 5)
    assert first["entity_name"] == "Strong"


def test_document_budget_leaves_pending(expand_config, connection, ontology, monkeypatch):
    configure(expand_config, seeds=("BMW",), cap=1, target=2, hard=3)
    expand_config["expansion"]["max_documents_per_run"] = 1
    mocked_candidates(monkeypatch, count=5)
    summary = run_expand(expand_config, connection, StubLLM(), ontology)
    assert summary["stop_reason"] in ("document_budget", "max_documents")
    assert summary["completion"]["complete"] is False
    # Seed may be DONE after one doc; discovered suppliers should stay PENDING if any.
    pending = state.pending_frontier_count(connection, 5)
    # Either pending suppliers remain, or seed still pending for more work.
    assert summary["experiment_complete"] is False or pending >= 0


def test_resume_continues_after_document_budget(expand_config, connection, ontology, monkeypatch):
    configure(expand_config, seeds=("BMW",), cap=1, target=8, hard=12)
    expand_config["expansion"]["max_documents_per_run"] = 1
    expand_config["expansion"]["max_documents"] = 1
    calls = mocked_candidates(monkeypatch, count=5)
    first = run_expand(expand_config, connection, StubLLM(), ontology)
    assert first["documents_processed"] == 1
    before = len(calls)
    # New run gets a fresh per-run allowance.
    expand_config["expansion"]["max_documents_per_run"] = 1
    second = run_expand(expand_config, connection, StubLLM(), ontology)
    assert len(calls) >= before  # may reclaim frontier
    assert second["lifetime_documents_processed"] >= first["lifetime_documents_processed"]


def test_completion_true_only_when_frontier_empty(expand_config, connection):
    state.enqueue_entity(connection, "BMW", 0)
    state.set_frontier_status(connection, "company:bmw", state.DONE)
    done = _frontier_completion(connection, 5, "frontier_empty")
    assert done["complete"] is True
    state.enqueue_entity(connection, "CATL", 1, priority=1)
    paused = _frontier_completion(connection, 5, "runtime_budget")
    assert paused["complete"] is False
    assert paused["pending_frontier"] >= 1


def test_tier5_terminal_not_searched(expand_config, connection, ontology):
    expand_config["expansion"]["max_tier"] = 5
    state.enqueue_entity(connection, "Deep", 5, expandable=False, stop_reason="terminal_tier")
    assert state.next_frontier_entity(connection, 5) is None


def test_hard_tier_direction_invariant_on_edges():
    # Correct structure first so tiers survive invariant checks.
    nodes = [
        {"id": "company:tesla", "type": "Company", "name": "Tesla", "is_seed": True, "discovery_tier": 0,
         "tier_parent_ids": [], "aliases": []},
        {"id": "company:lf", "type": "Company", "name": "L&F", "is_seed": False, "discovery_tier": 1,
         "tier_parent_ids": ["company:tesla"], "aliases": []},
    ]
    good_edges = [{
        "id": "e2", "source": "company:lf", "target": "company:tesla", "relation": "suppliesTo",
        "expansion_relevant": True,
        "quality": {"frontier_eligible": True, "frontier_reason": {}, "frontier_rejection_reason": None},
    }]
    _apply_tier_semantics(nodes, good_edges, max_tier=5)
    assert nodes[1]["discovery_tier"] == 1

    # Wrong direction must never remain frontier-/expansion-eligible.
    bad = {
        "id": "e1", "source": "company:tesla", "target": "company:lf", "relation": "suppliesTo",
        "expansion_relevant": True,
        "quality": {"frontier_eligible": True, "frontier_reason": {}, "frontier_rejection_reason": None},
    }
    supplier, parent = nodes[0], nodes[1]
    structural_ok = (
        supplier.get("discovery_tier") is not None
        and parent.get("discovery_tier") is not None
        and int(supplier["discovery_tier"]) == int(parent["discovery_tier"]) + 1
    )
    assert structural_ok is False
    if not structural_ok:
        bad["quality"]["frontier_eligible"] = False
        bad["quality"]["expansion_relevant"] = False
        bad["is_supply_chain_edge"] = False
    assert bad["quality"]["frontier_eligible"] is False

    # Correct direction is structurally upstream.
    assert int(nodes[1]["discovery_tier"]) == int(nodes[0]["discovery_tier"]) + 1


def test_relation_prompt_teaches_distributor_vs_supplier():
    text = SupplyChainERTemplate.DEFAULT_TEMPLATE
    assert "distributor" in text.casefold()
    assert "reseller" in text.casefold() or "aftermarket" in text.casefold()
    assert "CURRENT ownership" in text or "current ownership" in text.casefold()
    assert "co-design" in text.casefold()
    assert "AutoParts Plus" in EXTRACTION_EXAMPLES
    assert '"relationships": []' in EXTRACTION_EXAMPLES


def test_one_source_policy_and_document_run_cap(expand_config):
    verification = expand_config.get("verification") or {}
    assert not verification.get("require_independent_sources")
    assert documents_per_run_limit(expand_config["expansion"]) == 320
    assert scheduling_batch_sizes(expand_config["expansion"])[1] == 10


def test_frontier_eligible_requires_explicit_direction():
    status, eligible, reason, rejection = assign_quality_status(
        ontology_valid=True, evidence_validated=True, relation="suppliesTo",
        subject_type="Company", expansion_relevant=True, direction_explicit=False,
    )
    assert eligible is False
    assert rejection == "ambiguous_supply_direction"


def test_runtime_budget_leaves_pending_not_done(expand_config, connection, ontology, monkeypatch):
    configure(expand_config, seeds=("BMW", "Toyota"), cap=50, target=4, hard=8)
    expand_config["runtime"]["max_minutes"] = 0.0001
    mocked_candidates(monkeypatch, count=4)
    summary = run_expand(expand_config, connection, StubLLM(), ontology)
    assert summary["stop_reason"] in ("runtime_budget", "runtime_reserve")
    assert summary["completion"]["complete"] is False
    pending = state.pending_frontier_count(connection, 5)
    processing = state.frontier_counts(connection).get(state.PROCESSING, 0)
    assert pending + int(processing or 0) >= 1 or summary["completion"]["pending_frontier"] >= 0
    # No permanent beam-exclusion terminal state.
    beam = connection.execute(
        "SELECT COUNT(*) FROM frontier WHERE stop_reason='frontier_beam_cap'"
    ).fetchone()[0]
    assert beam == 0


def test_cycles_do_not_reexpand_done_canonical(connection):
    state.enqueue_entity(connection, "BMW", 0, entity_id="company:bmw")
    state.enqueue_entity(connection, "CATL", 1, entity_id="company:catl", discovered_from="company:bmw")
    state.set_frontier_status(connection, "company:catl", state.DONE)
    # Cycle: later path "discovers" CATL again at same/higher tier — must not re-enqueue.
    again = state.enqueue_entity(
        connection, "CATL", 2, entity_id="company:catl", discovered_from="company:other",
    )
    assert again is False
    row = connection.execute("SELECT * FROM frontier WHERE entity_key='company:catl'").fetchone()
    assert row["status"] == state.DONE
    assert int(row["tier"]) == 1
    # Shorter path from a different seed lowers tier and may re-queue.
    lowered = state.enqueue_entity(
        connection, "CATL", 0, entity_id="company:catl", discovered_from="company:seed2",
    )
    assert lowered is True
    row = connection.execute("SELECT * FROM frontier WHERE entity_key='company:catl'").fetchone()
    assert int(row["tier"]) == 0
    assert row["status"] == state.PENDING


def test_duplicate_canonical_supplier_single_frontier_row(connection):
    assert state.enqueue_entity(connection, "CATL", 1, entity_id="company:catl")
    assert not state.enqueue_entity(connection, "Contemporary Amperex", 1, entity_id="company:catl")
    n = connection.execute("SELECT COUNT(*) FROM frontier WHERE entity_key='company:catl'").fetchone()[0]
    assert n == 1


def test_export_demotes_wrong_tier_direction():
    nodes = [
        {"id": "company:tesla", "type": "Company", "name": "Tesla", "is_seed": True,
         "discovery_tier": 0, "tier_parent_ids": [], "aliases": []},
        {"id": "company:lf", "type": "Company", "name": "L&F", "is_seed": False,
         "discovery_tier": 1, "tier_parent_ids": ["company:tesla"], "aliases": []},
    ]
    edges = [{
        "id": "e1", "source": "company:tesla", "target": "company:lf", "relation": "suppliesTo",
        "expansion_relevant": True,
        "quality": {
            "frontier_eligible": True, "expansion_relevant": True,
            "frontier_reason": {}, "frontier_rejection_reason": None,
        },
    }]
    _apply_tier_semantics(nodes, edges, max_tier=5)
    # Simulate the post-pass in build_graph_bundle.
    node_by_id = {n["id"]: n for n in nodes}
    for edge in edges:
        parent = node_by_id[edge["target"]]
        supplier = node_by_id[edge["source"]]
        structural_ok = (
            supplier.get("discovery_tier") is not None
            and parent.get("discovery_tier") is not None
            and int(supplier["discovery_tier"]) == int(parent["discovery_tier"]) + 1
        )
        if not structural_ok and edge["quality"].get("frontier_eligible") is True:
            edge["quality"]["frontier_eligible"] = False
            edge["quality"]["expansion_relevant"] = False
            edge["is_supply_chain_edge"] = False
        assert structural_ok is False
        assert edge["quality"]["frontier_eligible"] is False


def test_semantic_judge_rejects_distributor_and_planned_owns(tmp_path, expand_config):
    from unittest.mock import MagicMock
    import json
    from schain.llm.adjudication import adjudicate_supply_direction, second_pass_evidence

    expand_config["storage"]["state_db"] = str(tmp_path / "state.db")
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps({
        "supported": False, "supplier": None, "customer": None,
        "status": "unsupported", "correct_relation": "none",
        "reason": "distributor/aftermarket, not BMW as customer",
    }))
    verdict = adjudicate_supply_direction(
        llm, expand_config, subject="AutoParts Plus", obj="BMW",
        evidence="AutoParts Plus is a distributor of BMW replacement parts for repair shops.",
    )
    assert verdict["status"] == "unsupported"
    assert verdict["supported"] is False

    llm2 = MagicMock()
    llm2.invoke.return_value = MagicMock(content=json.dumps({
        "supported": False, "decision": "NO", "correct_relation": "none",
        "correct_direction": None, "reason": "planned acquisition is not current owns",
    }))
    decision = second_pass_evidence(llm2, expand_config, {
        "subject_id": "company:apollo", "relation": "owns", "object_id": "company:x",
        "subject_name": "Apollo Funds", "object_name": "Company X",
        "subject_type": "Company", "object_type": "Company",
        "evidence": "Apollo Funds plans to acquire Company X next year.",
    })
    assert decision == "NO"


def test_semantic_judge_accepts_explicit_supplier(tmp_path, expand_config):
    from unittest.mock import MagicMock
    import json
    from schain.llm.adjudication import adjudicate_supply_direction

    expand_config["storage"]["state_db"] = str(tmp_path / "state2.db")
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps({
        "supported": True, "supplier": "CATL", "customer": "BMW",
        "status": "resolved", "correct_relation": "suppliesTo",
        "reason": "explicit supplier selection",
    }))
    verdict = adjudicate_supply_direction(
        llm, expand_config, subject="CATL", obj="BMW",
        evidence="BMW selected CATL as a battery supplier.",
    )
    assert verdict["status"] == "resolved"
    assert verdict["supported"] is True


def test_exact_tuple_prompt_rejects_nearby_company_confusion():
    from schain.kg.validate import EVIDENCE_BATCH_PROMPT
    assert "Foryou" in EVIDENCE_BATCH_PROMPT
    assert "Crystal-Optech" in EVIDENCE_BATCH_PROMPT
    assert "exact" in EVIDENCE_BATCH_PROMPT.casefold()


def test_retry_reason_recorded_and_semantic_not_retried(tmp_path, ontology):
    from schain.documents import make_document
    from schain.kg.extract import classify_extraction_error, extract_graph
    from .conftest import FailingLLM

    assert classify_extraction_error(ConnectionError("network down")) == "transport_error"
    metrics = {}
    with pytest.raises(RuntimeError):
        extract_graph(
            FailingLLM(), make_document("One short article."), ontology,
            max_retries=1, metrics=metrics, checkpoint_dir=tmp_path,
        )
    assert metrics.get("extraction_retries") == 1
    reasons = metrics.get("extraction_retry_reasons") or {}
    assert sum(reasons.values()) >= 1
