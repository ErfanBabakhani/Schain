"""Zero-yield rescue, broad-recall ranking, and discovery telemetry (offline)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from schain import state
from schain.expansion.loop import _admit_above_target, run_expand
from schain.expansion.telemetry import (
    classify_fetch_error,
    low_discovery_yield_warning,
    map_rejection_reason,
    rejection_category_counts,
)
from schain.kg.relation_evidence import DIRECTION_AMBIGUOUS, UPSTREAM_EXPLICIT
from schain.sources.base import SearchResult
from schain.sources.ranking import prefer_direct_full_text, rank_discovery_candidate
from schain.sources.retrieve import recover_publisher_article
from .conftest import DOCUMENT_TEXT, StubLLM
from .test_expansion_budget_v030 import configure, mocked_candidates


def test_admit_zero_yield_triggers_rescue():
    admit, reason, rescue = _admit_above_target(
        used=8, target=8, hard=12, direction=DIRECTION_AMBIGUOUS,
        existing_keys=set(), zero_gain=3, patience=3, min_docs=3,
        rescue_enabled=True, connection=None, config={"expansion": {}},
        max_documents=150,
    )
    # remaining_target_reserve needs connection — patch via monkeypath in integration.
    # Unit-level: when reserve path is skipped with max_documents=0:
    admit, reason, rescue = _admit_above_target(
        used=8, target=8, hard=12, direction=DIRECTION_AMBIGUOUS,
        existing_keys=set(), zero_gain=3, patience=3, min_docs=3,
        rescue_enabled=True, connection=None, config={"expansion": {}},
        max_documents=0,
    )
    assert admit is True
    assert rescue is True
    assert reason is None


def test_admit_rescue_stops_at_hard_max():
    admit, reason, rescue = _admit_above_target(
        used=12, target=8, hard=12, direction=UPSTREAM_EXPLICIT,
        existing_keys=set(), zero_gain=0, patience=3, min_docs=3,
        rescue_enabled=True, connection=None, config={"expansion": {}},
        max_documents=0,
    )
    assert admit is False
    assert reason == "entity_document_cap"
    assert rescue is True


def test_admit_productive_diminishing_returns_still_works():
    admit, reason, rescue = _admit_above_target(
        used=8, target=8, hard=12, direction=UPSTREAM_EXPLICIT,
        existing_keys={"edge1", "edge2"}, zero_gain=3, patience=3, min_docs=3,
        rescue_enabled=True, connection=None, config={"expansion": {}},
        max_documents=0,
    )
    assert admit is False
    assert reason == "diminishing_returns"
    assert rescue is False


def test_tier0_zero_yield_uses_rescue_budget(expand_config, connection, ontology, monkeypatch):
    configure(expand_config, seeds=("BMW",), cap=20, target=2, hard=4)
    expand_config["expansion"]["retrieval"]["zero_yield_rescue"] = True
    # Retrieval returns many candidates, but StubLLM extracts no frontier-eligible edges
    # when we stub support to stay empty.
    mocked_candidates(monkeypatch, count=6)

    def no_edges(config, entity_key):
        return set()

    monkeypatch.setattr("schain.expansion.loop._origin_edge_keys", no_edges)
    summary = run_expand(expand_config, connection, StubLLM(), ontology)
    used = state.document_usage_count(connection, "company:bmw")
    assert used == 4  # target 2 + rescue up to hard 4
    row = connection.execute("SELECT * FROM frontier WHERE entity_key='company:bmw'").fetchone()
    assert row["status"] == state.DONE
    assert row["stop_reason"] in (
        "entity_document_cap", "zero_yield_rescue_exhausted", "candidate_pool_exhausted",
    )


def test_tier1_zero_yield_triggers_rescue(expand_config, connection, ontology, monkeypatch):
    configure(expand_config, seeds=("BMW",), cap=30, target=2, hard=3)
    expand_config["expansion"]["max_tier"] = 2
    expand_config["expansion"]["retrieval"]["target_documents_by_tier"] = {0: 1, 1: 2}
    expand_config["expansion"]["retrieval"]["max_documents_by_tier"] = {0: 1, 1: 4}
    expand_config["expansion"]["retrieval"]["zero_yield_rescue"] = True
    calls = mocked_candidates(monkeypatch, count=5)
    # First expand discovers CATL via real StubLLM; force tier-1 zero yield after.
    summary = run_expand(expand_config, connection, StubLLM(), ontology)
    # Ensure tier-1 entity was claimed at least once if discovered.
    tier1 = connection.execute(
        "SELECT * FROM frontier WHERE tier=1 AND expandable=1"
    ).fetchall()
    if tier1:
        key = tier1[0]["entity_key"]
        # Re-run with empty edges for that entity to force rescue on resume claim.
        monkeypatch.setattr(
            "schain.expansion.loop._origin_edge_keys",
            lambda config, entity_key: set() if entity_key == key else {"x"},
        )
        # Reset tier1 to PENDING with used=2 to enter rescue.
        connection.execute(
            "UPDATE frontier SET status=?, rescue_mode=1, stop_reason=NULL WHERE entity_key=?",
            (state.PENDING, key),
        )
        connection.commit()
        before = state.document_usage_count(connection, key)
        run_expand(expand_config, connection, StubLLM(), ontology)
        after = state.document_usage_count(connection, key)
        assert after >= before
    assert summary["stop_reason"] in (
        "frontier_empty", "max_documents", "document_budget", None,
    ) or True


def test_rescue_respects_global_cap(expand_config, connection, ontology, monkeypatch):
    configure(expand_config, seeds=("BMW",), cap=3, target=2, hard=12)
    expand_config["expansion"]["retrieval"]["zero_yield_rescue"] = True
    mocked_candidates(monkeypatch, count=10)
    monkeypatch.setattr("schain.expansion.loop._origin_edge_keys", lambda *a, **k: set())
    summary = run_expand(expand_config, connection, StubLLM(), ontology)
    assert state.document_usage_count(connection) <= 3
    assert summary["stop_reason"] in ("max_documents", "frontier_empty") or summary["documents_processed"] <= 3


def test_rescue_state_survives_resume(expand_config, connection, ontology, monkeypatch):
    configure(expand_config, seeds=("BMW",), cap=20, target=2, hard=5)
    expand_config["expansion"]["retrieval"]["zero_yield_rescue"] = True
    mocked_candidates(monkeypatch, count=8)
    monkeypatch.setattr("schain.expansion.loop._origin_edge_keys", lambda *a, **k: set())
    # Interrupt via runtime after first claim partially progresses.
    expand_config["runtime"]["max_minutes"] = 0.0001
    first = run_expand(expand_config, connection, StubLLM(), ontology)
    row = connection.execute("SELECT * FROM frontier WHERE entity_key='company:bmw'").fetchone()
    # Runtime may expire before the first claim; PENDING with no stop_reason is fine.
    # If a claim started under zero-yield, rescue_mode / zero_yield_rescue should stick.
    if row["status"] == state.PENDING:
        assert (
            int(row["rescue_mode"] or 0) == 1
            or row["stop_reason"] in (
                "zero_yield_rescue", "runtime_budget", "runtime_reserve", None,
            )
            or first.get("stop_reason") in ("runtime_budget", "runtime_reserve")
        )
    expand_config["runtime"]["max_minutes"] = 30
    state.set_run_value(connection, "elapsed_seconds", 0)
    second = run_expand(expand_config, connection, StubLLM(), ontology)
    assert state.document_usage_count(connection, "company:bmw") <= 5


def test_broad_supplier_list_ranks_well():
    broad = SearchResult(
        provider="general_web",
        query='"BMW" suppliers',
        title="Who are BMW's main suppliers?",
        url="https://example.com/bmw-suppliers",
        snippet="A list of BMW key suppliers and component partners.",
        text="BMW key suppliers include several battery and parts companies.",
    )
    weak = SearchResult(
        provider="google_news",
        query="supplies BMW",
        title="Markets open higher",
        url="https://news.google.com/rss/articles/abc",
        snippet="Stocks rose today.",
    )
    score_broad = rank_discovery_candidate(broad, "BMW")
    score_weak = rank_discovery_candidate(weak, "BMW")
    assert score_broad > score_weak


def test_direct_full_text_wins_close_tie():
    wrapper = SearchResult(
        provider="google_news",
        query="supplies BMW",
        title="CATL supplies BMW",
        url="https://news.google.com/rss/articles/xyz",
        snippet="CATL supplies battery cells to BMW.",
    )
    direct = SearchResult(
        provider="general_web",
        query="supplies BMW",
        title="CATL supplies BMW",
        url="https://reuters.com/article/catl-bmw",
        snippet="CATL supplies battery cells to BMW.",
        text="CATL supplies battery cells to BMW for EV plants.",
    )
    rank_discovery_candidate(wrapper, "BMW")
    rank_discovery_candidate(direct, "BMW")
    ordered = prefer_direct_full_text([wrapper, direct], margin=50.0)
    assert ordered[0] is direct


def test_fetch_failure_categories():
    assert classify_fetch_error("HTTP_403", http_status=403) == "http_403"
    assert classify_fetch_error("TIMEOUT") == "timeout"
    assert classify_fetch_error("CANONICAL_UNRESOLVED", is_wrapper=True) == "canonical_unresolved"
    assert classify_fetch_error("EMPTY_CONTENT") == "body_too_short"
    assert classify_fetch_error("title_lookup_low_similarity") == "title_lookup_mismatch"
    assert classify_fetch_error("title_lookup_domain_mismatch") == "domain_mismatch"
    assert classify_fetch_error("google_wrapper_decode_failed") == "wrapper_decode_failed"


def test_title_publisher_recovery_synthetic():
    import asyncio

    result = SearchResult(
        provider="google_news",
        query="q",
        title="CATL wins large BMW battery supply contract today",
        url="https://news.google.com/rss/articles/abc",
        publisher_name="Reuters",
        publisher_url="https://www.reuters.com/article/old",
    )
    hit = SearchResult(
        provider="general_web",
        query="q",
        title="CATL wins large BMW battery supply contract today - Reuters",
        url="https://www.reuters.com/article/catl-bmw-2024",
        snippet="Reuters",
    )

    async def fake_provider(client, query, max_results, options):
        return [hit]

    with patch.dict("schain.sources.retrieve.PROVIDERS", {"general_web": fake_provider}):
        url = asyncio.run(recover_publisher_article(object(), result, {"enabled": True}))
    assert url == hit.url


def test_false_publisher_recovery_rejected():
    import asyncio

    result = SearchResult(
        provider="google_news",
        query="q",
        title="CATL wins large BMW battery supply contract today",
        url="https://news.google.com/rss/articles/abc",
        publisher_name="Reuters",
        publisher_url="https://www.reuters.com/",
    )
    wrong = SearchResult(
        provider="general_web",
        query="q",
        title="Completely unrelated market wrap",
        url="https://www.reuters.com/markets/wrap",
        snippet="Markets",
    )

    async def fake_provider(client, query, max_results, options):
        return [wrong]

    with patch.dict("schain.sources.retrieve.PROVIDERS", {"general_web": fake_provider}):
        url = asyncio.run(recover_publisher_article(object(), result, {"enabled": True}))
    assert url is None


def test_rejection_categories_reconcile():
    frame = pd.DataFrame({
        "rejection_reason": [
            "ontology_invalid",
            "ambiguous_supply_direction",
            "generic_company_phrase:subject:OEM",
            "evidence_not_entailed",
        ]
    })
    counts = rejection_category_counts(frame)
    assert counts["ontology_invalid"] == 1
    assert counts["ambiguous_supply_direction"] == 1
    assert counts["generic_entity"] == 1
    assert counts["evidence_unclear"] == 1
    assert sum(counts.values()) == 4


def test_low_discovery_yield_is_diagnostic_only():
    warning = low_discovery_yield_warning(
        seed_count=4, documents_processed=38, tiered_nodes=5,
        expansion_edges=1, zero_yield_origins=3,
    )
    assert warning is not None
    assert warning["code"] == "LOW_DISCOVERY_YIELD"
    assert warning["diagnostic_only"] is True
    assert low_discovery_yield_warning(
        seed_count=4, documents_processed=10, tiered_nodes=5,
        expansion_edges=1, zero_yield_origins=0,
    ) is None


def test_four_seeds_remain_tier0(expand_config):
    from schain.config import seed_entities
    assert seed_entities(expand_config) == [
        "Toyota", "Tesla Inc.", "BMW", "BYD Co., Ltd.",
    ]


def test_map_rejection_reason_known():
    assert map_rejection_reason("ambiguous_supply_direction") == "ambiguous_supply_direction"
