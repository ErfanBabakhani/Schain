"""Coverage / graph-growth upgrade: budgets, recovery, telemetry (offline)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from schain import state
from schain.config import load_config, seed_entities
from schain.expansion.budget import documents_per_run_limit, entity_document_limits
from schain.expansion.loop import _frontier_completion, _persisted_graph_counts, run_expand
from schain.kg.list_extract import extract_supplier_list_facts
from schain.llm.query_planner import plan_upstream_queries
from schain.sources.base import SearchResult
from schain.sources.retrieve import clean_news_title, recover_publisher_article
from .conftest import DOCUMENT_TEXT, StubLLM
from .test_expansion_budget_v030 import configure, mocked_candidates


def test_production_seeds_and_tier5():
    config = load_config("config/expansion.yaml")
    assert seed_entities(config) == [
        "Toyota", "Tesla Inc.", "BMW", "BYD Co., Ltd.",
    ]
    assert config["expansion"]["max_tier"] == 5
    assert documents_per_run_limit(config["expansion"]) == 320
    assert entity_document_limits(config["expansion"], 0) == (24, 36)


def test_multi_run_document_budget_is_not_lifetime_cap(expand_config, connection, ontology, monkeypatch):
    configure(expand_config, seeds=("BMW",), cap=5, target=8, hard=12)
    expand_config["expansion"]["max_documents"] = 5
    expand_config["expansion"]["max_documents_per_run"] = 5
    mocked_candidates(monkeypatch, count=10)
    first = run_expand(expand_config, connection, StubLLM(), ontology)
    assert first["documents_processed_this_run"] == 5
    assert first["lifetime_documents_processed"] == 5
    assert first["stop_reason"] in ("document_budget", "max_documents")
    assert first["completion"]["complete"] is False
    expand_config["expansion"]["max_documents_per_run"] = 5
    expand_config["expansion"]["max_documents"] = 5
    second = run_expand(expand_config, connection, StubLLM(), ontology)
    assert second["lifetime_documents_processed"] > 5
    assert second["documents_processed_this_run"] >= 1


def test_terminal_tier_does_not_count_as_pending_expandable(connection):
    state.enqueue_entity(connection, "BMW", 0, entity_id="company:bmw")
    state.set_frontier_status(connection, "company:bmw", state.DONE)
    state.enqueue_entity(
        connection, "DeepMine", 5, entity_id="company:deepmine",
        expandable=False, stop_reason="terminal_tier",
    )
    completion = _frontier_completion(connection, 5, "frontier_empty")
    assert completion["pending_expandable_frontier"] == 0
    assert completion["terminal_frontier"] >= 1
    assert completion["complete"] is True


def test_graph_growth_uses_persisted_facts_not_neo4j(expand_config, connection, tmp_path):
    expand_config["storage"]["extracted_facts"] = str(tmp_path / "facts.parquet")
    expand_config["storage"]["edge_support"] = str(tmp_path / "edges.parquet")
    state.enqueue_entity(connection, "BMW", 0, entity_id="company:bmw")
    state.enqueue_entity(connection, "CATL", 1, entity_id="company:catl")
    facts = pd.DataFrame([{
        "doc_id": "d1", "fact_id": "f1", "chunk_index": 0,
        "subject_id": "company:catl", "subject_name": "CATL", "subject_type": "Company",
        "relation": "suppliesTo",
        "object_id": "company:bmw", "object_name": "BMW", "object_type": "Company",
        "evidence": "CATL supplies BMW.",
    }])
    facts.to_parquet(expand_config["storage"]["extracted_facts"])
    counts = _persisted_graph_counts(expand_config, connection)
    assert counts["total_nodes"] >= 2
    assert counts["tiered_nodes"] == 2
    assert counts["accepted_facts"] == 1


def test_clean_news_title_strips_publisher_suffix():
    assert "Reuters" not in clean_news_title(
        "CATL wins large BMW battery contract - Reuters"
    )
    assert "BMW" in clean_news_title("CATL wins large BMW battery contract | Bloomberg")


def test_title_lookup_recovers_matching_publisher_article():
    result = SearchResult(
        provider="google_news",
        query="q",
        title="CATL wins large BMW battery supply contract today - Reuters",
        url="https://news.google.com/rss/articles/abc",
        publisher_name="Reuters",
        publisher_url="https://www.reuters.com/",
    )
    hit = SearchResult(
        provider="general_web",
        query="q",
        title="CATL wins large BMW battery supply contract today",
        url="https://www.reuters.com/article/catl-bmw-2024",
        snippet="Reuters",
    )

    async def fake_provider(client, query, max_results, options):
        return [hit]

    with patch.dict("schain.sources.retrieve.PROVIDERS", {"general_web": fake_provider}):
        url = asyncio.run(recover_publisher_article(object(), result, {"enabled": True}))
    assert url == hit.url


def test_title_lookup_rejects_wrong_article():
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
        title="Completely unrelated market wrap for Friday trading",
        url="https://www.reuters.com/markets/wrap",
        snippet="Markets",
    )

    async def fake_provider(client, query, max_results, options):
        return [wrong]

    with patch.dict("schain.sources.retrieve.PROVIDERS", {"general_web": fake_provider}):
        url = asyncio.run(recover_publisher_article(object(), result, {"enabled": True}))
    assert url is None
    assert result.fetch_error in ("title_lookup_mismatch", "title_lookup_no_result", "domain_mismatch")


def test_title_lookup_no_result_when_provider_empty():
    result = SearchResult(
        provider="google_news",
        query="q",
        title="CATL wins large BMW battery supply contract today",
        url="https://news.google.com/rss/articles/abc",
        publisher_name="Reuters",
        publisher_url="https://www.reuters.com/",
    )

    async def fake_provider(client, query, max_results, options):
        return []

    with patch.dict("schain.sources.retrieve.PROVIDERS", {"general_web": fake_provider}):
        url = asyncio.run(recover_publisher_article(object(), result, {"enabled": True}))
    assert url is None
    assert result.fetch_error == "title_lookup_no_result"


def test_query_planner_includes_high_yield_and_categories(tmp_path, expand_config):
    expand_config["storage"]["state_db"] = str(tmp_path / "state.db")
    expand_config["expansion"]["retrieval"]["query_planner_max_queries"] = 12
    expand_config["expansion"]["retrieval"]["query_planner_max_categories"] = 8
    plan = plan_upstream_queries(
        llm=None, config=expand_config, entity_name="BMW", tier=0,
    )
    assert plan["source"] == "heuristic"
    assert len(plan["upstream_categories"]) <= 8
    joined = " ".join(plan["queries"]).casefold()
    assert "supplier list" in joined or "annual report" in joined or "sustainability" in joined
    assert any("battery" in q.casefold() or "semiconductor" in q.casefold() for q in plan["queries"])


def test_supplier_award_list_extracts_multiple_grounded_facts():
    text = (
        "BMW supplier awards 2024\n"
        "- CATL\n"
        "- Samsung SDI\n"
        "- Bosch\n"
    )
    rows = extract_supplier_list_facts(
        text, doc_id="d1", origin_name="BMW", origin_id="company:bmw",
    )
    assert len(rows) >= 2
    assert all(r["object_name"] == "BMW" for r in rows)
    assert all(r["relation"] == "suppliesTo" for r in rows)
    assert all("evidence" in r and r["evidence"] for r in rows)


def test_pdf_text_document_type_marked_for_pdf_url():
    from schain.sources.retrieve import url_looks_pdf
    assert url_looks_pdf("https://example.com/reports/suppliers.pdf")
    assert not url_looks_pdf("https://example.com/article")
