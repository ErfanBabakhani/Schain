"""Retrieval orchestration: per-query checkpointing, persistence, deduplication."""

from __future__ import annotations

import pandas as pd
import pytest

from schain import state
from schain.sources import retrieve
from schain.sources.base import SearchResult

ARTICLE = (
    "BMW has named CATL as a battery supplier for its European electric vehicle "
    "plants, under a multi-year sourcing agreement signed this year."
)


@pytest.fixture
def two_provider_config(expand_config):
    """One query, two providers, so the assertions are about orchestration only."""
    expand_config["expansion"]["sources"] = {
        "wikipedia": {"enabled": True}, "gdelt": {"enabled": True},
        "google_news": {"enabled": False},
    }
    expand_config["expansion"]["search"] = {
        "defaults": {"max_results": 5, "providers": ["wikipedia", "gdelt"]},
        "tiers": {0: {"queries": [{"query": '"{entity}" suppliers'}]}},
    }
    expand_config["expansion"]["relevance"] = {
        "min_chars": 20, "require_entity_mention": True, "keywords": ["supplier"],
    }
    return expand_config


def provider_returning(url, calls, name):
    async def search(client, query, max_results, options):
        calls.append(name)
        return [SearchResult(provider=name, query=query, title=f"{name} story",
                             url=url, text=ARTICLE)]

    return search


def test_results_are_persisted_and_the_query_is_not_reissued(
    two_provider_config, connection, monkeypatch
):
    calls: list[str] = []
    monkeypatch.setitem(retrieve.PROVIDERS, "wikipedia",
                        provider_returning("https://a.test/1", calls, "wikipedia"))
    monkeypatch.setitem(retrieve.PROVIDERS, "gdelt",
                        provider_returning("https://b.test/2", calls, "gdelt"))

    first = retrieve.retrieve_for_entity(
        two_provider_config, "BMW", "company:bmw", 0, connection
    )
    assert len(first) == 2
    assert sorted(calls) == ["gdelt", "wikipedia"]

    # everything was written down, including which query produced it
    frame = pd.read_parquet(two_provider_config["storage"]["search_results"])
    assert len(frame) == 2
    assert frame.search_id.notna().all()
    assert set(frame.provider) == {"wikipedia", "gdelt"}

    # a second run reuses the stored results instead of calling the providers again
    calls.clear()
    second = retrieve.retrieve_for_entity(
        two_provider_config, "BMW", "company:bmw", 0, connection
    )
    assert calls == []
    assert {result.url for result in second} == {result.url for result in first}


def test_the_same_article_from_two_providers_is_deduplicated(
    two_provider_config, connection, monkeypatch
):
    calls: list[str] = []
    shared_url = "https://news.test/same-article"
    monkeypatch.setitem(retrieve.PROVIDERS, "wikipedia",
                        provider_returning(shared_url, calls, "wikipedia"))
    monkeypatch.setitem(retrieve.PROVIDERS, "gdelt",
                        provider_returning(shared_url, calls, "gdelt"))

    results = retrieve.retrieve_for_entity(
        two_provider_config, "BMW", "company:bmw", 0, connection
    )
    assert len(results) == 1


def test_irrelevant_results_are_dropped_before_any_llm_call(
    two_provider_config, connection, monkeypatch
):
    async def unrelated(client, query, max_results, options):
        return [SearchResult(provider="wikipedia", query=query, title="Museum opening",
                             url="https://x.test/museum",
                             text="The BMW museum in Munich reopened to visitors today.")]

    monkeypatch.setitem(retrieve.PROVIDERS, "wikipedia", unrelated)
    two_provider_config["expansion"]["sources"]["gdelt"]["enabled"] = False

    assert retrieve.retrieve_for_entity(
        two_provider_config, "BMW", "company:bmw", 0, connection
    ) == []


def test_a_failing_provider_is_recorded_and_retried_later(
    two_provider_config, connection, monkeypatch
):
    async def broken(client, query, max_results, options):
        raise RuntimeError("provider exploded")

    calls: list[str] = []
    monkeypatch.setitem(retrieve.PROVIDERS, "wikipedia", broken)
    monkeypatch.setitem(retrieve.PROVIDERS, "gdelt",
                        provider_returning("https://b.test/2", calls, "gdelt"))

    results = retrieve.retrieve_for_entity(
        two_provider_config, "BMW", "company:bmw", 0, connection
    )
    assert len(results) == 1                       # the healthy provider still returned
    sid = state.search_id("company:bmw", 0, "wikipedia", '"BMW" suppliers')
    assert state.search_completed(connection, sid) is False   # so it will be retried


def test_disabled_providers_are_not_called(two_provider_config, connection, monkeypatch):
    calls: list[str] = []
    monkeypatch.setitem(retrieve.PROVIDERS, "wikipedia",
                        provider_returning("https://a.test/1", calls, "wikipedia"))
    monkeypatch.setitem(retrieve.PROVIDERS, "gdelt",
                        provider_returning("https://b.test/2", calls, "gdelt"))
    two_provider_config["expansion"]["sources"]["gdelt"]["enabled"] = False

    retrieve.retrieve_for_entity(two_provider_config, "BMW", "company:bmw", 0, connection)
    assert calls == ["wikipedia"]


def test_unknown_provider_in_config_is_skipped_with_a_warning(
    two_provider_config, connection, caplog
):
    two_provider_config["expansion"]["search"]["defaults"]["providers"] = ["not_a_provider"]
    assert retrieve.retrieve_for_entity(
        two_provider_config, "BMW", "company:bmw", 0, connection
    ) == []
    assert "unknown provider" in caplog.text
