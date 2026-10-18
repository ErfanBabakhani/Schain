"""Retrieval orchestration: per-query checkpointing, persistence, deduplication."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from schain import state
from schain.sources import retrieve
from schain.sources.base import SearchResult

ARTICLE = (
    "BMW has named CATL as a battery supplier for its European electric vehicle "
    "plants, under a multi-year sourcing agreement signed this year."
)


@pytest.fixture(autouse=True)
def no_live_canonical_resolution(monkeypatch):
    """These orchestration fixtures do not provide publisher pages."""
    async def unresolved(*args, **kwargs):
        return None
    monkeypatch.setattr(retrieve, "resolve_canonical_url", unresolved)


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


def test_a_document_is_still_returned_when_the_body_cannot_be_fetched(
    two_provider_config, connection, monkeypatch, caplog
):
    """A failed article fetch must degrade to title+snippet, not drop the result."""
    async def no_body(client, url):
        return None, 403, "HTTP_403"

    async def headline_only(client, query, max_results, options):
        return [SearchResult(
            provider="google_news", query=query,
            title="BMW names a new battery supplier for European plants",
            url="https://news.google.com/rss/articles/CBMwrapper",
            snippet="BMW has signed a multi-year sourcing agreement with a battery supplier.",
        )]

    caplog.set_level(logging.INFO, logger="schain.sources.retrieve")
    monkeypatch.setattr("schain.sources.retrieve.fetch_article_text", no_body)
    monkeypatch.setitem(retrieve.PROVIDERS, "google_news", headline_only)
    # google_news is the provider whose RSS links are redirect wrappers, so it is the
    # one configured to attempt a body fetch here.
    two_provider_config["expansion"]["sources"] = {
        "google_news": {"enabled": True, "fetch_full_text": True},
    }
    two_provider_config["expansion"]["search"]["defaults"]["providers"] = ["google_news"]

    results = retrieve.retrieve_for_entity(
        two_provider_config, "BMW", "company:bmw", 0, connection
    )
    assert len(results) == 1
    assert results[0].content_source == "title_snippet"
    assert "falling back to title and snippet" in caplog.text


def test_the_fallback_marker_is_persisted_with_the_search_results(
    two_provider_config, connection, monkeypatch
):
    async def headline_only(client, query, max_results, options):
        return [SearchResult(provider="google_news", query=query,
                             title="BMW supplier news for its European plants",
                             url="https://news.test/x",
                             snippet="A multi-year battery supplier agreement was signed.")]

    monkeypatch.setitem(retrieve.PROVIDERS, "wikipedia", headline_only)
    two_provider_config["expansion"]["sources"]["gdelt"]["enabled"] = False
    retrieve.retrieve_for_entity(two_provider_config, "BMW", "company:bmw", 0, connection)

    frame = pd.read_parquet(two_provider_config["storage"]["search_results"])
    assert frame.iloc[0]["content_source"] == "title_snippet"


# --- the general-web provider alongside the others -----------------------------------


@pytest.fixture
def mixed_source_config(expand_config):
    """One query, all four source families, so the mix can be exercised together."""
    expand_config["expansion"]["sources"] = {
        "google_news": {"enabled": True}, "gdelt": {"enabled": True},
        "general_web": {"enabled": True, "backend": "duckduckgo"},
        "wikipedia": {"enabled": True},
    }
    expand_config["expansion"]["search"] = {
        "defaults": {"max_results": 5,
                     "providers": ["google_news", "gdelt", "general_web", "wikipedia"]},
        "tiers": {0: {"queries": [{"query": '"{entity}" suppliers'}]}},
    }
    expand_config["expansion"]["relevance"] = {
        "min_chars": 20, "require_entity_mention": True, "keywords": ["supplier"],
    }
    return expand_config


def test_all_four_source_families_feed_the_same_pipeline(
    mixed_source_config, connection, monkeypatch
):
    """News, general web and reference results are treated identically from here on."""
    calls: list[str] = []
    for name in ("google_news", "gdelt", "general_web", "wikipedia"):
        monkeypatch.setitem(retrieve.PROVIDERS, name,
                            provider_returning(f"https://{name}.test/1", calls, name))

    results = retrieve.retrieve_for_entity(
        mixed_source_config, "BMW", "company:bmw", 0, connection
    )
    assert sorted(calls) == ["gdelt", "general_web", "google_news", "wikipedia"]
    assert {r.provider for r in results} == {"gdelt", "general_web", "google_news", "wikipedia"}


def test_general_web_can_be_disabled_cleanly(mixed_source_config, connection, monkeypatch):
    calls: list[str] = []
    for name in ("google_news", "gdelt", "general_web", "wikipedia"):
        monkeypatch.setitem(retrieve.PROVIDERS, name,
                            provider_returning(f"https://{name}.test/1", calls, name))
    mixed_source_config["expansion"]["sources"]["general_web"]["enabled"] = False

    retrieve.retrieve_for_entity(mixed_source_config, "BMW", "company:bmw", 0, connection)
    assert "general_web" not in calls
    assert sorted(calls) == ["gdelt", "google_news", "wikipedia"]


def test_a_failing_general_web_provider_does_not_block_the_others(
    mixed_source_config, connection, monkeypatch
):
    """The optional general-web source must never take the run down with it."""
    calls: list[str] = []
    for name in ("google_news", "gdelt", "wikipedia"):
        monkeypatch.setitem(retrieve.PROVIDERS, name,
                            provider_returning(f"https://{name}.test/1", calls, name))

    async def broken(client, query, max_results, options):
        raise RuntimeError("general web backend is unavailable")

    monkeypatch.setitem(retrieve.PROVIDERS, "general_web", broken)

    results = retrieve.retrieve_for_entity(
        mixed_source_config, "BMW", "company:bmw", 0, connection
    )
    assert {r.provider for r in results} == {"gdelt", "google_news", "wikipedia"}
    sid = state.search_id("company:bmw", 0, "general_web", '"BMW" suppliers')
    assert state.search_completed(connection, sid) is False   # so it is retried later


def test_per_provider_limits_reach_the_providers(mixed_source_config, connection, monkeypatch):
    """Wikipedia gets a small ceiling while news providers get a large one."""
    seen: dict[str, int] = {}

    def recorder(name):
        async def search(client, query, max_results, options):
            seen[name] = max_results
            return []
        return search

    for name in ("google_news", "gdelt", "general_web", "wikipedia"):
        monkeypatch.setitem(retrieve.PROVIDERS, name, recorder(name))
    mixed_source_config["expansion"]["search"]["defaults"]["max_results_by_provider"] = {
        "wikipedia": 10, "general_web": 30,
    }
    mixed_source_config["expansion"]["search"]["defaults"]["max_results"] = 50

    retrieve.retrieve_for_entity(mixed_source_config, "BMW", "company:bmw", 0, connection)
    assert seen == {"google_news": 50, "gdelt": 50, "general_web": 30, "wikipedia": 10}
