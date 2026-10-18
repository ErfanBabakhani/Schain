"""Publisher recovery, bounded HTTP spend, durable caches and truthful telemetry; offline."""
from __future__ import annotations

import asyncio
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx
import pandas as pd
import pytest

from schain import state
from schain.documents import make_document
from schain.sources import google_news, general_web, retrieve
from schain.sources.base import SearchResult
from schain.sources import manifest as mf

TITLE = "CATL supplies battery cells to BMW under a new contract"
BODY = TITLE + ". " + "BMW sources components from CATL for vehicle manufacturing. " * 8
WRAPPER = "https://news.google.com/rss/articles/opaque"
ARTICLE_URL = "https://publisher.test/news/bmw-battery-supplier"


def client_for(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_google_publisher_resolved_before_body_fetch(monkeypatch):
    events = []
    def handler(request):
        events.append((request.method, request.url.host))
        if request.url.host == "news.google.com":
            return httpx.Response(302, headers={"Location": ARTICLE_URL})
        return httpx.Response(200)
    async def fetch(client, url):
        events.append(("BODY", url))
        assert url == ARTICLE_URL
        return BODY, 200, None
    monkeypatch.setattr(retrieve, "fetch_article_text", fetch)
    result = SearchResult("google_news", "q", TITLE, WRAPPER, publisher_url="https://publisher.test")
    async def go():
        async with client_for(handler) as client:
            await retrieve._add_full_text(client, asyncio.Semaphore(2), [result],
                                          {"google_news": {"fetch_full_text": True}})
    asyncio.run(go())
    assert result.canonical_source_url == ARTICLE_URL
    assert result.retrieval_url == WRAPPER
    assert result.content_source == "full_text"
    assert events[-1] == ("BODY", ARTICLE_URL)


def test_unresolved_wrapper_keeps_explicit_snippet_without_fetching_wrapper(monkeypatch):
    async def no_resolution(*args, **kwargs):
        return None
    async def forbidden_fetch(*args, **kwargs):
        raise AssertionError("a Google wrapper is not publisher full text")
    monkeypatch.setattr(retrieve, "resolve_canonical_url", no_resolution)
    monkeypatch.setattr(retrieve, "fetch_article_text", forbidden_fetch)
    result = SearchResult("google_news", "q", TITLE, WRAPPER, snippet=TITLE)
    asyncio.run(retrieve._add_full_text(None, asyncio.Semaphore(1), [result],
                                      {"google_news": {"fetch_full_text": True}}))
    assert result.document_text() == TITLE
    assert result.content_source == "title_snippet"
    assert result.fetch_error == "canonical_unresolved"


def test_failed_publisher_fetch_retains_usable_title(monkeypatch):
    async def fail(*args, **kwargs):
        return None, 403, "HTTP_403"
    monkeypatch.setattr(retrieve, "fetch_article_text", fail)
    result = SearchResult("google_news", "q", TITLE, WRAPPER, canonical_source_url=ARTICLE_URL)
    asyncio.run(retrieve._add_full_text(None, asyncio.Semaphore(1), [result],
                                      {"google_news": {"fetch_full_text": True}}))
    assert result.document_text() == TITLE
    assert result.fetch_status_code == 403


def test_head_failure_still_tries_get_and_canonical_metadata():
    def handler(request):
        if request.method == "HEAD":
            raise httpx.ConnectError("HEAD unavailable", request=request)
        return httpx.Response(200, text=f'<html><link rel="canonical" href="{ARTICLE_URL}"></html>')
    async def go():
        async with client_for(handler) as client:
            return await retrieve.resolve_canonical_url(client, WRAPPER, publisher_url="https://publisher.test")
    assert asyncio.run(go()) == ARTICLE_URL


def test_title_recovery_requires_publisher_domain_and_matching_article(monkeypatch):
    queries = []
    async def search(client, query, limit, options):
        queries.append(query)
        return [
            SearchResult("general_web", query, TITLE, "https://wrong.test/article"),
            SearchResult("general_web", query, "Different company story", ARTICLE_URL),
            SearchResult("general_web", query, TITLE, "https://publisher.test/"),
            SearchResult("general_web", query, TITLE, ARTICLE_URL),
        ]
    monkeypatch.setitem(retrieve.PROVIDERS, "general_web", search)
    result = SearchResult("google_news", "q", TITLE, WRAPPER,
                          publisher_name="Publisher", publisher_url="https://publisher.test")
    recovered = asyncio.run(retrieve.recover_publisher_article(None, result, {}))
    assert recovered == ARTICLE_URL
    assert 'site:publisher.test' in queries[0]
    assert f'"{TITLE}"' in queries[0]


def test_google_rss_publisher_metadata_never_becomes_document_actor():
    publisher = "Electric & Hybrid Vehicle Technology International"
    root = ET.Element("rss")
    item = ET.SubElement(ET.SubElement(root, "channel"), "item")
    ET.SubElement(item, "title").text = f"{TITLE} - {publisher}"
    ET.SubElement(item, "link").text = WRAPPER
    ET.SubElement(item, "source", url="https://publisher.test").text = publisher
    ET.SubElement(item, "description").text = f'<a href="{WRAPPER}">{TITLE}</a>&nbsp;<font>{publisher}</font>'
    def handler(request):
        assert request.url.params["q"] == '"A&B" supplier #2'
        return httpx.Response(200, text=ET.tostring(root, encoding="unicode"))
    async def go():
        async with client_for(handler) as client:
            return await google_news.search(client, '"A&B" supplier #2', 1)
    result = asyncio.run(go())[0]
    document = make_document(result.document_text(), **retrieve.document_kwargs_from_result(result))
    assert document["publisher_name"] == publisher
    assert document["publisher_url"] == "https://publisher.test/"
    assert publisher not in document["text"]
    assert document["text"] == TITLE


def test_general_web_query_preserves_ampersand_and_fragment():
    def handler(request):
        assert request.url.params["q"] == '"A&B" supplier #2'
        return httpx.Response(200, text="<html></html>")
    async def go():
        async with client_for(handler) as client:
            return await general_web.search(client, '"A&B" supplier #2', 1)
    assert asyncio.run(go()) == []


def test_dash_in_real_headline_is_not_a_publisher_or_removed_evidence():
    title = "CATL expands factory - BMW purchases battery cells from CATL"
    for mode in ("full_text", "title_snippet", "file"):
        document = make_document(title, title=title, content_source=mode)
        assert document["text"] == title
        assert document["publisher_name"] is None


@pytest.fixture
def config(expand_config):
    expand_config["expansion"]["sources"] = {"gdelt": {"enabled": True, "fetch_full_text": True}}
    expand_config["expansion"]["search"] = {
        "defaults": {"providers": ["gdelt"], "max_results": 8},
        "tiers": {0: {"queries": [{"query": 'supplies "{entity}"'}, {"query": 'supplier to "{entity}"'}]}},
    }
    expand_config["expansion"]["retrieval"]["max_full_text_fetches_per_entity"] = 2
    return expand_config


def test_full_text_fetches_dedup_before_fetch_and_obey_persisted_ceiling(config, connection, monkeypatch):
    calls = []
    searches = []
    async def provider(client, query, limit, options):
        searches.append(query)
        return [SearchResult("gdelt", query, f"{TITLE} number {i}", f"https://publisher.test/{i}")
                for i in range(5)]
    async def fetch(client, url):
        calls.append(url)
        return BODY, 200, None
    monkeypatch.setitem(retrieve.PROVIDERS, "gdelt", provider)
    monkeypatch.setattr(retrieve, "fetch_article_text", fetch)
    first = retrieve.retrieve_for_entity(config, "BMW", "company:bmw", 0, connection)
    assert len(calls) == 2 and len(set(calls)) == 2
    assert first and all(r.content_source == "full_text" for r in first[:2])
    retrieve.retrieve_for_entity(config, "BMW", "company:bmw", 0, connection)
    assert len(calls) == 2  # resume does not refill the fetch ceiling
    assert len(searches) == 2


def test_raw_results_durable_before_search_done(config, connection, monkeypatch):
    async def provider(client, query, limit, options):
        return [SearchResult("gdelt", query, TITLE, ARTICLE_URL, text=BODY)]
    monkeypatch.setitem(retrieve.PROVIDERS, "gdelt", provider)
    original = state.record_search
    def record(*args, **kwargs):
        if args[6] == state.DONE:
            saved = pd.read_parquet(config["storage"]["search_results"])
            assert args[1] in set(saved.search_id)
        return original(*args, **kwargs)
    monkeypatch.setattr(state, "record_search", record)
    assert retrieve.retrieve_for_entity(config, "BMW", "company:bmw", 0, connection)


def test_legacy_done_search_without_cached_rows_is_recovered(config, connection, monkeypatch):
    count = []
    async def provider(client, query, limit, options):
        count.append(query)
        return [SearchResult("gdelt", query, TITLE, ARTICLE_URL, text=BODY)]
    query = 'supplies "BMW"'
    sid = state.search_id("company:bmw", 0, "gdelt", query)
    state.record_search(connection, sid, "company:bmw", 0, "gdelt", query, state.DONE, 1)
    monkeypatch.setitem(retrieve.PROVIDERS, "gdelt", provider)
    assert retrieve.retrieve_for_entity(config, "BMW", "company:bmw", 0, connection)
    assert query in count


def test_nullable_legacy_search_columns_do_not_fake_full_text(tmp_path):
    path = tmp_path / "search.parquet"
    rows = [SearchResult("gdelt", "q", TITLE, ARTICLE_URL, search_id="s").as_row(),
            SearchResult("google_news", "q", TITLE, WRAPPER, search_id="s", text=BODY).as_row()]
    rows[0]["canonical_source_url"] = float("nan")
    pd.DataFrame(rows).to_parquet(path, index=False)
    loaded = retrieve.load_search_results(path, ["s"])
    assert loaded[0].canonical_source_url is None
    assert loaded[0].content_source == "title_snippet"


def test_provider_http_failure_is_not_checkpointed_as_zero_results(config, connection, monkeypatch):
    monkeypatch.setattr(retrieve, "build_client", lambda *args: client_for(lambda request: httpx.Response(503)))
    assert retrieve.retrieve_for_entity(config, "BMW", "company:bmw", 0, connection) == []
    sid = state.search_id("company:bmw", 0, "gdelt", 'supplies "BMW"')
    assert not state.search_completed(connection, sid)


def test_provider_metrics_credit_only_used_provider_and_survive_resume(tmp_path):
    path = tmp_path / "manifest.csv"
    manifest = mf.RetrievalManifest(path, "run")
    results = [SearchResult(provider, "q", TITLE, ARTICLE_URL, text=BODY)
               for provider in ("gdelt", "general_web")]
    for result in results:
        manifest.record_candidate("BMW", 0, result.provider, "NEWS", "q", 1, result)
    manifest.mark_selected(results[0])
    manifest.record_document(results[0], "doc1", mf.PROCESSED, ["edge1", "edge1"])
    manifest.flush()
    reloaded = mf.RetrievalManifest(path, "run")
    reloaded.record_document(results[0], "doc1", mf.SKIPPED)
    summary = reloaded.summary()
    assert summary["raw_candidates"] == 2
    assert summary["sent_to_prompt1"] == 1
    metrics = summary["provider_metrics"]
    assert metrics["gdelt"]["selected_documents"] == 1
    assert metrics["gdelt"]["accepted_expansion_edges"] == 1
    assert metrics["general_web"]["selected_documents"] == 0


def test_unusable_fetched_body_does_not_erase_explicit_clean_headline(config, connection, monkeypatch, tmp_path):
    unusable_body = "Please sign in to continue reading. " + "Our site uses cookies to improve your experience. " * 8
    async def provider(client, query, limit, options):
        return [SearchResult("gdelt", query, TITLE, ARTICLE_URL,
                             snippet=TITLE, publisher_name="Publisher")]
    async def fetch(client, url):
        return unusable_body, 200, None
    monkeypatch.setitem(retrieve.PROVIDERS, "gdelt", provider)
    monkeypatch.setattr(retrieve, "fetch_article_text", fetch)
    manifest = mf.RetrievalManifest(tmp_path / "fallback-manifest.csv", "run")
    results = retrieve.retrieve_for_entity(config, "BMW", "company:bmw", 0, connection, manifest)
    assert len(results) == 1
    result = results[0]
    assert result.document_text() == TITLE
    assert result.content_source == "title_snippet"
    assert result.fetched_text == unusable_body
    assert result.content_selection_reason == "full_text_not_usable"
    row = manifest.rows[manifest.key_for(result.provider, result.query, result.url)]
    assert row["fetch_status"] == mf.SUCCESS
    assert row["content_mode"] == mf.SNIPPET_ONLY
    assert row["fetched_content_length_chars"] == len(unusable_body)
    stored = pd.read_parquet(config["storage"]["search_results"])
    fallback_rows = stored[stored.content_selection_reason == "full_text_not_usable"]
    assert len(fallback_rows) > 0
    assert fallback_rows.iloc[0].fetched_text == unusable_body
