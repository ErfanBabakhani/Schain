"""Retrieval providers, parsed against recorded API shapes. No network access."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from schain.sources import gdelt, google_news, wikipedia
from schain.sources.retrieve import is_relevant, mentions_entity

WIKI_SEARCH = {
    "query": {
        "search": [
            {"title": "BMW", "snippet": "<span>BMW</span> suppliers", "timestamp": "2026-01-01T00:00:00Z"},
            {"title": "List of BMW vehicles", "snippet": "vehicles", "timestamp": "2026-01-02T00:00:00Z"},
        ]
    }
}
WIKI_EXTRACTS = {
    "query": {
        "pages": {
            "1": {"title": "BMW", "extract": "BMW sources battery cells from CATL."},
            "2": {"title": "List of BMW vehicles", "extract": "A list of vehicles."},
        }
    }
}
GDELT_RESPONSE = {
    "articles": [
        {"url": "https://news.test/a", "title": "BMW picks supplier",
         "seendate": "20260101T120000Z", "language": "English"},
        {"url": "", "title": "no url, must be dropped"},
    ]
}
GOOGLE_NEWS_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item><title>BMW names battery supplier</title>
        <link>https://news.test/one</link>
        <description>&lt;a href="x"&gt;BMW&lt;/a&gt; picked a supplier</description>
        <pubDate>Tue, 01 Jan 2026 10:00:00 GMT</pubDate></item>
  <item><title>Second story</title><link>https://news.test/two</link>
        <description>more</description></item>
  <item><title>Third story</title><link>https://news.test/three</link></item>
</channel></rss>
"""


def client_for(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)


def run(coroutine):
    return asyncio.run(coroutine)


# --- Wikipedia ---------------------------------------------------------------------


def test_wikipedia_searches_and_returns_article_text():
    def handler(request):
        if request.url.params.get("list") == "search":
            assert request.url.params["srsearch"] == '"BMW" suppliers'
            return httpx.Response(200, json=WIKI_SEARCH)
        assert request.url.params.get("prop") == "extracts"
        return httpx.Response(200, json=WIKI_EXTRACTS)

    async def go():
        async with client_for(handler) as client:
            return await wikipedia.search(client, '"BMW" suppliers', 10, {"language": "en"})

    results = run(go())
    assert [r.title for r in results] == ["BMW", "List of BMW vehicles"]
    assert results[0].text == "BMW sources battery cells from CATL."
    assert results[0].url == "https://en.wikipedia.org/wiki/BMW"
    assert results[0].provider == "wikipedia"


def test_wikipedia_is_a_search_provider_not_a_page_fetcher():
    """The query, not the entity name, decides which articles come back."""
    seen = {}

    def handler(request):
        if request.url.params.get("list") == "search":
            seen["srsearch"] = request.url.params["srsearch"]
            return httpx.Response(200, json=WIKI_SEARCH)
        return httpx.Response(200, json=WIKI_EXTRACTS)

    async def go():
        async with client_for(handler) as client:
            return await wikipedia.search(client, '"BMW" battery suppliers', 5, {})

    results = run(go())
    assert seen["srsearch"] == '"BMW" battery suppliers'
    assert len(results) == 2  # whatever the search returned, not one canonical page


def test_wikipedia_respects_the_api_limit():
    captured = {}

    def handler(request):
        if request.url.params.get("list") == "search":
            captured["srlimit"] = int(request.url.params["srlimit"])
            return httpx.Response(200, json={"query": {"search": []}})
        return httpx.Response(200, json=WIKI_EXTRACTS)

    async def go():
        async with client_for(handler) as client:
            return await wikipedia.search(client, "q", 500, {})

    assert run(go()) == []
    assert captured["srlimit"] == wikipedia.SEARCH_LIMIT


def test_provider_failure_returns_nothing_instead_of_raising():
    def handler(request):
        return httpx.Response(500, text="upstream error")

    async def go():
        async with client_for(handler) as client:
            return await wikipedia.search(client, "q", 5, {})

    assert run(go()) == []


# --- GDELT -------------------------------------------------------------------------


def test_gdelt_parses_articles_and_drops_rows_without_a_url():
    def handler(request):
        assert int(request.url.params["maxrecords"]) == 20
        return httpx.Response(200, json=GDELT_RESPONSE)

    async def go():
        async with client_for(handler) as client:
            return await gdelt.search(client, '"BMW" suppliers', 20, {})

    results = run(go())
    assert [r.url for r in results] == ["https://news.test/a"]
    assert results[0].text is None  # body is fetched later, if enabled


def test_gdelt_caps_maxrecords():
    captured = {}

    def handler(request):
        captured["maxrecords"] = int(request.url.params["maxrecords"])
        return httpx.Response(200, json={"articles": []})

    async def go():
        async with client_for(handler) as client:
            return await gdelt.search(client, "q", 9999, {})

    run(go())
    assert captured["maxrecords"] == gdelt.MAX_RECORDS


# --- Google News -------------------------------------------------------------------


def test_google_news_parses_the_rss_feed():
    def handler(request):
        assert "news.google.com/rss/search" in str(request.url)
        return httpx.Response(200, text=GOOGLE_NEWS_RSS)

    async def go():
        async with client_for(handler) as client:
            return await google_news.search(client, '"BMW" suppliers', 10, {})

    results = run(go())
    assert [r.title for r in results] == [
        "BMW names battery supplier", "Second story", "Third story"
    ]
    assert results[0].snippet == "BMW picked a supplier"   # markup stripped
    assert results[0].published == "Tue, 01 Jan 2026 10:00:00 GMT"


def test_google_news_honours_max_results():
    def handler(request):
        return httpx.Response(200, text=GOOGLE_NEWS_RSS)

    async def go():
        async with client_for(handler) as client:
            return await google_news.search(client, "q", 2, {})

    assert len(run(go())) == 2


def test_malformed_feed_is_not_fatal():
    def handler(request):
        return httpx.Response(200, text="<rss><channel><item>broken")

    async def go():
        async with client_for(handler) as client:
            return await google_news.search(client, "q", 5, {})

    assert run(go()) == []


# --- relevance filter --------------------------------------------------------------


RELEVANCE = {"min_chars": 40, "require_entity_mention": True,
             "keywords": ["supplier", "battery"]}


def test_entity_mention_survives_legal_suffix_differences():
    assert mentions_entity("The BMW plant in Munich", "BMW AG")
    assert not mentions_entity("The Toyota plant in Aichi", "BMW AG")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("BMW has named a new battery supplier for its European plants.", True),
        ("BMW supplier", False),                                    # too short
        ("Toyota has named a new battery supplier for its plants.", False),  # other entity
        ("BMW opened a museum in Munich for visitors this summer.", False),  # no keyword
    ],
)
def test_relevance_filter(text, expected):
    assert is_relevant(text, "BMW", RELEVANCE) is expected


def test_filter_without_keywords_accepts_any_mention():
    assert is_relevant("BMW opened a museum in Munich for visitors this summer.",
                       "BMW", {"min_chars": 10, "require_entity_mention": True})
