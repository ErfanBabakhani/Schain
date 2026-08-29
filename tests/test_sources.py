"""Retrieval providers, parsed against recorded API shapes. No network access."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from schain.sources import gdelt, general_web, google_news, wikipedia
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


# --- full text versus the title+snippet fallback -----------------------------------


def test_content_source_reports_full_text_when_a_body_was_fetched():
    from schain.sources.base import FULL_TEXT, SearchResult

    result = SearchResult(provider="gdelt", query="q", title="T", url="u",
                          snippet="S", text="A full article body.")
    assert result.content_source == FULL_TEXT
    assert result.document_text() == "A full article body."


def test_content_source_reports_the_fallback_when_no_body_was_available():
    """Google News RSS links are redirect wrappers, so this is the normal case there."""
    from schain.sources.base import TITLE_SNIPPET, SearchResult

    result = SearchResult(provider="google_news", query="q",
                          title="BMW names battery supplier", url="u",
                          snippet="BMW picked a supplier")
    assert result.content_source == TITLE_SNIPPET
    # still a usable document, just a shorter one
    assert result.document_text() == "BMW names battery supplier\nBMW picked a supplier"


def test_provenance_is_carried_into_the_persisted_row():
    from schain.sources.base import SearchResult

    row = SearchResult(provider="google_news", query="q", title="T", url="u").as_row()
    assert row["content_source"] == "title_snippet"


def test_provenance_survives_a_reload_because_it_is_computed():
    """A cached row rehydrates without the column, and the marker is still right."""
    from schain.sources.base import SearchResult

    fields = set(SearchResult.__dataclass_fields__)
    row = {k: v for k, v in SearchResult(
        provider="gdelt", query="q", title="T", url="u", text="body").as_row().items()
        if k in fields}
    assert SearchResult(**row).content_source == "full_text"


def test_article_extraction_dependency_is_actually_importable():
    """Guards a clean install: trafilatura pulls justext, which needs lxml.html.clean.

    That module moved out of lxml, so without the lxml-html-clean pin the import fails and
    every retrieved document silently degrades to title+snippet.
    """
    import trafilatura

    html = "<html><body><article><p>" + "BMW named a battery supplier. " * 6 + "</p></article></body></html>"
    assert trafilatura.extract(html)


# --- general web ---------------------------------------------------------------------


DDG_HTML = """
<div class="result__body">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.bmwgroup.com%2Fsuppliers.html&rut=x">BMW Group Supplier Network</a>
  <a class="result__snippet">Information for BMW Group suppliers and sourcing.</a>
</div>
<div class="result__body">
  <a class="result__a" href="https://example.com/sustainability-report-2026">Sustainability Report 2026</a>
  <a class="result__snippet">Supply chain due diligence and raw materials.</a>
</div>
"""

BRAVE_JSON = {"web": {"results": [
    {"title": "Supplier list", "url": "https://corp.test/suppliers",
     "description": "Our tier-1 suppliers.", "page_age": "2026-01-01"},
    {"title": "No url", "description": "dropped"},
]}}


def test_general_web_returns_ordinary_pages_not_news():
    def handler(request):
        assert "duckduckgo.com" in str(request.url)
        return httpx.Response(200, text=DDG_HTML)

    async def go():
        async with client_for(handler) as client:
            return await general_web.search(client, '"BMW" suppliers', 10, {})

    results = run(go())
    assert [r.url for r in results] == [
        "https://www.bmwgroup.com/suppliers.html",
        "https://example.com/sustainability-report-2026",
    ]
    assert results[0].provider == "general_web"
    assert "sourcing" in results[0].snippet


def test_general_web_unwraps_redirect_links():
    from schain.sources.general_web import _clean_link

    assert _clean_link("//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.test%2Fx&rut=1") == "https://a.test/x"
    assert _clean_link("https://direct.test/page") == "https://direct.test/page"


def test_general_web_honours_the_per_query_limit():
    def handler(request):
        return httpx.Response(200, text=DDG_HTML)

    async def go():
        async with client_for(handler) as client:
            return await general_web.search(client, "q", 1, {})

    assert len(run(go())) == 1


def test_brave_backend_is_used_when_a_key_is_present(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("X-Subscription-Token")
        seen["host"] = request.url.host
        return httpx.Response(200, json=BRAVE_JSON)

    async def go():
        async with client_for(handler) as client:
            return await general_web.search(client, "q", 10, {"backend": "brave"})

    results = run(go())
    assert seen["host"] == "api.search.brave.com"
    assert seen["auth"] == "test-key"
    assert [r.url for r in results] == ["https://corp.test/suppliers"]


def test_brave_backend_without_a_key_skips_instead_of_failing(monkeypatch, caplog):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    def handler(request):
        raise AssertionError("no request should be made without a key")

    async def go():
        async with client_for(handler) as client:
            return await general_web.search(client, "q", 5, {"backend": "brave"})

    assert run(go()) == []
    assert "BRAVE_SEARCH_API_KEY" in caplog.text


def test_unknown_backend_is_skipped_not_fatal(caplog):
    async def go():
        async with client_for(lambda r: httpx.Response(200, text="")) as client:
            return await general_web.search(client, "q", 5, {"backend": "nonsense"})

    assert run(go()) == []
    assert "unknown general_web backend" in caplog.text


def test_an_unparseable_results_page_is_not_fatal():
    def handler(request):
        return httpx.Response(200, text="not html at all <<<")

    async def go():
        async with client_for(handler) as client:
            return await general_web.search(client, "q", 5, {})

    assert run(go()) == []
