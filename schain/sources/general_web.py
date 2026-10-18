"""General-web retrieval: ordinary public pages, not news.

News tells us about *new* supply agreements. The stable part of a supply chain lives
elsewhere: company supplier and sourcing pages, press releases, investor-relations pages,
annual and sustainability reports, and regulatory filings. This provider reaches those
through a normal web search, using the same interface as every other provider.

Two backends, chosen by ``backend`` in the source config:

* ``duckduckgo`` (default) - the HTML endpoint, no credentials, nothing to set up.
* ``brave``                - the official Brave Search API, used when
                             BRAVE_SEARCH_API_KEY is set. More reliable for a long run.

Parsing uses lxml, which is already installed as part of trafilatura, so this adds no
dependency. No browser automation and no crawler: one search request per query, exactly
like the other providers.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

from .base import SearchResult, get_json, get_text

logger = logging.getLogger(__name__)

DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_MAX_COUNT = 20  # the API's per-request ceiling
API_KEY_ENV = "BRAVE_SEARCH_API_KEY"


def _clean_link(href: str) -> str:
    """DuckDuckGo wraps results as /l/?uddg=<encoded target>; return the real URL."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parts = urlsplit(href)
    if "duckduckgo.com" in parts.netloc and parts.path.startswith("/l/"):
        target = parse_qs(parts.query).get("uddg", [""])[0]
        return unquote(target)
    return href


def parse_duckduckgo_html(body: str, query: str, max_results: int) -> list[SearchResult]:
    """Pull title, URL and snippet out of a DuckDuckGo HTML results page."""
    from lxml import html as lxml_html

    try:
        document = lxml_html.fromstring(body)
    except Exception as error:  # noqa: BLE001 - an unparseable page is not fatal
        logger.warning("general_web: could not parse the results page: %s", error)
        return []

    results: list[SearchResult] = []
    for node in document.xpath('//div[contains(@class, "result__body")]'):
        links = node.xpath('.//a[contains(@class, "result__a")]')
        if not links:
            continue
        url = _clean_link(links[0].get("href", ""))
        if not url.startswith("http"):
            continue
        snippets = node.xpath('.//a[contains(@class, "result__snippet")]')
        results.append(
            SearchResult(
                provider="general_web",
                query=query,
                title=" ".join(links[0].text_content().split()),
                url=url,
                snippet=" ".join(snippets[0].text_content().split()) if snippets else "",
            )
        )
        if len(results) >= max_results:
            break
    return results


async def _search_duckduckgo(
    client: httpx.AsyncClient, query: str, max_results: int
) -> list[SearchResult]:
    body = await get_text(client, f"{DUCKDUCKGO_URL}?{httpx.QueryParams({'q': query})}")
    if not body:
        return []
    return parse_duckduckgo_html(body, query, max_results)


async def _search_brave(
    client: httpx.AsyncClient, query: str, max_results: int, api_key: str
) -> list[SearchResult]:
    payload = await get_json(
        client, BRAVE_URL,
        {"q": query, "count": min(max_results, BRAVE_MAX_COUNT)},
        headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
    )
    entries = ((payload or {}).get("web", {}) or {}).get("results", []) or []
    return [
        SearchResult(
            provider="general_web",
            query=query,
            title=entry.get("title", ""),
            url=entry.get("url", ""),
            snippet=entry.get("description", ""),
            published=entry.get("page_age") or entry.get("age"),
        )
        for entry in entries[:max_results]
        if entry.get("url")
    ]


async def search(
    client: httpx.AsyncClient,
    query: str,
    max_results: int,
    options: dict[str, Any] | None = None,
) -> list[SearchResult]:
    """Run one general-web query. Returns [] rather than raising if the backend fails."""
    options = options or {}
    backend = options.get("backend", "duckduckgo")
    api_key = os.environ.get(API_KEY_ENV)

    if backend == "brave":
        if not api_key:
            logger.warning(
                "general_web backend is 'brave' but %s is not set; skipping this query. "
                "Set the key in .env or use backend: duckduckgo.", API_KEY_ENV,
            )
            return []
        results = await _search_brave(client, query, max_results, api_key)
    elif backend == "duckduckgo":
        results = await _search_duckduckgo(client, query, max_results)
    else:
        logger.warning("unknown general_web backend %r; skipping this query", backend)
        return []

    logger.info("general_web(%s): %d result(s) for %s", backend, len(results), query)
    return results
