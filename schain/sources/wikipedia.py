"""Wikipedia retrieval through the MediaWiki API.

Wikipedia is treated as a search provider, not as "fetch the entity's own page". A query
such as '"BMW" suppliers' is sent to the search endpoint, and the matching articles are
returned, exactly as the other providers behave.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import SearchResult, get_json

logger = logging.getLogger(__name__)

SEARCH_LIMIT = 50  # MediaWiki caps list=search at 50 per request for anonymous clients
EXTRACT_BATCH = 20  # prop=extracts caps exlimit at 20 titles per request


def api_url(language: str) -> str:
    return f"https://{language}.wikipedia.org/w/api.php"


async def _fetch_extracts(
    client: httpx.AsyncClient, language: str, titles: list[str]
) -> dict[str, str]:
    """Plain-text article bodies for up to EXTRACT_BATCH titles."""
    payload = await get_json(
        client,
        api_url(language),
        {
            "action": "query", "format": "json", "prop": "extracts",
            "explaintext": 1, "exlimit": len(titles), "redirects": 1,
            "titles": "|".join(titles),
        },
    )
    pages = ((payload or {}).get("query", {}) or {}).get("pages", {}) or {}
    return {
        page["title"]: page.get("extract", "")
        for page in pages.values()
        if page.get("title")
    }


async def search(
    client: httpx.AsyncClient,
    query: str,
    max_results: int,
    options: dict[str, Any] | None = None,
) -> list[SearchResult]:
    """Search Wikipedia and return the matching articles with their full plain text."""
    language = (options or {}).get("language", "en")
    payload = await get_json(
        client,
        api_url(language),
        {
            "action": "query", "format": "json", "list": "search",
            "srsearch": query, "srlimit": min(max_results, SEARCH_LIMIT),
        },
    )
    hits = ((payload or {}).get("query", {}) or {}).get("search", []) or []
    if not hits:
        return []

    titles = [hit["title"] for hit in hits]
    extracts: dict[str, str] = {}
    for start in range(0, len(titles), EXTRACT_BATCH):
        extracts.update(await _fetch_extracts(client, language, titles[start : start + EXTRACT_BATCH]))

    results = []
    for hit in hits:
        title = hit["title"]
        results.append(
            SearchResult(
                provider="wikipedia",
                query=query,
                title=title,
                url=f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                snippet=hit.get("snippet", ""),
                text=extracts.get(title) or None,
                published=hit.get("timestamp"),
                language=language,
            )
        )
    logger.info("wikipedia: %d result(s) for %s", len(results), query)
    return results
