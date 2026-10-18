"""GDELT retrieval through the public DOC 2.0 API.

Returns news article metadata; the article body is fetched later by retrieve.py when
full-text fetching is enabled. No credentials are required.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlsplit

import httpx

from .base import SearchResult, get_json

logger = logging.getLogger(__name__)

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250  # API ceiling


async def search(
    client: httpx.AsyncClient,
    query: str,
    max_results: int,
    options: dict[str, Any] | None = None,
) -> list[SearchResult]:
    options = options or {}
    params: dict[str, Any] = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "sort": options.get("sort", "DateDesc"),
        "maxrecords": min(max_results, MAX_RECORDS),
    }
    if options.get("timespan"):
        params["timespan"] = options["timespan"]

    payload = await get_json(client, API_URL, params)
    articles = (payload or {}).get("articles", []) or []

    results: list[SearchResult] = []
    for article in articles:
        url = article.get("url") or ""
        if not url:
            continue
        title = (article.get("title") or "").strip()
        # GDELT DOC 2.0 often omits snippets. Use title as a soft snippet so
        # origin/usability filters can evaluate the candidate rather than
        # discarding every title-only hit before full-text fetch.
        host = ""
        try:
            host = (urlsplit(url).hostname or "").lower()
        except ValueError:
            host = ""
        results.append(
            SearchResult(
                provider="gdelt",
                query=query,
                title=title,
                url=url,
                snippet=title,
                published=article.get("seendate"),
                language=article.get("language"),
                source_domain=host or None,
                publisher_name=host.split(".")[0] if host else None,
                publisher_url=f"https://{host}/" if host else None,
                document_type="news",
            )
        )
    logger.info("gdelt: %d result(s) for %s", len(results), query)
    return results
