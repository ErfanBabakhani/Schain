"""Google News retrieval through its public RSS search feed.

Google News has no official public API. The RSS search endpoint is the conventional
stable way to query it and needs no credentials, so that is what V1 uses. The feed gives
title, link, publication date and a short description; the article body is fetched later
by retrieve.py when full-text fetching is enabled.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from xml.etree import ElementTree

import httpx

from .base import SearchResult, get_text

logger = logging.getLogger(__name__)

FEED_URL = "https://news.google.com/rss/search"
_TAG_PATTERN = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    """RSS descriptions carry markup; the LLM should not see tags or the gaps they leave."""
    without_tags = _TAG_PATTERN.sub(" ", text or "").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", without_tags).strip()


async def search(
    client: httpx.AsyncClient,
    query: str,
    max_results: int,
    options: dict[str, Any] | None = None,
) -> list[SearchResult]:
    options = options or {}
    language = options.get("language", "en-US")
    country = language.split("-")[-1] if "-" in language else "US"
    url = (
        f"{FEED_URL}?q={httpx.QueryParams({'q': query})['q']}"
        f"&hl={language}&gl={country}&ceid={country}:{language.split('-')[0]}"
    )

    body = await get_text(client, url)
    if not body:
        return []
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as error:
        logger.warning("google_news: feed did not parse for %s: %s", query, error)
        return []

    results = []
    for item in root.iterfind(".//item")   :
        link = (item.findtext("link") or "").strip()
        if not link:
            continue
        results.append(
            SearchResult(
                provider="google_news",
                query=query,
                title=(item.findtext("title") or "").strip(),
                url=link,
                snippet=strip_html(item.findtext("description") or ""),
                published=(item.findtext("pubDate") or "").strip() or None,
                language=language,
            )
        )
        if len(results) >= max_results:
            break
    logger.info("google_news: %d result(s) for %s", len(results), query)
    return results
