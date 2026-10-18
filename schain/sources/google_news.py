"""Google News retrieval through its public RSS search feed.

Google News has no official public API. The RSS search endpoint is the conventional
stable way to query it and needs no credentials, so that is what V1 uses. The feed gives
title, link, publication date and a short description; the article body is fetched later
by retrieve.py when full-text fetching is enabled.
"""

from __future__ import annotations

import logging
import re
from html import unescape
from typing import Any
from xml.etree import ElementTree

import httpx

from .base import SearchResult, get_text

logger = logging.getLogger(__name__)

FEED_URL = "https://news.google.com/rss/search"
_TAG_PATTERN = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    """RSS descriptions carry markup; the LLM should not see tags or the gaps they leave."""
    without_tags = unescape(_TAG_PATTERN.sub(" ", text or "")).replace("\xa0", " ")
    return re.sub(r"\s+", " ", without_tags).strip()


def clean_attribution(text: str, publisher: str | None) -> str:
    """Remove only a known RSS publisher attribution, never a guessed article actor."""
    text = " ".join(str(text or "").split())
    if not publisher:
        return text
    return re.sub(r"\s+[-–—|]\s+" + re.escape(publisher) + r"\s*$", "", text,
                  flags=re.IGNORECASE).strip()


def clean_description(raw: str, publisher: str | None, title: str) -> str:
    """Google RSS repeats its publisher in a final font tag, outside the headline."""
    if publisher:
        raw = re.sub(
            r"<font\b[^>]*>\s*" + re.escape(publisher) + r"\s*</font>\s*$",
            "", raw, flags=re.IGNORECASE,
        )
    text = clean_attribution(strip_html(raw), publisher)
    # Some feeds omit font markup but still append exactly headline + attribution.
    if publisher and text.casefold() == f"{title} {publisher}".casefold():
        return title
    return text


async def search(
    client: httpx.AsyncClient,
    query: str,
    max_results: int,
    options: dict[str, Any] | None = None,
) -> list[SearchResult]:
    options = options or {}
    language = options.get("language", "en-US")
    country = language.split("-")[-1] if "-" in language else "US"
    url = f"{FEED_URL}?{httpx.QueryParams({'q': query, 'hl': language, 'gl': country, 'ceid': country + ':' + language.split('-')[0]})}"

    body = await get_text(client, url)
    if not body:
        return []
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as error:
        logger.warning("google_news: feed did not parse for %s: %s", query, error)
        return []

    results = []
    for item in root.iterfind(".//item"):
        link = (item.findtext("link") or "").strip()
        if not link:
            continue
        source = item.find("source")
        publisher = (source.text or "").strip() if source is not None else None
        publisher_url = source.get("url") if source is not None else None
        raw_title = (item.findtext("title") or "").strip()
        title = clean_attribution(raw_title, publisher)
        results.append(
            SearchResult(
                provider="google_news",
                query=query,
                title=title,
                url=link,
                snippet=clean_description(item.findtext("description") or "", publisher, title),
                publisher_name=publisher or None,
                publisher_url=publisher_url,
                raw_title=raw_title,
                published=(item.findtext("pubDate") or "").strip() or None,
                language=language,
            )
        )
        if len(results) >= max_results:
            break
    logger.info("google_news: %d result(s) for %s", len(results), query)
    return results
