"""Shared pieces for the retrieval providers.

A provider's only job is to turn one query string into candidate documents. Providers do
no extraction, no deduplication and no filtering - that happens once, in retrieve.py.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "schain-kg/0.1 (supply-chain knowledge graph research pipeline)"

# Values of SearchResult.content_source, recorded as document provenance.
FULL_TEXT = "full_text"
TITLE_SNIPPET = "title_snippet"


@dataclass
class SearchResult:
    """One candidate document returned by a provider."""

    provider: str
    query: str
    title: str
    url: str
    snippet: str = ""
    text: Optional[str] = None      # full document text when the provider supplies it
    published: Optional[str] = None
    language: Optional[str] = None
    entity_key: Optional[str] = None
    tier: Optional[int] = None
    search_id: Optional[str] = None
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def content_source(self) -> str:
        """Whether the document text is a fetched article body or the fallback.

        Computed rather than stored, so it cannot drift from the actual content and is
        still correct for results reloaded from the search-results Parquet layer.
        """
        return FULL_TEXT if self.text else TITLE_SNIPPET

    def as_row(self) -> dict[str, Any]:
        return {**asdict(self), "content_source": self.content_source}

    def document_text(self) -> str:
        """Best available text: the fetched body, otherwise title and snippet.

        The fallback is normal, not an error. Google News RSS links are redirect
        wrappers, so an article body is often unavailable; a title plus snippet is still
        a usable document, and content_source records which one this is.
        """
        if self.text:
            return self.text
        parts = [self.title, self.snippet]
        return "\n".join(part for part in parts if part).strip()


def build_client(timeout: float = 20.0) -> httpx.AsyncClient:
    """One async client shared by all providers in a run."""
    return httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any],
    headers: Optional[dict[str, str]] = None,
) -> Any:
    """GET returning parsed JSON, or None when the endpoint misbehaves.

    Retrieval endpoints fail in ordinary ways (rate limits, HTML error pages). A provider
    returning nothing must not abort a whole expansion run.
    """
    try:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as error:
        logger.warning("request to %s failed: %s", url, error)
        return None


async def get_text(client: httpx.AsyncClient, url: str) -> Optional[str]:
    """GET returning the raw response body, or None on failure."""
    try:
        response = await client.get(url)
        response.raise_for_status()
        return response.text
    except httpx.HTTPError as error:
        logger.warning("request to %s failed: %s", url, error)
        return None


async def fetch_article_text(
    client: httpx.AsyncClient, url: str
) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Fetch a page and extract its article body.

    Returns ``(text, http_status, error)``. The error is a short code the retrieval
    manifest records - HTTP_403, TIMEOUT, PARSE_FAILED, EMPTY_CONTENT - rather than an
    exception type, because it is written to a CSV a human reads later.

    Boilerplate removal is done by trafilatura, the established tool for this; writing an
    HTML-to-text extractor by hand would be reimplementing a solved problem.
    """
    status: Optional[int] = None
    try:
        response = await client.get(url)
        status = response.status_code
        response.raise_for_status()
        html = response.text
    except httpx.TimeoutException:
        logger.debug("timeout fetching %s", url)
        return None, status, "TIMEOUT"
    except httpx.HTTPStatusError as error:
        code = error.response.status_code
        logger.debug("HTTP %s fetching %s", code, url)
        return None, code, f"HTTP_{code}"
    except httpx.HTTPError as error:
        logger.debug("request to %s failed: %s", url, error)
        return None, status, "FETCH_FAILED"

    try:
        import trafilatura
    except ImportError as error:
        # A broken install would otherwise disable full-text extraction everywhere and
        # look like every page simply failing to parse, so say so once and loudly.
        logger.warning(
            "trafilatura is unavailable (%s); all retrieved documents will fall back to "
            "title and snippet. Reinstall the project dependencies.", error,
        )
        return None, status, "PARSE_FAILED"

    try:
        text = trafilatura.extract(html, include_comments=False, include_tables=False)
    except Exception as error:  # noqa: BLE001 - a page that will not parse is not fatal
        logger.debug("article extraction failed for %s: %s", url, error)
        return None, status, "PARSE_FAILED"

    if not text:
        return None, status, "EMPTY_CONTENT"
    return text, status, None
