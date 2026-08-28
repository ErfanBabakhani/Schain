"""Run the configured providers for one frontier entity.

Everything that is common to the providers happens here exactly once: per-query
checkpointing, bounded concurrency, optional full-text fetching, persistence,
deduplication and the cheap relevance filter that runs before any LLM call.

A query that already completed in an earlier run is not re-issued; its results are read
back from the search-results Parquet layer instead.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pandas as pd

from .. import state
from ..config import resolve_tier_queries, storage_path
from ..names import normalization_key
from . import PROVIDERS
from .base import TITLE_SNIPPET, SearchResult, build_client, fetch_article_text

logger = logging.getLogger(__name__)


# --- relevance ---------------------------------------------------------------------


def mentions_entity(text: str, entity_name: str) -> bool:
    """True when the normalized entity name appears in the normalized text."""
    return normalization_key(entity_name) in normalization_key(text)


def is_relevant(text: str, entity_name: str, relevance: dict[str, Any]) -> bool:
    """Cheap pre-LLM filter: long enough, mentions the entity, mentions supply language.

    Deliberately not a ranking model - it only drops results that obviously cannot
    contain a supply-chain fact about this entity.
    """
    if len(text) < relevance.get("min_chars", 0):
        return False
    if relevance.get("require_entity_mention", True) and not mentions_entity(text, entity_name):
        return False
    keywords = relevance.get("keywords") or []
    if keywords:
        lowered = text.casefold()
        return any(keyword.casefold() in lowered for keyword in keywords)
    return True


# --- persistence -------------------------------------------------------------------


def append_search_results(results: list[SearchResult], path) -> None:
    if not results:
        return
    frame = pd.DataFrame([result.as_row() for result in results])
    if path.exists():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=["search_id", "url"], keep="last")
    frame.reset_index(drop=True).to_parquet(path, index=False)


def load_search_results(path, search_ids: list[str]) -> list[SearchResult]:
    """Read back results of already-completed queries, so retrieval is not repeated."""
    if not search_ids or not path.exists():
        return []
    frame = pd.read_parquet(path)
    frame = frame[frame.search_id.isin(search_ids)]
    fields = set(SearchResult.__dataclass_fields__)
    return [
        SearchResult(**{key: value for key, value in row.items() if key in fields})
        for row in frame.to_dict("records")
    ]


# --- retrieval ---------------------------------------------------------------------


async def _run_query(
    client,
    semaphore: asyncio.Semaphore,
    provider: str,
    query: str,
    max_results: int,
    options: dict[str, Any],
    entity_key: str,
    tier: int,
    sid: str,
) -> list[SearchResult]:
    """One provider, one query, under the concurrency limit."""
    async with semaphore:
        results = await PROVIDERS[provider](client, query, max_results, options)
    for result in results:
        result.entity_key, result.tier, result.search_id = entity_key, tier, sid
    return results


async def _add_full_text(
    client, semaphore: asyncio.Semaphore, results: list[SearchResult], sources: dict[str, Any]
) -> None:
    """Fetch article bodies for results whose provider did not supply text."""

    async def fetch(result: SearchResult) -> None:
        if result.text or not sources.get(result.provider, {}).get("fetch_full_text", False):
            return
        async with semaphore:
            result.text = await fetch_article_text(client, result.url)
        if not result.text:
            # Expected for Google News, whose RSS links are redirect wrappers. The
            # document is still used, with title and snippet as its text.
            logger.info(
                "no article body for %s (%s); falling back to title and snippet",
                result.url, result.provider,
            )

    await asyncio.gather(*(fetch(result) for result in results))


async def retrieve_async(
    config: dict[str, Any],
    entity_name: str,
    entity_key: str,
    tier: int,
    connection,
) -> list[SearchResult]:
    expansion = config.get("expansion", {})
    sources = expansion.get("sources", {}) or {}
    relevance = expansion.get("relevance", {}) or {}
    resume = config.get("runtime", {}).get("resume", True)
    concurrency = config.get("concurrency", {})
    semaphore = asyncio.Semaphore(max(1, concurrency.get("retrieval", 8)))

    pending: list[tuple] = []
    cached_ids: list[str] = []
    for spec in resolve_tier_queries(config, tier):
        query = spec["query"].format(entity=entity_name)
        for provider in spec.get("providers", []):
            if provider not in PROVIDERS:
                logger.warning("unknown provider %r in config, skipping", provider)
                continue
            options = sources.get(provider, {}) or {}
            if not options.get("enabled", False):
                continue
            sid = state.search_id(entity_key, tier, provider, query)
            if resume and state.search_completed(connection, sid):
                cached_ids.append(sid)
                continue
            pending.append((provider, query, spec["max_results"], options, sid))

    results_path = storage_path(config, "search_results")
    fresh: list[SearchResult] = []

    if pending:
        async with build_client(concurrency.get("http_timeout_seconds", 20)) as client:
            batches = await asyncio.gather(
                *(
                    _run_query(client, semaphore, provider, query, max_results,
                               options, entity_key, tier, sid)
                    for provider, query, max_results, options, sid in pending
                ),
                return_exceptions=True,
            )
            for (provider, query, _, _, sid), batch in zip(pending, batches):
                if isinstance(batch, Exception):
                    logger.warning("%s failed for %s: %s", provider, query, batch)
                    state.record_search(connection, sid, entity_key, tier, provider,
                                        query, state.FAILED, error=str(batch))
                    continue
                fresh.extend(batch)
                state.record_search(connection, sid, entity_key, tier, provider,
                                    query, state.DONE, len(batch))
            await _add_full_text(client, semaphore, fresh, sources)
        append_search_results(fresh, results_path)

    combined = load_search_results(results_path, cached_ids) + fresh
    if cached_ids:
        logger.info("%d query result set(s) reused from a previous run", len(cached_ids))

    # Deduplicate the same article surfaced by several providers or queries.
    unique: dict[str, SearchResult] = {}
    for result in combined:
        unique.setdefault(result.url, result)

    relevant = [
        result for result in unique.values()
        if is_relevant(result.document_text(), entity_name, relevance)
    ]
    fallbacks = sum(1 for result in relevant if result.content_source == TITLE_SNIPPET)
    logger.info(
        "entity %s tier %d: %d candidate(s), %d unique, %d relevant "
        "(%d full text, %d title+snippet)",
        entity_name, tier, len(combined), len(unique), len(relevant),
        len(relevant) - fallbacks, fallbacks,
    )
    return relevant


def retrieve_for_entity(
    config: dict[str, Any],
    entity_name: str,
    entity_key: str,
    tier: int,
    connection,
) -> list[SearchResult]:
    """Blocking wrapper around :func:`retrieve_async`."""
    return asyncio.run(retrieve_async(config, entity_name, entity_key, tier, connection))
