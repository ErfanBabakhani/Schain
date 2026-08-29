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
from typing import Any, Optional

import pandas as pd

from .. import state
from ..config import query_limit, resolve_tier_queries, storage_path
from ..names import normalization_key
from . import PROVIDERS, SOURCE_TYPES
from . import manifest as mf
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


def relevance_settings(expansion: dict[str, Any]) -> dict[str, Any]:
    """The relevance filter settings, with the optional EV vocabulary folded in.

    The vocabulary only widens this keyword list. It is never combined with entity names
    to generate queries: the fixed tier templates stay the only query mechanism.
    """
    relevance = dict(expansion.get("relevance", {}) or {})
    if relevance.get("include_vocabulary", False):
        keywords = list(relevance.get("keywords") or [])
        for group in (expansion.get("vocabulary") or {}).values():
            keywords.extend(group or [])
        relevance["keywords"] = keywords
    return relevance


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
    client,
    semaphore: asyncio.Semaphore,
    results: list[SearchResult],
    sources: dict[str, Any],
    manifest: Optional[mf.RetrievalManifest] = None,
) -> None:
    """Fetch article bodies for results whose provider did not supply text."""

    def note(result: SearchResult, **fields: Any) -> None:
        if manifest:
            manifest.update(
                manifest.key_for(result.provider, result.query, result.url), **fields
            )

    async def fetch(result: SearchResult) -> None:
        if result.text:
            note(result, fetch_status=mf.NOT_ATTEMPTED, content_mode=mf.FULL_TEXT,
                 content_length_chars=len(result.text),
                 content_length_words=mf.word_count(result.text))
            return
        if not sources.get(result.provider, {}).get("fetch_full_text", False):
            note(result, fetch_status=mf.NOT_ATTEMPTED,
                 content_mode=mf.SNIPPET_ONLY if result.snippet or result.title else mf.NONE)
            return

        async with semaphore:
            text, http_status, error = await fetch_article_text(client, result.url)
        result.text = text
        if text:
            note(result, fetch_status=mf.SUCCESS, http_status=http_status or "",
                 content_mode=mf.FULL_TEXT, content_length_chars=len(text),
                 content_length_words=mf.word_count(text), error="")
            return

        # Expected for Google News, whose RSS links are redirect wrappers. The document
        # is still used, with title and snippet as its text.
        logger.info(
            "no article body for %s (%s): %s; falling back to title and snippet",
            result.url, result.provider, error,
        )
        note(result, fetch_status=mf.FAILED, http_status=http_status or "",
             content_mode=mf.SNIPPET_ONLY if result.snippet or result.title else mf.NONE,
             error=error or "")

    await asyncio.gather(*(fetch(result) for result in results))


async def retrieve_async(
    config: dict[str, Any],
    entity_name: str,
    entity_key: str,
    tier: int,
    connection,
    manifest: Optional[mf.RetrievalManifest] = None,
) -> list[SearchResult]:
    expansion = config.get("expansion", {})
    sources = expansion.get("sources", {}) or {}
    relevance = relevance_settings(expansion)
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
            pending.append((provider, query, query_limit(spec, provider), options, sid))

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
                if manifest:
                    for rank, result in enumerate(batch, start=1):
                        manifest.record_candidate(
                            entity_name, tier, provider,
                            SOURCE_TYPES.get(provider, ""), query, rank, result,
                        )
            await _add_full_text(client, semaphore, fresh, sources, manifest)
        append_search_results(fresh, results_path)

    cached = load_search_results(results_path, cached_ids)
    combined = cached + fresh
    if cached_ids:
        logger.info("%d query result set(s) reused from a previous run", len(cached_ids))
        # Reused results still belong in the audit trail, and re-recording them updates
        # the existing rows rather than adding duplicates.
        if manifest:
            for rank, result in enumerate(cached, start=1):
                manifest.record_candidate(
                    entity_name, tier, result.provider,
                    SOURCE_TYPES.get(result.provider, ""), result.query, rank, result,
                )
                manifest.update(
                    manifest.key_for(result.provider, result.query, result.url),
                    content_mode=mf.FULL_TEXT if result.text else mf.SNIPPET_ONLY,
                )

    # Deduplicate the same article surfaced by several providers or queries.
    unique: dict[str, SearchResult] = {}
    for result in combined:
        if result.url in unique:
            if manifest:
                manifest.update(
                    manifest.key_for(result.provider, result.query, result.url),
                    dedup_status=mf.DUPLICATE_URL,
                )
            continue
        unique[result.url] = result

    relevant: list[SearchResult] = []
    for result in unique.values():
        keep = is_relevant(result.document_text(), entity_name, relevance)
        if manifest:
            manifest.update_by_url(
                result.url,
                relevance_status=mf.RELEVANT if keep else mf.IRRELEVANT,
            )
        if keep:
            relevant.append(result)
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
    manifest: Optional[mf.RetrievalManifest] = None,
) -> list[SearchResult]:
    """Blocking wrapper around :func:`retrieve_async`."""
    return asyncio.run(
        retrieve_async(config, entity_name, entity_key, tier, connection, manifest)
    )
