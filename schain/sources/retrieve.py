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
import re
from difflib import SequenceMatcher
from dataclasses import replace
from typing import Any, Optional
from urllib.parse import urljoin, urlsplit

import pandas as pd

from .. import state
from ..config import query_limit, storage_path
from ..checkpoint import atomic_write_parquet
from ..documents import normalize_url
from ..names import normalization_key
from ..expansion.telemetry import classify_fetch_error
from . import PROVIDERS, SOURCE_TYPES
from . import manifest as mf
from .base import TITLE_SNIPPET, SearchResult, build_client, fetch_article_text, request_error
from .independence import source_group_id
from .origin import classify_origin_relevance, filter_origin_relevant, origin_name_variants, title_near_duplicate_key
from .ranking import (
    has_explicit_relation_language,
    is_usable_discovery_result,
    sort_discovery_results,
    mentions_entity_name,
)
from .reputation import load_source_reputation, normalize_host, score_source
from ..ontology import load_ontology

logger = logging.getLogger(__name__)


def _company_aliases(config: dict[str, Any]) -> dict[str, str]:
    try:
        ontology = load_ontology(config)
    except Exception:  # noqa: BLE001
        return {}
    return dict((ontology.get("aliases") or {}).get("Company") or {})


def diversify_by_title(
    results: list[SearchResult],
    *,
    max_near_duplicates: int = 2,
) -> list[SearchResult]:
    """Prefer provider/query diversity when titles are near-duplicates."""
    counts: dict[str, int] = {}
    out: list[SearchResult] = []
    for result in results:
        key = title_near_duplicate_key(result.title or "")
        if key:
            n = counts.get(key, 0)
            if n >= max_near_duplicates:
                continue
            counts[key] = n + 1
        out.append(result)
    return out


def diversify_by_domain(
    results: list[SearchResult], *, max_per_domain: int,
) -> list[SearchResult]:
    """Bound syndicated/single-site saturation while preserving rank order."""
    if max_per_domain <= 0:
        return list(results)
    counts: dict[str, int] = {}
    kept: list[SearchResult] = []
    for result in results:
        domain = (
            result.source_domain
            or (urlsplit(result.canonical_source_url or result.url or "").hostname or "")
        ).casefold()
        key = domain or f"provider:{result.provider}"
        if counts.get(key, 0) >= max_per_domain:
            continue
        counts[key] = counts.get(key, 0) + 1
        kept.append(result)
    return kept


# --- provenance enrichment ---------------------------------------------------------


def _is_google_wrapper(url: str | None) -> bool:
    host = normalize_host(url) or ""
    return host == "news.google.com" or host.endswith(".news.google.com")


_PUBLISHER_TITLE_SUFFIX = re.compile(
    r"\s*[\|\-–—:]\s*(?:reuters|bloomberg|forbes|yahoo|cnbc|bbc|cnn|"
    r"the\s+guardian|financial\s+times|nikkei|techcrunch|electrek|"
    r"automotive\s+news|reuters\.com|bloomberg\.com).*$",
    re.I,
)


def clean_news_title(title: str) -> str:
    """Strip publisher suffixes and noise so title-lookup queries can match."""
    text = (title or "").strip()
    text = _PUBLISHER_TITLE_SUFFIX.sub("", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _title_similarity(wanted: str, found: str) -> float:
    if not wanted or not found:
        return 0.0
    if wanted == found:
        return 1.0
    return SequenceMatcher(None, wanted, found).ratio()


def _article_url(url: str | None, publisher_url: str | None = None) -> bool:
    if not url or _is_google_wrapper(url):
        return False
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    publisher_host = normalize_host(publisher_url)
    host = normalize_host(url) or ""
    if host == "google.com" or host.endswith(".google.com"):
        return False  # consent/sign-in pages are not recovered publisher articles
    if publisher_host and not (host == publisher_host or host.endswith("." + publisher_host)):
        return False
    return bool(parts.path.strip("/") or parts.query)


async def resolve_canonical_url(client, url: str, timeout: float = 8.0,
                                publisher_url: str | None = None) -> Optional[str]:
    """Resolve an article redirect or explicit canonical metadata; wrappers aren't articles."""
    try:
        response = await client.head(url, follow_redirects=True, timeout=timeout)
        final = str(response.url)
        if response.is_success and final != url and _article_url(final, publisher_url):
            return final
    except Exception:  # noqa: BLE001 - HEAD failures must still permit GET
        pass
    try:
        response = await client.get(url, follow_redirects=True, timeout=timeout)
        response.raise_for_status()
        final = str(response.url)
        if final != url and _article_url(final, publisher_url):
            return final
        from lxml import html
        page = html.fromstring(response.text)
        candidates = page.xpath('//link[@rel="canonical"]/@href | //meta[@property="og:url"]/@content')
        for candidate in candidates:
            absolute = urljoin(final, candidate)
            if _article_url(absolute, publisher_url):
                return absolute
    except Exception:  # noqa: BLE001
        pass
    return None


async def recover_publisher_article(client, result: SearchResult,
                                    options: dict[str, Any]) -> Optional[str]:
    """Bounded title(+publisher) lookup; high similarity required.

    Prefer site-constrained lookup when publisher_url/host is known. Fall back to
    cleaned title + publisher_name search. Soft-quoted queries avoid empty DDG
    results from overly exact long headlines. Never attach arbitrary same-domain pages.
    """
    raw_title = (result.title or "").strip()
    title = clean_news_title(raw_title) or raw_title
    if len(title.split()) < 3:
        result.fetch_error = result.fetch_error or "title_lookup_mismatch"
        return None
    host = normalize_host(result.publisher_url) if result.publisher_url else None
    if not host and result.source_domain and "google." not in (result.source_domain or ""):
        host = normalize_host(result.source_domain)
    if not host and not result.publisher_name:
        result.fetch_error = "publisher_url_missing"
        return None

    # Prefer Brave for recovery when a key is available (more reliable than HTML DDG).
    recovery_options = dict(options or {})
    if not recovery_options.get("backend") or recovery_options.get("backend") == "duckduckgo":
        import os
        if os.environ.get("BRAVE_SEARCH_API_KEY"):
            recovery_options["backend"] = "brave"

    # Build progressively softer queries — exact quotes often return zero on HTML search.
    core_words = " ".join(title.split()[:12])
    queries: list[str] = []
    if host:
        queries.append(f'"{title}" site:{host}')
        queries.append(f"{core_words} site:{host}")
    if result.publisher_name:
        queries.append(f'"{title}" "{result.publisher_name}"')
        queries.append(f"{core_words} {result.publisher_name}")
    if not queries:
        queries.append(f'"{title}"')
        queries.append(core_words)
    # Deduplicate while preserving order.
    queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))

    wanted = title_near_duplicate_key(title)
    publisher_key = normalization_key(result.publisher_name or "")
    saw_results = False
    for query in queries:
        try:
            candidates = await PROVIDERS["general_web"](client, query, 5, recovery_options)
        except Exception:  # noqa: BLE001 - recovery is optional
            continue
        if not candidates:
            continue
        saw_results = True
        for candidate in candidates:
            found = title_near_duplicate_key(clean_news_title(candidate.title or ""))
            ratio = _title_similarity(wanted, found)
            # Cleaned titles often differ by a trailing publisher phrase; 0.82 is still strict.
            if wanted != found and ratio < 0.82:
                result.fetch_error = "title_lookup_mismatch"
                continue
            if host and not _article_url(candidate.url, result.publisher_url or f"https://{host}/"):
                result.fetch_error = "domain_mismatch"
                continue
            if publisher_key and not host:
                blob = normalization_key(
                    f"{candidate.title} {candidate.url} {candidate.snippet}"
                )
                if publisher_key not in blob:
                    result.fetch_error = "domain_mismatch"
                    continue
            return candidate.url
    if not saw_results:
        result.fetch_error = "title_lookup_no_result"
    elif not result.fetch_error:
        result.fetch_error = "title_lookup_mismatch"
    return None


def url_looks_pdf(url: str) -> bool:
    return (url or "").casefold().split("?", 1)[0].endswith(".pdf")


async def enrich_search_results(
    results: list[SearchResult],
    config: dict[str, Any],
    client=None,
) -> list[SearchResult]:
    """Attach informational metadata; URL recovery happens before body retrieval."""
    reputation = load_source_reputation(config)
    owns_client = client is None
    if owns_client:
        client = build_client(config.get("concurrency", {}).get("http_timeout_seconds", 20))
    try:
        for result in results:
            retrieval = result.retrieval_url or result.url
            result.retrieval_url = retrieval
            canonical = result.canonical_source_url
            result.canonical_source_url = canonical
            host = normalize_host(canonical or result.publisher_url or retrieval)
            result.source_domain = host
            scored = score_source(
                url=canonical or retrieval,
                domain=host,
                content_source=result.content_source,
                canonical_resolved=bool(canonical),
                reputation=reputation,
            )
            # Stash scores on the object for document construction (not dataclass fields).
            result._source_known = scored.known  # type: ignore[attr-defined]
            result._source_class = scored.source_class  # type: ignore[attr-defined]
            result._source_base_score = scored.base_score  # type: ignore[attr-defined]
            result._source_effective_score = scored.effective_score  # type: ignore[attr-defined]
            result._source_group_id = source_group_id(  # type: ignore[attr-defined]
                canonical_source_url=canonical,
                retrieval_url=retrieval,
                source_domain=host,
                content_hash=None,
                title=result.title,
                reputation=reputation,
            )
            logger.debug(
                "source reputation: domain=%s class=%s score=%.0f",
                scored.domain, scored.source_class, scored.effective_score,
            )
    finally:
        if owns_client and client is not None:
            await client.aclose()
    return results


def document_kwargs_from_result(result: SearchResult) -> dict:
    """Keyword args for :func:`make_document` from an enriched SearchResult."""
    return {
        "source_url": result.canonical_source_url or result.url,
        "title": result.title,
        "content_source": result.content_source,
        "retrieval_url": result.retrieval_url or result.url,
        "canonical_source_url": result.canonical_source_url,
        "source_domain": result.source_domain,
        "source_known": getattr(result, "_source_known", None),
        "source_class": getattr(result, "_source_class", None),
        "source_base_score": getattr(result, "_source_base_score", None),
        "source_effective_score": getattr(result, "_source_effective_score", None),
        "source_group_id": getattr(result, "_source_group_id", None),
        "provider": result.provider,
        "query": result.query,
        "origin_entity": result.entity_key,
        "tier": result.tier,
        "publisher_name": result.publisher_name,
        "publisher_url": result.publisher_url,
    }


def mentions_entity(text: str, entity_name: str, aliases: Optional[list[str]] = None) -> bool:
    """True when the normalized entity name appears in the normalized text."""
    return mentions_entity_name(text, entity_name, aliases)


def is_relevant(
    text: str,
    entity_name: str,
    relevance: dict[str, Any],
    *,
    content_source: Optional[str] = None,
    aliases: Optional[list[str]] = None,
    policy: Optional[dict] = None,
) -> bool:
    """Cheap pre-LLM filter: long enough, mentions the entity, mentions supply language.

    Title/snippet results may be shorter than ``min_chars`` when they still name the
    entity and contain explicit supplier/supply-agreement language.
    """
    min_chars = int(relevance.get("min_chars", 0) or 0)
    if content_source == TITLE_SNIPPET and has_explicit_relation_language(text, policy=policy):
        min_chars = min(min_chars, int(relevance.get("title_snippet_min_chars", 40)))
    if len(text) < min_chars:
        return False
    if relevance.get("require_entity_mention", True) and not mentions_entity(text, entity_name, aliases):
        return False
    keywords = relevance.get("keywords") or []
    if keywords:
        lowered = text.casefold()
        if any(keyword.casefold() in lowered for keyword in keywords):
            return True
        # Explicit relation phrases cover cases the keyword list might miss.
        return has_explicit_relation_language(text, policy=policy)
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
    atomic_write_parquet(frame.reset_index(drop=True), path)


def load_search_results(path, search_ids: list[str]) -> list[SearchResult]:
    """Read back results of already-completed queries, so retrieval is not repeated."""
    if not search_ids or not path.exists():
        return []
    frame = pd.read_parquet(path)
    frame = frame[frame.search_id.isin(search_ids)]
    fields = set(SearchResult.__dataclass_fields__)
    return [
        SearchResult(**{key: (None if pd.isna(value) else value)
                        for key, value in row.items() if key in fields})
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
    request_error.set(None)
    async with semaphore:
        results = await PROVIDERS[provider](client, query, max_results, options)
    if not results and request_error.get():
        raise RuntimeError(f"{provider} search failed: {request_error.get()}")
    for result in results:
        result.entity_key, result.tier, result.search_id = entity_key, tier, sid
    return results


async def _add_full_text(
    client,
    semaphore: asyncio.Semaphore,
    results: list[SearchResult],
    sources: dict[str, Any],
    manifest: Optional[mf.RetrievalManifest] = None,
    *,
    max_title_lookups: int = 0,
) -> None:
    """Resolve publisher article URLs first; fetch each direct URL at most once.

    When Google News wrapper decoding fails, title+publisher/domain recovery MUST
    run (bounded). Canonical-unresolved must not silently bypass that path.
    Missing publisher metadata does NOT burn title-lookup budget.
    """
    async def resolve(result: SearchResult) -> None:
        result.retrieval_url = result.retrieval_url or result.url
        if not _is_google_wrapper(result.url) or result.canonical_source_url:
            return
        if result.canonical_resolution_attempted:
            return
        result.canonical_resolution_attempted = True
        async with semaphore:
            result.canonical_source_url = await resolve_canonical_url(
                client, result.url, publisher_url=result.publisher_url,
            )

    await asyncio.gather(*(resolve(result) for result in results))
    lookups = 0
    web_options = dict(sources.get("general_web") or {})
    # Title recovery needs the provider even if discovery browsing was disabled.
    web_options.setdefault("enabled", True)
    for result in results:
        if lookups >= max_title_lookups:
            break
        if not _is_google_wrapper(result.url) or result.canonical_source_url or result.full_text_attempted:
            continue
        title = clean_news_title(result.title or "") or (result.title or "").strip()
        has_publisher = bool(
            result.publisher_url
            or result.publisher_name
            or (result.source_domain and "google." not in (result.source_domain or ""))
        )
        if len(title.split()) < 3:
            result.fetch_error = result.fetch_error or "title_lookup_mismatch"
            continue
        if not has_publisher:
            # Do not burn bounded recovery attempts on unusable metadata.
            result.fetch_error = result.fetch_error or "publisher_url_missing"
            continue
        lookups += 1
        result.title_lookup_attempted = True
        async with semaphore:
            recovered = await recover_publisher_article(client, result, web_options)
            if recovered:
                result.canonical_source_url = recovered
            elif not result.fetch_error:
                result.fetch_error = "title_lookup_no_result"

    fetch_tasks: dict[str, asyncio.Task] = {}

    async def fetch_url(url: str):
        async with semaphore:
            return await fetch_article_text(client, url)

    async def fetch(result: SearchResult) -> None:
        if result.text or result.full_text_attempted:
            return
        if not sources.get(result.provider, {}).get("fetch_full_text", False):
            return
        target = result.canonical_source_url or result.url
        if url_looks_pdf(target):
            result.document_type = "pdf"
        # Unresolved Google wrappers: no body GET. Do not mark full_text_attempted
        # so a later run can retry after better recovery / ranking.
        if _is_google_wrapper(target):
            if not result.fetch_error:
                result.fetch_error = classify_fetch_error(
                    "CANONICAL_UNRESOLVED",
                    is_wrapper=True,
                    title_lookup_attempted=bool(result.title_lookup_attempted),
                )
            logger.info(
                "no article body for %s (%s): %s; falling back to title and snippet",
                result.url, result.provider, result.fetch_error,
            )
            return
        result.full_text_attempted = True
        key = normalize_url(target) or target
        if key not in fetch_tasks:
            fetch_tasks[key] = asyncio.create_task(fetch_url(target))
        text, http_status, error = await fetch_tasks[key]
        result.text, result.fetch_status_code, result.fetch_error = text, http_status, error
        if text:
            result.fetch_error = None
        if not text:
            category = classify_fetch_error(
                error,
                http_status=http_status,
                is_wrapper=False,
                title_lookup_attempted=bool(result.title_lookup_attempted),
            )
            result.fetch_error = category
            logger.info(
                "no article body for %s (%s): %s; falling back to title and snippet",
                result.url, result.provider, category,
            )

    await asyncio.gather(*(fetch(result) for result in results))
    if manifest:
        for result in results:
            manifest.record_content(result)


async def retrieve_async(
    config: dict[str, Any],
    entity_name: str,
    entity_key: str,
    tier: int,
    connection,
    manifest: Optional[mf.RetrievalManifest] = None,
    *,
    llm=None,
    escalation_stages: Optional[list[str]] = None,
) -> list[SearchResult]:
    from ..expansion.escalation import STAGE_A, STAGE_B, build_escalation_queries
    from ..kg.relation_evidence import load_relation_evidence

    expansion = config.get("expansion", {})
    sources = expansion.get("sources", {}) or {}
    relevance = relevance_settings(expansion)
    retrieval_cfg = expansion.get("retrieval") or {}
    resume = config.get("runtime", {}).get("resume", True)
    concurrency = config.get("concurrency", {})
    semaphore = asyncio.Semaphore(max(1, concurrency.get("retrieval", 8)))
    aliases = _company_aliases(config)
    for alias in config.get("_distributed_entity_aliases") or []:
        aliases.setdefault(str(alias), entity_name)
    variants = origin_name_variants(entity_name, aliases)
    for alias in config.get("_distributed_entity_aliases") or []:
        if str(alias).casefold() not in {name.casefold() for name in variants}:
            variants.append(str(alias))
    query_variants = list(dict.fromkeys([
        *variants,
        *(str(alias) for alias, canonical in aliases.items()
          if normalization_key(canonical) == normalization_key(entity_name)),
    ]))
    evidence_cfg = (config.get("validation") or {}).get("evidence") or {}
    policy = load_relation_evidence(evidence_cfg.get("policy_file"))

    results_path = storage_path(config, "search_results")
    cached_frame = pd.read_parquet(results_path) if resume and results_path.exists() else pd.DataFrame()
    cached_counts = cached_frame.search_id.value_counts().to_dict() if "search_id" in cached_frame else {}
    pending: list[tuple] = []
    cached_ids: list[str] = []
    seen_sids: set[str] = set()
    stages = escalation_stages or [STAGE_A, STAGE_B]
    query_specs = build_escalation_queries(
        config, entity_name, tier, llm=llm, stages=stages, aliases=query_variants,
    )
    stage_by_query = {
        " ".join((spec.get("query") or "").split()).casefold(): spec.get("escalation_stage")
        for spec in query_specs
    }
    for spec in query_specs:
        query = spec["query"]
        for provider in spec.get("providers", []):
            if provider not in PROVIDERS:
                logger.warning("unknown provider %r in config, skipping", provider)
                continue
            options = sources.get(provider, {}) or {}
            if not options.get("enabled", False):
                continue
            sid = state.search_id(entity_key, tier, provider, query)
            if sid in seen_sids:
                continue
            seen_sids.add(sid)
            search = connection.execute(
                "SELECT status,result_count FROM searches WHERE search_id=?", (sid,),
            ).fetchone() if resume else None
            # Repair a legacy crash between marking a search DONE and saving its rows.
            if search and search["status"] == state.DONE and (
                search["result_count"] == 0 or cached_counts.get(sid, 0) >= search["result_count"]
            ):
                cached_ids.append(sid)
                continue
            pending.append((provider, query, query_limit(spec, provider), options, sid))

    fresh: list[SearchResult] = []
    completed: list[tuple] = []
    async with build_client(concurrency.get("http_timeout_seconds", 20)) as client:
        if pending:
            batches = await asyncio.gather(
                *(_run_query(client, semaphore, provider, query, max_results,
                             options, entity_key, tier, sid)
                  for provider, query, max_results, options, sid in pending),
                return_exceptions=True,
            )
            for (provider, query, _, _, sid), batch in zip(pending, batches):
                if isinstance(batch, Exception):
                    logger.warning("%s failed for %s: %s", provider, query, batch)
                    state.record_search(connection, sid, entity_key, tier, provider,
                                        query, state.FAILED, error=str(batch))
                    continue
                fresh.extend(batch)
                completed.append((provider, query, sid, len({r.url for r in batch})))
            # Durability order matters: result rows first, then completion checkpoints.
            # A crash later can retry enrichment without reissuing successful searches.
            append_search_results(fresh, results_path)
            for provider, query, sid, count in completed:
                state.record_search(connection, sid, entity_key, tier, provider,
                                    query, state.DONE, count)

        cached = load_search_results(results_path, cached_ids)
        combined = cached + fresh
        if cached_ids:
            logger.info("%d query result set(s) reused from a previous run", len(cached_ids))
        for rank, result in enumerate(combined, start=1):
            if manifest:
                manifest.record_candidate(entity_name, tier, result.provider,
                                          SOURCE_TYPES.get(result.provider, ""), result.query,
                                          rank, result)

        # Deduplicate before network enrichment. Prefer supplied full text on ties.
        unique: dict[str, SearchResult] = {}
        duplicates: dict[str, list[SearchResult]] = {}
        for result in combined:
            key = normalize_url(result.canonical_source_url or result.url) or result.url
            duplicates.setdefault(key, []).append(result)
            previous = unique.get(key)
            if previous is None or (result.text and not previous.text):
                unique[key] = result

        preliminary = sort_discovery_results(list(unique.values()), entity_name,
                                             aliases=variants, policy=policy)
        max_fetches = max(0, int(retrieval_cfg.get("max_full_text_fetches_per_entity", 24)))
        max_lookups = max(0, int(retrieval_cfg.get("max_title_lookups_per_entity", 3)))
        prior_lookups = sum(bool(r.title_lookup_attempted) for r in unique.values())
        remaining_lookups = max(0, max_lookups - prior_lookups)
        # Count already-attempted bodies toward the same per-entity network ceiling.
        preliminary_fetch = diversify_by_domain(
            diversify_by_title(preliminary),
            max_per_domain=int(retrieval_cfg.get("max_full_text_fetches_per_domain") or 0),
        )
        fetchable = [r for r in preliminary_fetch if not r.text and
                     sources.get(r.provider, {}).get("fetch_full_text", False)]
        attempted = sum(bool(r.full_text_attempted) for r in unique.values())
        remaining_fetches = max(0, max_fetches - attempted)
        fetchable = [r for r in fetchable if not r.full_text_attempted][:remaining_fetches]
        # Ensure top unresolved Google News wrappers get title recovery even when
        # they sit just outside the body-fetch shortlist (snippet evidence still valid).
        fetchable_ids = {id(r) for r in fetchable}
        for wrapper in preliminary_fetch:
            if remaining_lookups <= 0:
                break
            if id(wrapper) in fetchable_ids:
                continue
            if not _is_google_wrapper(wrapper.url) or wrapper.canonical_source_url or wrapper.text:
                continue
            if wrapper.full_text_attempted:
                continue
            fetchable.append(wrapper)
            fetchable_ids.add(id(wrapper))
            remaining_lookups -= 1
        await _add_full_text(
            client, semaphore, fetchable, sources, manifest,
            max_title_lookups=max(0, max_lookups - prior_lookups),
        )
        # Preserve raw provider/query candidates, but share fetched content/provenance
        # for exact URL duplicates without issuing more requests.
        for key, representative in unique.items():
            for result in duplicates[key]:
                if result is not representative:
                    for name in ("text", "canonical_source_url", "canonical_resolution_attempted", "title_lookup_attempted",
                                 "full_text_attempted", "fetch_status_code", "fetch_error"):
                        setattr(result, name, getattr(representative, name))
                    if manifest:
                        manifest.update(manifest.key_for(result.provider, result.query, result.url),
                                        dedup_status=mf.DUPLICATE_URL)
                if manifest:
                    manifest.record_content(result)

        await enrich_search_results(list(unique.values()), config, client=client)
        # Canonical aliases (including direct page + Google wrapper) collapse here.
        canonical_unique: dict[str, SearchResult] = {}
        ranked_full = sort_discovery_results(list(unique.values()), entity_name,
                                            aliases=variants, policy=policy)
        for result in ranked_full:
            key = normalize_url(result.canonical_source_url or result.url) or result.url
            if key in canonical_unique:
                if manifest:
                    manifest.update(manifest.key_for(result.provider, result.query, result.url),
                                    dedup_status=mf.DUPLICATE_URL)
                continue
            canonical_unique[key] = result
        relevant: list[SearchResult] = []
        fallback_results: list[SearchResult] = []
        min_full = int(relevance.get("min_chars", 200) or 200)
        for result in canonical_unique.values():
            keep = is_relevant(result.document_text(), entity_name, relevance,
                               content_source=result.content_source, aliases=variants,
                               policy=policy)
            if result.text:
                body_origin, _ = classify_origin_relevance(
                    result, entity_name, aliases=aliases, policy=policy,
                )
                body_usable = is_usable_discovery_result(
                    result, entity_name, min_chars_full=min_full, aliases=variants, policy=policy,
                )
                if not (keep and body_origin and body_usable):
                    # An HTTP 200/paywall/irrelevant extraction must not erase an
                    # independently usable, attributable headline. Preserve the
                    # fetched body for audit while clearly marking selected content.
                    fallback = replace(result, text=None)
                    fallback_origin, _ = classify_origin_relevance(
                        fallback, entity_name, aliases=aliases, policy=policy,
                    )
                    if (fallback_origin and is_relevant(
                            fallback.document_text(), entity_name, relevance,
                            content_source=TITLE_SNIPPET, aliases=variants, policy=policy,
                        ) and is_usable_discovery_result(
                            fallback, entity_name, aliases=variants, policy=policy,
                        )):
                        result.fetched_text, result.text = result.text, None
                        result.content_selection_reason = "full_text_not_usable"
                        fallback_results.append(result)
                        keep = True
            if manifest:
                manifest.record_content(result)
                manifest.update(manifest.key_for(result.provider, result.query, result.url),
                                relevance_status=mf.RELEVANT if keep else mf.IRRELEVANT)
            if keep:
                relevant.append(result)
        origin_kept, origin_reasons = filter_origin_relevant(
            relevant, entity_name, aliases=aliases, policy=policy,
        )
        if fallback_results:
            await enrich_search_results(fallback_results, config, client=client)
        usable = [r for r in origin_kept if is_usable_discovery_result(
            r, entity_name, min_chars_full=min_full, min_chars_snippet=40,
            aliases=variants, policy=policy,
        )]
        ranked = sort_discovery_results(usable, entity_name, aliases=variants, policy=policy)
        selected = diversify_by_title(ranked)
        selected = diversify_by_domain(
            selected,
            max_per_domain=int(retrieval_cfg.get("max_candidates_per_domain") or 0),
        )
        max_candidates = int(retrieval_cfg.get("max_candidates_per_entity") or 0)
        if max_candidates > 0:
            selected = selected[:max_candidates]
        for result in selected:
            qkey = " ".join((result.query or "").split()).casefold()
            stage = stage_by_query.get(qkey) or "primary"
            result.escalation_stage = stage
            if result.document_type == "pdf" or url_looks_pdf(result.canonical_source_url or result.url or ""):
                result.selection_reason = "PDF/report"
            elif stage in ("llm_planned",):
                result.selection_reason = "LLM-planned"
            elif stage in ("official_pdf",):
                result.selection_reason = "official-source"
            elif stage in ("component",):
                result.selection_reason = "rescue"
            else:
                result.selection_reason = "primary"
        if manifest:
            for result in selected:
                manifest.mark_selected(result)
                manifest.record_content(result)
        # Persist enrichment, failures and ranks too, so resume does not retry every
        # wrapper/full-text failure. Null-safe loading keeps old Parquet rows usable.
        append_search_results(combined, results_path)

    logger.info(
        "entity=%s tier=%s raw=%d deduped=%d relevant=%d origin_relevant=%d "
        "ranked=%d selected=%d full_text_attempts=%d origin_filter=%s",
        entity_name, tier, len(combined), len(canonical_unique), len(relevant),
        len(origin_kept), len(ranked), len(selected), len(fetchable), origin_reasons,
    )
    for index, result in enumerate(selected[:5], start=1):
        logger.info("%d. direction=%s score=%.1f %r", index,
                    getattr(result, "_discovery_direction", None),
                    float(getattr(result, "_discovery_rank", 0.0)), (result.title or "")[:100])
    return selected


def retrieve_for_entity(
    config: dict[str, Any],
    entity_name: str,
    entity_key: str,
    tier: int,
    connection,
    manifest: Optional[mf.RetrievalManifest] = None,
    *,
    llm=None,
    escalation_stages: Optional[list[str]] = None,
) -> list[SearchResult]:
    """Blocking wrapper around :func:`retrieve_async`."""
    return asyncio.run(
        retrieve_async(
            config, entity_name, entity_key, tier, connection, manifest,
            llm=llm, escalation_stages=escalation_stages,
        )
    )
