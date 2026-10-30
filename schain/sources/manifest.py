"""Retrieval audit manifest: one CSV row per candidate a provider returned.

Purely for audit and debugging. SQLite and the Parquet layers remain the source of truth
for resume; nothing here is a second checkpoint system, and losing the CSV costs nothing
but the audit trail.

Every candidate gets a row, including ones that were later deduplicated, judged
irrelevant, failed to fetch, or never reached the LLM - the point is to be able to ask
afterwards why a document did or did not become graph facts.

Rows are keyed by (provider, query, url) and rewritten in place, so resuming a run
updates the existing row instead of appending a duplicate.
"""

from __future__ import annotations

import csv
import logging
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

FIELDS = [
    "run_id", "entity", "tier", "provider", "source_type", "query", "result_rank",
    "title", "url", "domain", "published_at", "retrieved_at",
    "snippet_length_chars", "content_length_chars", "content_length_words",
    "fetch_status", "http_status", "content_mode",
    "dedup_status", "relevance_status", "llm_status", "error",
    "canonical_source_url", "publisher_name", "publisher_url",
    "selected", "doc_id", "discovery_direction", "discovery_rank",
    "accepted_expansion_edges", "expansion_edge_keys",
    "content_selection_reason", "fetched_content_length_chars",
    "title_lookup_attempted", "selection_reason", "document_type",
    "escalation_stage",
]

# fetch_status
SUCCESS, FAILED, NOT_ATTEMPTED = "SUCCESS", "FAILED", "NOT_ATTEMPTED"
# content_mode
FULL_TEXT, SNIPPET_ONLY, NONE = "FULL_TEXT", "SNIPPET_ONLY", "NONE"
# dedup_status
UNIQUE, DUPLICATE_URL, DUPLICATE_CONTENT = "UNIQUE", "DUPLICATE_URL", "DUPLICATE_CONTENT"
# relevance_status
RELEVANT, IRRELEVANT, NOT_CHECKED = "RELEVANT", "IRRELEVANT", "NOT_CHECKED"
# llm_status
PROCESSED, SKIPPED, LLM_FAILED, NOT_PROCESSED = (
    "PROCESSED", "SKIPPED", "FAILED", "NOT_PROCESSED",
)


def domain_of(url: str) -> str:
    try:
        return urlsplit(url).netloc.lower()
    except ValueError:
        return ""


def word_count(text: Optional[str]) -> int:
    return len(text.split()) if text else 0


class RetrievalManifest:
    """Collects candidate rows for one EXPAND run and writes them to a CSV."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8", newline="") as handle:
                for existing in csv.DictReader(handle):
                    if existing.get("run_id") == run_id:
                        row = {field: existing.get(field, "") for field in FIELDS}
                        self.rows[self.key_for(row["provider"], row["query"], row["url"])] = row

    # --- recording ------------------------------------------------------------------

    @staticmethod
    def key_for(provider: str, query: str, url: str) -> tuple[str, str, str]:
        return (provider, query, url)

    def record_candidate(
        self,
        entity: str,
        tier: int,
        provider: str,
        source_type: str,
        query: str,
        rank: int,
        result,
    ) -> tuple[str, str, str]:
        """Add (or refresh) the row for one candidate returned by a provider."""
        key = self.key_for(provider, query, result.url)
        row = self.rows.get(key, {field: "" for field in FIELDS})
        row.update(
            run_id=self.run_id, entity=entity, tier=tier, provider=provider,
            source_type=source_type, query=query, result_rank=rank,
            title=(result.title or "").replace("\n", " ").strip(),
            url=result.url, domain=domain_of(result.url),
            published_at=result.published or "",
            retrieved_at=result.retrieved_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            snippet_length_chars=len(result.snippet or ""),
            content_length_chars=len(result.text or ""),
            content_length_words=word_count(result.text),
            fetch_status=row.get("fetch_status") or NOT_ATTEMPTED,
            http_status=row.get("http_status", ""),
            content_mode=FULL_TEXT if result.text else NONE,
            dedup_status=row.get("dedup_status") or UNIQUE,
            relevance_status=row.get("relevance_status") or NOT_CHECKED,
            llm_status=row.get("llm_status") or NOT_PROCESSED,
            error=row.get("error", ""),
            canonical_source_url=result.canonical_source_url or "",
            publisher_name=result.publisher_name or "",
            publisher_url=result.publisher_url or "",
            discovery_direction=result.discovery_direction or "",
            discovery_rank=result.discovery_rank if result.discovery_rank is not None else "",
        )
        self.rows[key] = row
        return key

    def record_content(self, result) -> None:
        """Record fetched/cached content without attributing another LLM call."""
        self.update(
            self.key_for(result.provider, result.query, result.url),
            canonical_source_url=result.canonical_source_url or "",
            domain=domain_of(result.canonical_source_url or result.publisher_url or result.url),
            publisher_name=result.publisher_name or "", publisher_url=result.publisher_url or "",
            fetch_status=(SUCCESS if result.text or result.fetched_text else FAILED) if result.full_text_attempted else NOT_ATTEMPTED,
            http_status=result.fetch_status_code or "",
            content_mode=FULL_TEXT if result.text else (SNIPPET_ONLY if result.title or result.snippet else NONE),
            content_length_chars=len(result.text or ""), content_length_words=word_count(result.text),
            error=result.fetch_error or "",
            content_selection_reason=result.content_selection_reason or "",
            fetched_content_length_chars=len(result.fetched_text or result.text or ""),
            title_lookup_attempted=("true" if result.title_lookup_attempted else ""),
            selection_reason=getattr(result, "selection_reason", None) or "",
            document_type=getattr(result, "document_type", None) or "",
            escalation_stage=getattr(result, "escalation_stage", None) or "",
        )

    def mark_selected(self, result) -> None:
        """Mark membership of the ranked pool (distinct from actually processing it)."""
        self.update(self.key_for(result.provider, result.query, result.url), selected=True,
                    discovery_direction=result.discovery_direction or "",
                    discovery_rank=result.discovery_rank if result.discovery_rank is not None else "")

    def record_document(self, result, doc_id: str, llm_status: str,
                        accepted_expansion_edges=None) -> None:
        """Credit the actual selected provider row, preserving raw candidate audit rows.

        Pass semantic edge keys for exact union counts. Integer values, retained for
        compatibility, are observations and never represented as unique edge counts.
        """
        fields: dict[str, Any] = {"doc_id": doc_id, "llm_status": llm_status}
        if accepted_expansion_edges is not None:
            if isinstance(accepted_expansion_edges, int):
                fields["accepted_expansion_edges"] = accepted_expansion_edges
            else:
                keys = sorted(set(str(k) for k in accepted_expansion_edges))
                fields["accepted_expansion_edges"] = len(keys)
                fields["expansion_edge_keys"] = json.dumps(keys)
        self.update(self.key_for(result.provider, result.query, result.url), **fields)

    def update(self, key: tuple[str, str, str], **fields: Any) -> None:
        row = self.rows.get(key)
        if row is not None:
            if row.get("llm_status") == PROCESSED and fields.get("llm_status") == SKIPPED:
                fields.pop("llm_status")
            row.update({k: v for k, v in fields.items() if k in FIELDS})

    def update_by_url(self, url: str, **fields: Any) -> int:
        """Update every row for a URL. Used once a document's fate is known."""
        updated = 0
        for key, row in self.rows.items():
            if key[2] == url:
                self.update(key, **fields)
                updated += 1
        return updated

    # --- output ---------------------------------------------------------------------

    def flush(self) -> Path:
        """Rewrite the CSV. Rewriting keeps one row per candidate across resumes."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for row in self.rows.values():
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        return self.path

    def summary(self) -> dict[str, Any]:
        """Counts derived from the rows themselves, so they cannot disagree with the CSV."""
        rows = list(self.rows.values())
        fetch = Counter(row["fetch_status"] for row in rows)
        mode = Counter(row["content_mode"] for row in rows)
        dedup = Counter(row["dedup_status"] for row in rows)
        relevance = Counter(row["relevance_status"] for row in rows)
        llm = Counter(row["llm_status"] for row in rows)
        def identity(row):
            return row.get("doc_id") or row.get("canonical_source_url") or row["url"]
        processed = {identity(r) for r in rows if r["llm_status"] in (PROCESSED, LLM_FAILED)}
        provider_metrics: dict[str, Any] = {}
        for provider in sorted({r["provider"] for r in rows}):
            provider_rows = [r for r in rows if r["provider"] == provider]
            used = [r for r in provider_rows if r["llm_status"] in (PROCESSED, LLM_FAILED)]
            keys: set[str] = set()
            observations = 0
            for row in used:
                observations += int(row.get("accepted_expansion_edges") or 0)
                try:
                    keys.update(json.loads(row.get("expansion_edge_keys") or "[]"))
                except (ValueError, TypeError):
                    pass
            provider_metrics[provider] = {
                "raw_candidates": len(provider_rows),
                "ranked_candidate_documents": len({identity(r) for r in provider_rows
                    if str(r.get("selected", "")).casefold() in ("true", "1")}),
                "selected_documents": len({identity(r) for r in used}),
                "full_text_documents": len({identity(r) for r in provider_rows if r["content_mode"] == FULL_TEXT}),
                "selected_full_text_documents": len({identity(r) for r in used if r["content_mode"] == FULL_TEXT}),
                "successful_fetches": len({identity(r) for r in provider_rows if r["fetch_status"] == SUCCESS}),
                "failed_fetches": len({identity(r) for r in provider_rows if r["fetch_status"] == FAILED}),
                "accepted_expansion_edges": len(keys),
                "accepted_expansion_edge_observations": observations,
            }
        return {
            "raw_candidates": len(rows),
            "unique_urls": len({row["url"] for row in rows}),
            "duplicate_urls": dedup[DUPLICATE_URL],
            "successful_fetches": fetch[SUCCESS],
            "failed_fetches": fetch[FAILED],
            "full_text_documents": mode[FULL_TEXT],
            "snippet_only_documents": mode[SNIPPET_ONLY],
            "duplicate_content_documents": dedup[DUPLICATE_CONTENT],
            "relevant_documents": relevance[RELEVANT],
            "irrelevant_documents": relevance[IRRELEVANT],
            "sent_to_prompt1": len(processed),
            "prompt1_failures": len({identity(r) for r in rows if r["llm_status"] == LLM_FAILED}),
            "by_provider": dict(Counter(row["provider"] for row in rows)),
            "by_tier": dict(Counter(str(row["tier"]) for row in rows)),
            "provider_metrics": provider_metrics,
            "fetch_failure_reasons": dict(Counter(
                (r.get("error") or "unknown_fetch_error")
                for r in rows if r["fetch_status"] == FAILED and (r.get("error") or "").strip()
            )),
            "title_lookups_attempted": sum(
                1 for r in rows if str(r.get("title_lookup_attempted", "")).casefold() in ("true", "1")
            ),
            "title_lookup_results": sum(
                1 for r in rows
                if str(r.get("title_lookup_attempted", "")).casefold() in ("true", "1")
                and (r.get("canonical_source_url") or "").strip()
            ),
            "title_lookup_matches": sum(
                1 for r in rows
                if str(r.get("title_lookup_attempted", "")).casefold() in ("true", "1")
                and r.get("content_mode") == FULL_TEXT
            ),
        }

    def log_summary(self) -> None:
        """Print the compact end-of-run retrieval summary."""
        summary = self.summary()
        logger.info("--- retrieval summary ---")
        for field in (
            "raw_candidates", "unique_urls", "duplicate_urls",
            "successful_fetches", "failed_fetches",
            "full_text_documents", "snippet_only_documents",
            "duplicate_content_documents",
            "relevant_documents", "irrelevant_documents",
            "sent_to_prompt1", "prompt1_failures",
        ):
            logger.info("  %-28s %s", field.replace("_", " "), summary[field])
        logger.info("  %-28s %s", "by provider", summary["by_provider"])
        logger.info("  %-28s %s", "by tier", summary["by_tier"])
        for provider, counts in summary["provider_metrics"].items():
            logger.info("  provider=%s %s", provider, counts)
        logger.info("  %-28s %s", "manifest", self.path)
