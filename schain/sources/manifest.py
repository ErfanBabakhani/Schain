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
        )
        self.rows[key] = row
        return key

    def update(self, key: tuple[str, str, str], **fields: Any) -> None:
        row = self.rows.get(key)
        if row is not None:
            row.update({k: v for k, v in fields.items() if k in FIELDS})

    def update_by_url(self, url: str, **fields: Any) -> int:
        """Update every row for a URL. Used once a document's fate is known."""
        updated = 0
        for key, row in self.rows.items():
            if key[2] == url:
                row.update({k: v for k, v in fields.items() if k in FIELDS})
                updated += 1
        return updated

    # --- output ---------------------------------------------------------------------

    def flush(self) -> Path:
        """Rewrite the CSV. Rewriting keeps one row per candidate across resumes."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for row in self.rows.values():
                writer.writerow(row)
        return self.path

    def summary(self) -> dict[str, Any]:
        """Counts derived from the rows themselves, so they cannot disagree with the CSV."""
        rows = list(self.rows.values())
        fetch = Counter(row["fetch_status"] for row in rows)
        mode = Counter(row["content_mode"] for row in rows)
        dedup = Counter(row["dedup_status"] for row in rows)
        relevance = Counter(row["relevance_status"] for row in rows)
        llm = Counter(row["llm_status"] for row in rows)
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
            "sent_to_prompt1": llm[PROCESSED] + llm[LLM_FAILED],
            "prompt1_failures": llm[LLM_FAILED],
            "by_provider": dict(Counter(row["provider"] for row in rows)),
            "by_tier": dict(Counter(str(row["tier"]) for row in rows)),
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
        logger.info("  %-28s %s", "manifest", self.path)
