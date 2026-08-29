"""The EXPAND loop.

    seeds -> frontier -> tier queries -> retrieval -> dedup/filter
          -> Prompt 1 -> Prompt 2 -> facts -> Neo4j MERGE
          -> upstream companies -> next-tier frontier -> repeat to max_tier

The frontier lives in SQLite, so an interrupted run continues where it stopped instead of
restarting from the seeds.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from datetime import datetime, timezone

from .. import state
from ..config import REPO_ROOT, storage_path
from ..documents import make_document
from ..kg.facts import FACT_COLUMNS
from ..pipeline import process_document
from ..sources import manifest as mf
from ..sources.retrieve import retrieve_for_entity
from .frontier import discover_upstream, seed_frontier

logger = logging.getLogger(__name__)

# If this many documents fail in a row, the run stops instead of grinding through the
# whole frontier with, for example, an unreachable LLM endpoint.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 5


def _facts_for_entity(config: dict[str, Any], connection, entity_key: str) -> pd.DataFrame:
    """Facts extracted from the documents retrieved for this entity."""
    path = storage_path(config, "extracted_facts")
    if not path.exists():
        return pd.DataFrame(columns=FACT_COLUMNS)
    doc_ids = state.document_ids_for_entity(connection, entity_key)
    if not doc_ids:
        return pd.DataFrame(columns=FACT_COLUMNS)
    frame = pd.read_parquet(path)
    return frame[frame.doc_id.isin(doc_ids)].reset_index(drop=True)


def run_expand(
    config: dict[str, Any],
    connection,
    llm,
    ontology: dict[str, Any],
    driver=None,
) -> dict[str, Any]:
    """Work the frontier until it is empty or a configured limit is reached."""
    expansion = config.get("expansion", {})
    runtime = config.get("runtime", {})
    max_tier = expansion.get("max_tier", 3)
    max_documents = expansion.get("max_documents", 0) or 0
    max_failures = runtime.get("max_consecutive_failures", DEFAULT_MAX_CONSECUTIVE_FAILURES)

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    manifest = mf.RetrievalManifest(
        REPO_ROOT / "logs" / f"retrieval_manifest_{run_id}.csv", run_id
    )

    requeued = state.requeue_stale_processing(connection)
    if requeued:
        logger.info("returned %d entit(ies) left PROCESSING by a previous run", requeued)
    seed_frontier(connection, config)

    summary = {
        "entities_expanded": 0, "documents_processed": 0, "documents_skipped": 0,
        "documents_failed": 0, "facts": 0, "nodes_merged": 0,
        "relationships_merged": 0, "discovered": {},
    }
    consecutive_failures = 0

    while True:
        row = state.next_frontier_entity(connection, max_tier)
        if row is None:
            logger.info("frontier is empty within tier <= %d", max_tier)
            break

        entity_name, entity_key, tier = row["entity_name"], row["entity_key"], row["tier"]
        logger.info("--- expanding %s (tier %d) ---", entity_name, tier)

        try:
            results = retrieve_for_entity(
                config, entity_name, entity_key, tier, connection, manifest
            )
        except Exception as error:  # noqa: BLE001 - retrieval trouble must not kill the run
            logger.error("retrieval failed for %s: %s", entity_name, error)
            state.set_frontier_status(connection, entity_key, state.FAILED, str(error))
            continue

        for result in results:
            if max_documents and summary["documents_processed"] >= max_documents:
                logger.info("reached max_documents=%d, stopping", max_documents)
                state.set_frontier_status(connection, entity_key, state.PENDING)
                manifest.flush()
                manifest.log_summary()
                summary["retrieval"] = manifest.summary()
                summary["retrieval_manifest"] = str(manifest.path)
                return summary
            try:
                document = make_document(
                    result.document_text(), source_url=result.url, title=result.title,
                    content_source=result.content_source,
                )
            except ValueError:
                manifest.update_by_url(result.url, llm_status=mf.SKIPPED,
                                       error="EMPTY_CONTENT")
                continue

            # Checked before processing: if this content or URL is already registered,
            # the candidate is a duplicate regardless of why the pipeline skips it.
            already_known = state.find_document(
                connection, document["content_hash"], document.get("source_url")
            ) is not None

            try:
                document_summary = process_document(
                    document, config, connection, llm, ontology, driver,
                    origin_entity=entity_key, tier=tier,
                )
            except Exception as error:  # noqa: BLE001 - already recorded in the state db
                # Prompt 1 may well have succeeded and a later stage failed; the document
                # status says which, so llm_status stays accurate.
                reached = state.document_status(connection, document["doc_id"])
                manifest.update_by_url(
                    result.url,
                    llm_status=mf.PROCESSED if reached in (state.EXTRACTED, state.RESOLVED)
                    else mf.LLM_FAILED,
                    error=str(error)[:200],
                )
                summary["documents_failed"] += 1
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    state.set_frontier_status(connection, entity_key, state.PENDING)
                    raise RuntimeError(
                        f"{consecutive_failures} documents failed in a row; last error: {error}"
                    ) from error
                continue

            consecutive_failures = 0
            if document_summary["skipped"]:
                manifest.update_by_url(
                    result.url,
                    llm_status=mf.SKIPPED,
                    dedup_status=mf.DUPLICATE_CONTENT if already_known else mf.UNIQUE,
                )
                summary["documents_skipped"] += 1
                continue
            manifest.update_by_url(result.url, llm_status=mf.PROCESSED)
            summary["documents_processed"] += 1
            summary["facts"] += document_summary["facts"]
            summary["nodes_merged"] += document_summary["nodes_merged"]
            summary["relationships_merged"] += document_summary["relationships_merged"]

        facts = _facts_for_entity(config, connection, entity_key)
        discovered = discover_upstream(
            connection, config, ontology, entity_name, entity_key, tier, facts
        )
        if discovered:
            summary["discovered"][entity_name] = discovered
        state.set_frontier_status(connection, entity_key, state.DONE)
        summary["entities_expanded"] += 1
        manifest.flush()   # audit trail is kept current as the run proceeds

    manifest.flush()
    manifest.log_summary()
    summary["frontier"] = state.frontier_counts(connection)
    summary["frontier_by_tier"] = state.frontier_tier_counts(connection)
    summary["retrieval"] = manifest.summary()
    summary["retrieval_manifest"] = str(manifest.path)
    return summary
