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

from .. import state
from ..config import storage_path
from ..documents import make_document
from ..kg.facts import FACT_COLUMNS
from ..pipeline import process_document
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
            results = retrieve_for_entity(config, entity_name, entity_key, tier, connection)
        except Exception as error:  # noqa: BLE001 - retrieval trouble must not kill the run
            logger.error("retrieval failed for %s: %s", entity_name, error)
            state.set_frontier_status(connection, entity_key, state.FAILED, str(error))
            continue

        for result in results:
            if max_documents and summary["documents_processed"] >= max_documents:
                logger.info("reached max_documents=%d, stopping", max_documents)
                state.set_frontier_status(connection, entity_key, state.PENDING)
                return summary
            try:
                document = make_document(
                    result.document_text(), source_url=result.url, title=result.title,
                    content_source=result.content_source,
                )
            except ValueError:
                continue

            try:
                document_summary = process_document(
                    document, config, connection, llm, ontology, driver,
                    origin_entity=entity_key, tier=tier,
                )
            except Exception as error:  # noqa: BLE001 - already recorded as FAILED
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
                summary["documents_skipped"] += 1
                continue
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

    summary["frontier"] = state.frontier_counts(connection)
    summary["frontier_by_tier"] = state.frontier_tier_counts(connection)
    return summary
