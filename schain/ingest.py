"""INGEST - process documents already in hand.

No search and no graph expansion: the input is a file or a directory, and every document
runs through the shared pipeline. Documents already committed in an earlier run are
skipped, so a resumed run only does what is left.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from . import state
from .documents import iter_input_files, load_document
from .pipeline import process_document

logger = logging.getLogger(__name__)


_DOCUMENT_METRIC_FIELDS = (
    "elapsed_seconds", "llm_calls", "extraction_chunks", "extraction_calls",
    "extraction_cache_hits", "extraction_retries", "candidate_facts",
    "accepted_facts", "rejected_facts",
)


def _accumulate_document_metrics(summary: dict[str, Any], document_summary: dict[str, Any]) -> None:
    for field in _DOCUMENT_METRIC_FIELDS:
        summary[field] += document_summary.get(field, 0) or 0


def run_ingest(
    config: dict[str, Any],
    connection,
    llm,
    ontology: dict[str, Any],
    driver=None,
    input_path: Optional[Path] = None,
    checkpoint_manager=None,
) -> dict[str, Any]:
    """Ingest every input document and return a summary."""
    ingest_config = config.get("ingest", {})
    target = Path(input_path or ingest_config.get("input", "data/corpus"))
    patterns = ingest_config.get("file_patterns", ["*.txt", "*.md", "*.pdf"])

    files = iter_input_files(target, patterns)
    if not files:
        logger.warning("no input files matched %s under %s", patterns, target)
    logger.info("INGEST: %d input file(s) from %s", len(files), target)

    summary = {
        "input": str(target), "files": len(files), "documents_processed": 0,
        "documents_skipped": 0, "documents_failed": 0, "facts": 0,
        "nodes_merged": 0, "relationships_merged": 0,
        "elapsed_seconds": 0.0, "llm_calls": 0, "extraction_chunks": 0,
        "extraction_calls": 0, "extraction_cache_hits": 0, "extraction_retries": 0,
        "candidate_facts": 0, "accepted_facts": 0, "rejected_facts": 0,
    }

    for path in files:
        run_audit = config.get("_audit_trace")
        if run_audit is not None:
            run_audit.emit(
                "input_file_discovered", stage="document_loading",
                path=str(path), filename=path.name,
            )
        try:
            document = load_document(path)
        except (ValueError, OSError) as error:
            logger.error("could not read %s: %s", path, error)
            summary["documents_failed"] += 1
            if run_audit is not None:
                run_audit.emit(
                    "document_load_failed", stage="document_loading",
                    path=str(path), filename=path.name,
                    error_type=type(error).__name__, error=str(error),
                )
            continue

        try:
            document_summary = process_document(
                document, config, connection, llm, ontology, driver
            )
        except Exception as error:  # noqa: BLE001 - already recorded as FAILED in state
            logger.error("document %s failed: %s", path.name, error)
            summary["documents_failed"] += 1
            failed_summary = getattr(error, "_schain_document_summary", None)
            if isinstance(failed_summary, dict):
                _accumulate_document_metrics(summary, failed_summary)
            if checkpoint_manager is not None:
                checkpoint_manager.maybe_snapshot(
                    connection, reason=f"document_failed:{document['doc_id']}",
                )
            continue

        if document_summary["skipped"]:
            summary["documents_skipped"] += 1
            if checkpoint_manager is not None:
                checkpoint_manager.maybe_snapshot(
                    connection, reason=f"document_skipped:{document_summary['doc_id']}",
                )
            continue
        summary["documents_processed"] += 1
        summary["facts"] += document_summary["facts"]
        summary["nodes_merged"] += document_summary["nodes_merged"]
        summary["relationships_merged"] += document_summary["relationships_merged"]
        _accumulate_document_metrics(summary, document_summary)
        if checkpoint_manager is not None:
            checkpoint_manager.maybe_snapshot(
                connection, reason=f"document_completed:{document_summary['doc_id']}",
            )

    summary["documents"] = state.document_counts(connection)
    summary["save"] = {
        "enabled": driver is not None,
        "backend": "neo4j" if driver is not None else None,
        "mode": "incremental" if driver is not None else None,
        "commits": state.neo4j_commit_summary(connection),
    }
    summary["completion"] = {
        "complete": summary["documents_failed"] == 0,
        "export_success": False,
        "pipeline_processing_complete": summary["documents_failed"] == 0,
        "documents_failed": summary["documents_failed"],
        "documents_skipped": summary["documents_skipped"],
        "graph_has_accepted_facts": summary["accepted_facts"] > 0,
        "stop_reason": (
            "document_failures" if summary["documents_failed"] else "ingest_finished"
        ),
    }
    return summary
