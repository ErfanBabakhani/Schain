"""Process local documents through the shared SChain pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from . import state
from .documents import iter_input_files, load_document
from .pipeline import process_document

logger = logging.getLogger(__name__)

_DOCUMENT_METRIC_FIELDS = (
    "elapsed_seconds",
    "llm_calls",
    "extraction_chunks",
    "extraction_calls",
    "extraction_cache_hits",
    "extraction_retries",
    "candidate_facts",
    "accepted_facts",
    "rejected_facts",
)


def _new_summary(target: Path, file_count: int) -> dict[str, Any]:
    return {
        "input": str(target),
        "files": file_count,
        "documents_processed": 0,
        "documents_skipped": 0,
        "documents_failed": 0,
        "facts": 0,
        "nodes_merged": 0,
        "relationships_merged": 0,
        "elapsed_seconds": 0.0,
        "llm_calls": 0,
        "extraction_chunks": 0,
        "extraction_calls": 0,
        "extraction_cache_hits": 0,
        "extraction_retries": 0,
        "candidate_facts": 0,
        "accepted_facts": 0,
        "rejected_facts": 0,
    }


def _accumulate_document_metrics(
    summary: dict[str, Any], document_summary: dict[str, Any]
) -> None:
    for field in _DOCUMENT_METRIC_FIELDS:
        summary[field] += document_summary.get(field, 0) or 0


def _snapshot(checkpoint_manager, connection, reason: str) -> None:
    if checkpoint_manager is not None:
        checkpoint_manager.maybe_snapshot(connection, reason=reason)


def _load_input(path: Path, audit, summary: dict[str, Any]) -> Optional[dict[str, Any]]:
    if audit is not None:
        audit.emit(
            "input_file_discovered",
            stage="document_loading",
            path=str(path),
            filename=path.name,
        )
    try:
        return load_document(path)
    except (ValueError, OSError) as error:
        logger.error("could not read %s: %s", path, error)
        summary["documents_failed"] += 1
        if audit is not None:
            audit.emit(
                "document_load_failed",
                stage="document_loading",
                path=str(path),
                filename=path.name,
                error_type=type(error).__name__,
                error=str(error),
            )
        return None


def _run_document(
    document: dict[str, Any],
    path: Path,
    config: dict[str, Any],
    connection,
    llm,
    ontology: dict[str, Any],
    driver,
    checkpoint_manager,
    summary: dict[str, Any],
) -> Optional[dict[str, Any]]:
    try:
        return process_document(document, config, connection, llm, ontology, driver)
    except Exception as error:  # noqa: BLE001 - pipeline records the FAILED state
        logger.error("document %s failed: %s", path.name, error)
        summary["documents_failed"] += 1
        failed_summary = getattr(error, "_schain_document_summary", None)
        if isinstance(failed_summary, dict):
            _accumulate_document_metrics(summary, failed_summary)
        _snapshot(
            checkpoint_manager,
            connection,
            reason=f"document_failed:{document['doc_id']}",
        )
        return None


def _record_document(
    document_summary: dict[str, Any],
    summary: dict[str, Any],
    connection,
    checkpoint_manager,
) -> None:
    doc_id = document_summary["doc_id"]
    if document_summary["skipped"]:
        summary["documents_skipped"] += 1
        _snapshot(checkpoint_manager, connection, reason=f"document_skipped:{doc_id}")
        return

    summary["documents_processed"] += 1
    for field in ("facts", "nodes_merged", "relationships_merged"):
        summary[field] += document_summary[field]
    _accumulate_document_metrics(summary, document_summary)
    _snapshot(checkpoint_manager, connection, reason=f"document_completed:{doc_id}")


def _finalize_summary(summary: dict[str, Any], connection, driver) -> dict[str, Any]:
    failed = summary["documents_failed"]
    summary["documents"] = state.document_counts(connection)
    summary["save"] = {
        "enabled": driver is not None,
        "backend": "neo4j" if driver is not None else None,
        "mode": "incremental" if driver is not None else None,
        "commits": state.neo4j_commit_summary(connection),
    }
    summary["completion"] = {
        "complete": failed == 0,
        "export_success": False,
        "pipeline_processing_complete": failed == 0,
        "documents_failed": failed,
        "documents_skipped": summary["documents_skipped"],
        "graph_has_accepted_facts": summary["accepted_facts"] > 0,
        "stop_reason": "document_failures" if failed else "ingest_finished",
    }
    return summary


def run_ingest(
    config: dict[str, Any],
    connection,
    llm,
    ontology: dict[str, Any],
    driver=None,
    input_path: Optional[Path] = None,
    checkpoint_manager=None,
) -> dict[str, Any]:
    """Ingest every matching input document and return aggregate run metrics."""
    ingest_config = config.get("ingest", {})
    target = Path(input_path or ingest_config.get("input", "data/corpus"))
    patterns = ingest_config.get("file_patterns", ["*.txt", "*.md", "*.pdf"])
    files = iter_input_files(target, patterns)

    if not files:
        logger.warning("no input files matched %s under %s", patterns, target)
    logger.info("INGEST: %d input file(s) from %s", len(files), target)

    summary = _new_summary(target, len(files))
    audit = config.get("_audit_trace")
    for path in files:
        document = _load_input(path, audit, summary)
        if document is None:
            continue
        document_summary = _run_document(
            document,
            path,
            config,
            connection,
            llm,
            ontology,
            driver,
            checkpoint_manager,
            summary,
        )
        if document_summary is not None:
            _record_document(document_summary, summary, connection, checkpoint_manager)

    return _finalize_summary(summary, connection, driver)
