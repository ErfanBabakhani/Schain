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


def run_ingest(
    config: dict[str, Any],
    connection,
    llm,
    ontology: dict[str, Any],
    driver=None,
    input_path: Optional[Path] = None,
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
    }

    for path in files:
        try:
            document = load_document(path)
        except (ValueError, OSError) as error:
            logger.error("could not read %s: %s", path, error)
            summary["documents_failed"] += 1
            continue

        try:
            document_summary = process_document(
                document, config, connection, llm, ontology, driver
            )
        except Exception as error:  # noqa: BLE001 - already recorded as FAILED in state
            logger.error("document %s failed: %s", path.name, error)
            summary["documents_failed"] += 1
            continue

        if document_summary["skipped"]:
            summary["documents_skipped"] += 1
            continue
        summary["documents_processed"] += 1
        summary["facts"] += document_summary["facts"]
        summary["nodes_merged"] += document_summary["nodes_merged"]
        summary["relationships_merged"] += document_summary["relationships_merged"]

    summary["documents"] = state.document_counts(connection)
    return summary
