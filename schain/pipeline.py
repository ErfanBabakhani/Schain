"""The per-document path shared by INGEST and EXPAND.

    document -> Prompt 1 -> Prompt 2 -> facts -> Neo4j MERGE

Each stage records its completion in the state database before the next one starts, so a
crash never forces an expensive stage to run twice:

    PENDING    nothing done yet
    EXTRACTED  Prompt 1 finished, its output is on disk
    RESOLVED   Prompt 2 finished, facts are in the Parquet layer
    COMMITTED  the facts are merged into Neo4j

A resumed run re-enters at whatever stage the document stopped at.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import neo4j
from neo4j_graphrag.components.types import Neo4jGraph

import pandas as pd

from . import state
from .config import neo4j_settings, storage_dir, storage_path
from .documents import append_raw_jsonl, upsert_normalized_parquet
from .kg.extract import extract_graph
from .kg.facts import facts_for_document, graph_to_facts, upsert_facts_parquet
from .kg.resolve import resolve_entities
from .kg.writer import (
    ensure_constraints,
    facts_to_graph_rows,
    merge_nodes,
    merge_relationships,
)

logger = logging.getLogger(__name__)


def _extraction_file(config: dict[str, Any], doc_id: str) -> Path:
    return storage_dir(config, "extractions") / f"{doc_id}.json"


def save_extraction(
    config: dict[str, Any], doc_id: str, graph: Neo4jGraph, chunk_index: dict[str, int]
) -> None:
    """Persist Prompt 1 output before anything else can fail."""
    payload = {"graph": graph.model_dump(), "chunk_index": chunk_index}
    _extraction_file(config, doc_id).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def load_extraction(
    config: dict[str, Any], doc_id: str
) -> tuple[Neo4jGraph, dict[str, int]]:
    payload = json.loads(_extraction_file(config, doc_id).read_text(encoding="utf-8"))
    return Neo4jGraph.model_validate(payload["graph"]), payload["chunk_index"]


def _write_resolution_parquet(
    config: dict[str, Any], doc_id: str, resolution: dict[str, dict[str, Any]]
) -> None:
    """Append the entity-resolution results for this document to the Parquet layer."""
    if not resolution:
        return
    frame = pd.DataFrame(
        [{"doc_id": doc_id, "node_id": node_id, **entry} for node_id, entry in resolution.items()]
    )
    frame["aliases"] = frame["aliases"].apply(list)
    path = storage_path(config, "entity_resolution")
    if path.exists():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=["doc_id", "node_id"], keep="last")
    frame.reset_index(drop=True).to_parquet(path, index=False)


def clear_generated_data(config: dict[str, Any]) -> list[str]:
    """Delete checkpoint state and generated data for an explicit fresh run.

    Neo4j is deliberately left alone: the graph writes are idempotent MERGE statements, so
    a rerun converges on the same graph, and dropping a database is not something a
    --restart flag should do silently.
    """
    removed: list[str] = []
    state.reset(storage_path(config, "state_db"))
    removed.append(str(storage_path(config, "state_db")))
    for key in ("raw_documents", "normalized_documents", "extracted_facts",
                "search_results", "entity_resolution"):
        path = storage_path(config, key)
        if path.exists():
            path.unlink()
            removed.append(str(path))
    extractions = storage_dir(config, "extractions")
    for file in extractions.glob("*.json"):
        file.unlink()
        removed.append(str(file))
    return removed


def open_driver(config: dict[str, Any]) -> neo4j.Driver:
    """Open and verify a Neo4j connection using credentials from the environment."""
    settings = neo4j_settings(config)
    driver = neo4j.GraphDatabase.driver(
        settings["uri"], auth=(settings["username"], settings["password"])
    )
    driver.verify_connectivity()
    ensure_constraints(driver, settings["database"])
    return driver


def commit_facts(
    driver: neo4j.Driver,
    config: dict[str, Any],
    facts: pd.DataFrame,
    ontology: dict[str, Any],
    resolution: Optional[dict[str, dict[str, Any]]] = None,
) -> tuple[int, int]:
    """MERGE one document's facts into Neo4j. Idempotent by canonical id."""
    settings = neo4j_settings(config)
    nodes, relationships = facts_to_graph_rows(facts, resolution)
    written_nodes = merge_nodes(
        driver, settings["database"], nodes, ontology["entity_types"], settings["batch_size"]
    )
    written_rels = merge_relationships(
        driver, settings["database"], relationships, ontology["relation_types"],
        settings["batch_size"],
    )
    return written_nodes, written_rels


def process_document(
    document: dict[str, Any],
    config: dict[str, Any],
    connection,
    llm,
    ontology: dict[str, Any],
    driver: Optional[neo4j.Driver] = None,
    origin_entity: Optional[str] = None,
    tier: Optional[int] = None,
) -> dict[str, Any]:
    """Run one document through the pipeline, resuming from wherever it stopped."""
    resume = config.get("runtime", {}).get("resume", True)
    ingest_config = config.get("ingest", {})
    model = config.get("llm", {}).get("model")

    doc_id, is_new = state.register_document(connection, document, origin_entity, tier)
    summary: dict[str, Any] = {
        "doc_id": doc_id, "facts": 0, "nodes_merged": 0,
        "relationships_merged": 0, "status": None, "skipped": None,
    }

    if not is_new and doc_id != document["doc_id"]:
        # Same content hash or normalized URL as a document already known.
        summary["skipped"] = "duplicate"
        summary["status"] = state.document_status(connection, doc_id)
        logger.info("doc %s is a duplicate of known content, skipping", document["doc_id"])
        return summary

    if is_new:
        append_raw_jsonl(document, storage_path(config, "raw_documents"))
        upsert_normalized_parquet(document, storage_path(config, "normalized_documents"))

    status = state.document_status(connection, doc_id) or state.PENDING
    if status == state.COMMITTED and resume:
        summary.update(skipped="already committed", status=status)
        logger.info("doc %s already committed, skipping", doc_id)
        return summary

    graph: Optional[Neo4jGraph] = None
    chunk_index: dict[str, int] = {}
    resolution: dict[str, dict[str, Any]] = {}

    try:
        # --- Prompt 1 ---
        if status in (state.PENDING, state.PROCESSING, state.FAILED):
            state.set_document_status(connection, doc_id, state.PROCESSING, increment_attempts=True)
            graph, chunk_index = extract_graph(
                llm, document, ontology,
                chunk_size=ingest_config.get("chunk_size", 2000),
                chunk_overlap=ingest_config.get("chunk_overlap", 200),
                on_error=ingest_config.get("on_error", "IGNORE"),
                max_concurrency=config.get("concurrency", {}).get("llm_max_concurrency", 4),
                structured_output=config.get("llm", {}).get("structured_output", False),
            )
            save_extraction(config, doc_id, graph, chunk_index)
            state.set_document_status(connection, doc_id, state.EXTRACTED)
            status = state.EXTRACTED

        # --- Prompt 2 and the facts layer ---
        if status == state.EXTRACTED:
            if graph is None:
                graph, chunk_index = load_extraction(config, doc_id)
                logger.info("doc %s: reusing stored Prompt 1 output", doc_id)
            cache = state.load_resolution_cache(connection)
            resolution, new_entries = resolve_entities(
                graph, llm=llm, cache=cache, aliases=ontology.get("aliases")
            )
            if new_entries:
                state.save_resolution(connection, new_entries)
            _write_resolution_parquet(config, doc_id, resolution)
            facts = graph_to_facts(graph, document, chunk_index, resolution, model=model)
            upsert_facts_parquet(facts, storage_path(config, "extracted_facts"))
            state.set_document_status(connection, doc_id, state.RESOLVED)
            status = state.RESOLVED
            summary["facts"] = len(facts)

        # --- Neo4j ---
        if status == state.RESOLVED:
            facts = facts_for_document(storage_path(config, "extracted_facts"), doc_id)
            summary["facts"] = len(facts)
            if driver is None:
                logger.info("doc %s: no Neo4j driver, stopping at RESOLVED", doc_id)
            else:
                nodes, rels = commit_facts(driver, config, facts, ontology, resolution)
                state.set_document_status(connection, doc_id, state.COMMITTED)
                status = state.COMMITTED
                summary.update(nodes_merged=nodes, relationships_merged=rels)

    except Exception as error:  # noqa: BLE001 - record where it failed, then re-raise
        # Keep the furthest completed stage. If Prompt 1 finished and Prompt 2 or the
        # graph write failed, the document stays EXTRACTED/RESOLVED with the error
        # recorded, so the resumed run does not pay for Prompt 1 again. Only a failure
        # before Prompt 1 completed is FAILED.
        keep = status if status in (state.EXTRACTED, state.RESOLVED) else state.FAILED
        state.set_document_status(connection, doc_id, keep, error=str(error))
        logger.error("doc %s failed after %s (kept as %s): %s", doc_id, status, keep, error)
        raise

    summary["status"] = status
    return summary
