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
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import neo4j
from neo4j_graphrag.components.types import Neo4jGraph

import pandas as pd

from . import state
from .checkpoint import atomic_write_json, atomic_write_parquet
from .config import neo4j_settings, storage_dir, storage_path
from .documents import append_raw_jsonl, upsert_normalized_parquet
from .kg.extract import extract_graph
from .kg.facts import facts_for_document, graph_to_facts, upsert_facts_parquet
from .kg.resolve import resolve_entities
from .kg.support import refresh_edge_support
from .kg.validate import persist_rejected, validate_facts_frame
from .kg.writer import (
    ensure_constraints,
    facts_to_graph_rows,
    merge_nodes,
    merge_relationships,
)
from .llm.provider import MeasuredLLM

logger = logging.getLogger(__name__)


def _extraction_file(config: dict[str, Any], doc_id: str) -> Path:
    return storage_dir(config, "extractions") / f"{doc_id}.json"


def save_extraction(
    config: dict[str, Any], doc_id: str, graph: Neo4jGraph, chunk_index: dict[str, int],
    metrics: Optional[dict[str, Any]] = None,
) -> None:
    """Persist Prompt 1 output before anything else can fail."""
    payload = {"graph": graph.model_dump(), "chunk_index": chunk_index}
    if metrics is not None:
        payload["extraction_metrics"] = dict(metrics)
    atomic_write_json(_extraction_file(config, doc_id), payload)


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
    atomic_write_parquet(frame.reset_index(drop=True), path)


def _load_resolution(
    config: dict[str, Any], doc_id: str,
) -> dict[str, dict[str, Any]]:
    """Restore alias groups when the process resumes directly at the graph write."""
    path = storage_path(config, "entity_resolution")
    if not path.exists():
        return {}
    frame = pd.read_parquet(path, filters=[("doc_id", "==", doc_id)])
    return {
        row["node_id"]: {
            **row, "aliases": list(row["aliases"]) if row.get("aliases") is not None else [],
        }
        for row in frame.to_dict(orient="records")
    }


def _load_persisted_document(config: dict[str, Any], doc_id: str) -> Optional[dict[str, Any]]:
    normalized = storage_path(config, "normalized_documents")
    if normalized.exists():
        stored = pd.read_parquet(normalized, filters=[("doc_id", "==", doc_id)])
        if not stored.empty:
            return {key: None if pd.isna(value) else value
                    for key, value in stored.iloc[0].to_dict().items()}
    raw = storage_path(config, "raw_documents")
    if raw.exists():
        with raw.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("doc_id") == doc_id:
                        return record
    return None


def _persist_document(
    document: dict[str, Any], config: dict[str, Any], *, is_new: bool, status: str,
) -> dict[str, Any]:
    """Repair registration interruptions and retain the first source attribution.

    The same text may be rediscovered under a different URL before resolution finishes.
    Resuming must keep the original record that produced the saved extraction.
    """
    normalized = storage_path(config, "normalized_documents")
    raw = storage_path(config, "raw_documents")
    stored = pd.DataFrame()
    if not is_new and normalized.exists():
        stored = pd.read_parquet(normalized, filters=[("doc_id", "==", document["doc_id"])])
    if not stored.empty:
        document = {
            key: None if pd.isna(value) else value
            for key, value in stored.iloc[0].to_dict().items()
        }
    elif not is_new and raw.exists():
        with raw.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("doc_id") == document["doc_id"]:
                        document = record
                        break

    if is_new or stored.empty or status in (state.PENDING, state.PROCESSING, state.FAILED):
        append_raw_jsonl(document, raw)
    if stored.empty:
        upsert_normalized_parquet(document, normalized)
    return document


def clear_generated_data(config: dict[str, Any]) -> list[str]:
    """Delete checkpoint state and generated data for an explicit fresh run.

    Neo4j is deliberately left alone: the graph writes are idempotent MERGE statements, so
    a rerun converges on the same graph, and dropping a database is not something a
    --restart flag should do silently.
    """
    removed: list[str] = []
    from .run_config import snapshot_path
    run_snapshot = snapshot_path(config)
    if run_snapshot.exists():
        run_snapshot.unlink()
        removed.append(str(run_snapshot))
    state.reset(storage_path(config, "state_db"))
    removed.append(str(storage_path(config, "state_db")))
    for key in ("raw_documents", "normalized_documents", "extracted_facts",
                "search_results", "entity_resolution", "graph_export",
                "rejected_facts", "edge_support", "verification_evidence"):
        path = storage_path(config, key)
        if path.exists():
            path.unlink()
            removed.append(str(path))
    extractions = storage_dir(config, "extractions")
    for file in extractions.glob("*.json"):
        file.unlink()
        removed.append(str(file))
    for directory in extractions.glob("*.chunks"):
        if directory.is_dir():
            shutil.rmtree(directory)
            removed.append(str(directory))
    metrics_directory = extractions / "metrics"
    if metrics_directory.exists():
        shutil.rmtree(metrics_directory)
        removed.append(str(metrics_directory))
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
    document_index: Optional[int] = None,
    document_budget: Optional[int] = None,
) -> dict[str, Any]:
    """Run one document through the pipeline, resuming from wherever it stopped."""
    started = time.monotonic()
    resume = config.get("runtime", {}).get("resume", True)
    ingest_config = config.get("ingest", {})
    model = config.get("llm", {}).get("model")

    document = dict(document)
    doc_id, is_new = state.register_document(connection, document, origin_entity, tier)
    if origin_entity is not None:
        document["origin_entity"] = origin_entity
    if tier is not None:
        document["tier"] = tier
    summary: dict[str, Any] = {
        "doc_id": doc_id, "facts": 0, "nodes_merged": 0,
        "relationships_merged": 0, "status": None, "skipped": None,
        "elapsed_seconds": 0.0, "llm_calls": 0, "llm_calls_by_stage": {},
        "llm_seconds_by_stage": {}, "stage_seconds": {},
        "extraction_chunks": 0, "extraction_calls": 0,
        "extraction_cache_hits": 0, "extraction_retries": 0,
        "candidate_facts": 0, "accepted_facts": 0, "rejected_facts": 0,
    }

    status = state.document_status(connection, doc_id) or state.PENDING
    if not is_new and doc_id != document["doc_id"]:
        # A changed fetch at the same URL must resume the saved original source when
        # unfinished, never pair new text with an old extraction/checkpoint identity.
        unfinished = status not in (state.RESOLVED, state.COMMITTED) or (
            status == state.RESOLVED and driver is not None
        )
        original = _load_persisted_document(config, doc_id) if unfinished else None
        if original is None or not original.get("text"):
            summary.update(skipped="duplicate", status=status)
            logger.info("doc %s is duplicate of %s, skipping", document["doc_id"], doc_id)
            return summary
        document = original
        logger.info("URL rediscovery: resuming unfinished original doc %s", doc_id)

    if status == state.COMMITTED and resume:
        summary.update(skipped="already committed", status=status)
        logger.info("doc %s already committed, skipping", doc_id)
        return summary
    if status == state.RESOLVED and driver is None and resume:
        summary.update(skipped="already resolved", status=status)
        logger.info("doc %s already resolved without Neo4j, skipping", doc_id)
        return summary

    title = (document.get("title") or "")[:80]
    if document_index is not None and document_budget:
        logger.info(
            "processing document %d/%d title=%r",
            document_index, document_budget, title,
        )
    elif document_index is not None:
        logger.info("processing document %d title=%r", document_index, title)
    else:
        logger.info(
            "processing document %s title=%r content=%s provider=%s",
            doc_id, title,
            document.get("content_source") or "unknown",
            document.get("provider") or "unknown",
        )

    graph: Optional[Neo4jGraph] = None
    chunk_index: dict[str, int] = {}
    resolution: dict[str, dict[str, Any]] = {}
    measured_llm = MeasuredLLM(
        llm, summary, deadline=config.get("_runtime_deadline_monotonic"),
    ) if llm is not None else None
    stage = "persistence"
    stage_started = time.monotonic()
    extraction_metrics: dict[str, Any] = {}

    def advance_stage(name: str) -> None:
        nonlocal stage, stage_started
        now = time.monotonic()
        summary["stage_seconds"][stage] = round(now - stage_started, 4)
        stage, stage_started = name, now
        if measured_llm is not None:
            measured_llm.stage = name

    try:
        document = _persist_document(document, config, is_new=is_new, status=status)
        # --- Prompt 1 ---
        if status in (state.PENDING, state.PROCESSING, state.FAILED):
            advance_stage("extraction")
            state.set_document_status(connection, doc_id, state.PROCESSING, increment_attempts=True)
            origin_name = document.get("origin_entity_name")
            if not origin_name and origin_entity:
                origin_row = connection.execute(
                    "SELECT entity_name FROM frontier WHERE entity_key = ?", (origin_entity,),
                ).fetchone()
                origin_name = origin_row["entity_name"] if origin_row else None
            graph, chunk_index = extract_graph(
                measured_llm, document, ontology,
                chunk_size=ingest_config.get("chunk_size", 2000),
                chunk_overlap=ingest_config.get("chunk_overlap", 200),
                on_error=ingest_config.get("on_error", "RAISE"),
                max_concurrency=config.get("concurrency", {}).get("llm_max_concurrency", 4),
                structured_output=config.get("llm", {}).get("structured_output", False),
                checkpoint_dir=storage_dir(config, "extractions") / f"{doc_id}.chunks",
                metrics=extraction_metrics,
                max_chunks_per_document=ingest_config.get("max_chunks_per_document"),
                max_retries=int(config.get("llm", {}).get("extraction_max_retries", 1)),
                origin_name=origin_name,
                evidence_policy_file=(config.get("validation", {}).get("evidence", {})
                                      .get("policy_file")),
            )
            summary.update(extraction_metrics)
            save_extraction(config, doc_id, graph, chunk_index, extraction_metrics)
            state.set_document_status(connection, doc_id, state.EXTRACTED)
            status = state.EXTRACTED

        # --- Prompt 2 and the facts layer ---
        if status == state.EXTRACTED:
            advance_stage("resolution")
            if graph is None:
                graph, chunk_index = load_extraction(config, doc_id)
                logger.info("doc %s: reusing stored Prompt 1 output", doc_id)
            cache = state.load_resolution_cache(connection)
            resolution, new_entries = resolve_entities(
                graph, llm=measured_llm, cache=cache, aliases=ontology.get("aliases")
            )
            if new_entries:
                state.save_resolution(connection, new_entries)
            _write_resolution_parquet(config, doc_id, resolution)
            facts = graph_to_facts(graph, document, chunk_index, resolution, model=model)
            # Deterministic supplier-list path (complements Prompt 1; still validated).
            try:
                from .kg.list_extract import extract_supplier_list_facts
                origin_aliases = []
                alias_map = (ontology.get("aliases") or {}).get("Company") or {}
                if origin_name:
                    origin_aliases = [origin_name, *alias_map.keys()]
                list_rows = extract_supplier_list_facts(
                    document.get("text") or "",
                    doc_id=doc_id,
                    origin_name=origin_name or "",
                    origin_id=origin_entity,
                    aliases=origin_aliases,
                    source_url=document.get("source_url") or document.get("canonical_source_url"),
                )
                if list_rows:
                    import pandas as pd
                    facts = pd.concat([facts, pd.DataFrame(list_rows)], ignore_index=True)
            except Exception as list_error:  # noqa: BLE001
                logger.warning("structured list extraction skipped: %s", list_error)
            summary["candidate_facts"] = len(facts)
            advance_stage("validation")
            accepted, rejected = validate_facts_frame(
                facts, config=config, ontology=ontology, llm=measured_llm,
                document_text=document.get("text") or "",
            )
            advance_stage("facts_persistence")
            persist_rejected(config, rejected)
            upsert_facts_parquet(accepted, storage_path(config, "extracted_facts"))
            refresh_edge_support(config)
            state.set_document_status(connection, doc_id, state.RESOLVED)
            status = state.RESOLVED
            summary["facts"] = len(accepted)
            summary["accepted_facts"] = len(accepted)
            summary["rejected_facts"] = len(rejected)
            logger.info(
                "document resolved facts=%d rejected=%d",
                len(accepted), len(rejected),
            )
            if not accepted.empty:
                for row in accepted.to_dict(orient="records"):
                    if (
                        row.get("relation") == "suppliesTo"
                        and row.get("subject_type") == "Company"
                        and row.get("object_type") == "Company"
                    ):
                        logger.info(
                            "accepted suppliesTo %s -> %s source=%s provider=%s content=%s",
                            row.get("subject_name"), row.get("object_name"),
                            row.get("source_class") or "unknown",
                            row.get("provider") or "unknown",
                            row.get("content_source") or "unknown",
                        )

        # --- Neo4j ---
        if status == state.RESOLVED:
            advance_stage("commit")
            facts = facts_for_document(storage_path(config, "extracted_facts"), doc_id)
            summary["facts"] = len(facts)
            if driver is None:
                logger.info("doc %s: no Neo4j driver, stopping at RESOLVED", doc_id)
            else:
                if not resolution:
                    resolution = _load_resolution(config, doc_id)
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
        summary.update(status=keep, error=str(error))
        raise
    finally:
        summary.update(extraction_metrics)
        if summary["status"] is None:
            summary["status"] = status
        summary["elapsed_seconds"] = round(time.monotonic() - started, 4)
        summary["stage_seconds"][stage] = round(time.monotonic() - stage_started, 4)
        metrics_path = storage_dir(config, "extractions") / "metrics" / f"{doc_id}.json"
        try:
            previous = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            atomic_write_json(metrics_path, {
                "doc_id": doc_id, "provider": document.get("provider"),
                "last_attempt": summary,
                "extraction_metrics": extraction_metrics or previous.get("extraction_metrics", {}),
                "total_elapsed_seconds": round(
                    float(previous.get("total_elapsed_seconds", 0)) + summary["elapsed_seconds"], 4),
                "total_llm_calls": int(previous.get("total_llm_calls", 0)) + summary["llm_calls"],
            })
        except (ValueError, OSError):
            logger.exception("could not persist document metrics for %s", doc_id)
        logger.info("document metrics doc=%s elapsed=%.2fs llm_calls=%d chunks=%d "
                    "cached=%d candidate_facts=%d accepted=%d rejected=%d",
                    doc_id, summary["elapsed_seconds"], summary["llm_calls"],
                    summary["extraction_chunks"], summary["extraction_cache_hits"],
                    summary["candidate_facts"], summary["accepted_facts"], summary["rejected_facts"])

    summary["status"] = status
    return summary
