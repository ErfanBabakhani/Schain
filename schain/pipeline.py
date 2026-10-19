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
from .audit import audit_for_document
from .checkpoint import atomic_write_json, atomic_write_parquet
from .config import neo4j_settings, storage_dir, storage_path
from .documents import append_raw_jsonl, upsert_normalized_parquet
from .kg.extract import extract_graph
from .kg.facts import (
    candidate_fact_id_for_audit, extraction_relationship_id, facts_for_document, graph_to_facts,
    upsert_facts_parquet,
)
from .kg.integrity import enforce_referential_integrity
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
    integrity_failures: Optional[list[dict[str, Any]]] = None,
) -> None:
    """Persist Prompt 1 output before anything else can fail."""
    payload = {"graph": graph.model_dump(), "chunk_index": chunk_index}
    if metrics is not None:
        payload["extraction_metrics"] = dict(metrics)
    if integrity_failures:
        payload["integrity_failures"] = list(integrity_failures)
    atomic_write_json(_extraction_file(config, doc_id), payload)


def _persist_integrity_failures(
    config: dict[str, Any], failures: list[dict[str, Any]],
) -> None:
    """Keep dangling-edge payloads in a separate, idempotent diagnostic artifact."""
    if not failures:
        return
    storage = config.get("storage") or {}
    path = (
        storage_path(config, "extraction_integrity")
        if storage.get("extraction_integrity")
        else storage_dir(config, "extractions").parent / "extraction_integrity.json"
    )
    previous: list[dict[str, Any]] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                previous = loaded
        except (OSError, ValueError):
            logger.warning("replacing damaged extraction-integrity artifact %s", path)
    by_id = {
        row.get("integrity_failure_id"): row
        for row in [*previous, *failures]
        if row.get("integrity_failure_id")
    }
    atomic_write_json(path, list(by_id.values()))


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
    if (config.get("storage") or {}).get("extraction_integrity"):
        integrity_path = storage_path(config, "extraction_integrity")
        if integrity_path.exists():
            integrity_path.unlink()
            removed.append(str(integrity_path))
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
    """Open and verify Neo4j with bounded retries for transient connectivity loss."""
    settings = neo4j_settings(config)
    save = config.get("save") or {}
    attempts = max(1, int(save.get("max_attempts", 3)))
    backoff = max(0.0, float(save.get("retry_backoff_seconds", 1.0)))
    for attempt in range(1, attempts + 1):
        driver = neo4j.GraphDatabase.driver(
            settings["uri"], auth=(settings["username"], settings["password"])
        )
        try:
            driver.verify_connectivity()
            ensure_constraints(driver, settings["database"])
            return driver
        except Exception as error:  # noqa: BLE001 - classify before retrying
            driver.close()
            if not _transient_neo4j_error(error) or attempt >= attempts:
                raise
            if backoff:
                time.sleep(backoff * (2 ** (attempt - 1)))
    raise RuntimeError("Neo4j connection retry loop exhausted")


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


def _transient_neo4j_error(error: Exception) -> bool:
    """Only retry failures Neo4j classifies as connectivity/transient failures."""
    transient_types = (
        neo4j.exceptions.ServiceUnavailable,
        neo4j.exceptions.SessionExpired,
        neo4j.exceptions.TransientError,
    )
    return isinstance(error, transient_types)


def commit_facts_with_retry(
    driver: neo4j.Driver,
    config: dict[str, Any],
    connection,
    doc_id: str,
    facts: pd.DataFrame,
    ontology: dict[str, Any],
    resolution: Optional[dict[str, dict[str, Any]]],
    audit,
) -> tuple[int, int]:
    """Persist one accepted document with a bounded, auditable retry policy.

    A retry may replay already-written nodes if connectivity failed between node and
    relationship statements. Both writers use stable MERGE identities and set-union
    provenance, so replay is idempotent. The SQLite ledger reports COMMITTED only
    after the complete logical document save returns successfully.
    """
    save = config.get("save") or {}
    max_attempts = max(1, int(save.get("max_attempts", 3)))
    backoff = max(0.0, float(save.get("retry_backoff_seconds", 1.0)))
    for local_attempt in range(1, max_attempts + 1):
        lifetime_attempt = state.begin_neo4j_commit(connection, doc_id)
        audit.emit(
            "neo4j_save_attempt", stage="commit", attempt_number=local_attempt,
            lifetime_attempt=lifetime_attempt, accepted_fact_count=len(facts),
        )
        try:
            nodes, relationships = commit_facts(
                driver, config, facts, ontology, resolution,
            )
        except Exception as error:  # noqa: BLE001 - classify before retrying
            state.finish_neo4j_commit(
                connection, doc_id, success=False, error=str(error),
            )
            retryable = _transient_neo4j_error(error) and local_attempt < max_attempts
            audit.emit(
                "neo4j_save_failed", stage="commit", attempt_number=local_attempt,
                lifetime_attempt=lifetime_attempt, retryable=retryable,
                error_type=type(error).__name__, error=str(error),
            )
            if not retryable:
                raise
            if backoff:
                time.sleep(backoff * (2 ** (local_attempt - 1)))
            continue
        state.finish_neo4j_commit(
            connection, doc_id, success=True,
            nodes_merged=nodes, relationships_merged=relationships,
        )
        audit.emit(
            "neo4j_save_succeeded", stage="commit", attempt_number=local_attempt,
            lifetime_attempt=lifetime_attempt, nodes_merged=nodes,
            relationships_merged=relationships,
        )
        return nodes, relationships
    raise RuntimeError("Neo4j save retry loop exhausted")


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
    audit = audit_for_document(config, doc_id)
    audit.emit(
        "document_observed",
        stage="persistence",
        filename=Path(str(document.get("source_path") or "")).name or None,
        source_path=document.get("source_path"),
        title=document.get("title"),
        source_url=document.get("source_url"),
        retrieval_url=document.get("retrieval_url"),
        canonical_source_url=document.get("canonical_source_url"),
        content_hash=document.get("content_hash"),
        character_count=len(document.get("text") or ""),
        normalized_source_text=(document.get("text") if audit.capture_content else None),
        initial_checkpoint_status=status,
    )
    audit.emit(
        "document_registered",
        stage="persistence",
        checkpoint_status=status,
        is_new=is_new,
    )
    if not is_new and doc_id != document["doc_id"]:
        # A changed fetch at the same URL must resume the saved original source when
        # unfinished, never pair new text with an old extraction/checkpoint identity.
        unfinished = status not in (state.RESOLVED, state.COMMITTED) or (
            status == state.RESOLVED and driver is not None
        )
        original = _load_persisted_document(config, doc_id) if unfinished else None
        if original is None or not original.get("text"):
            summary.update(skipped="duplicate", status=status)
            audit.emit(
                "document_cache_hit",
                stage="checkpoint",
                checkpoint_status=status,
                disposition="duplicate",
            )
            audit.emit(
                "document_skipped", stage="checkpoint",
                reason="duplicate", checkpoint_status=status,
            )
            logger.info("doc %s is duplicate of %s, skipping", document["doc_id"], doc_id)
            return summary
        document = original
        logger.info("URL rediscovery: resuming unfinished original doc %s", doc_id)

    if status == state.COMMITTED and resume:
        summary.update(skipped="already committed", status=status)
        logger.info("doc %s already committed, skipping", doc_id)
        audit.emit(
            "document_cache_hit",
            stage="checkpoint",
            checkpoint_status=status,
            disposition="already_committed",
        )
        audit.emit(
            "document_skipped", stage="checkpoint",
            reason="already_committed", checkpoint_status=status,
        )
        return summary
    if status == state.RESOLVED and driver is None and resume:
        summary.update(skipped="already resolved", status=status)
        logger.info("doc %s already resolved without Neo4j, skipping", doc_id)
        audit.emit(
            "document_cache_hit",
            stage="checkpoint",
            checkpoint_status=status,
            disposition="already_resolved",
        )
        audit.emit(
            "document_skipped", stage="checkpoint",
            reason="already_resolved", checkpoint_status=status,
        )
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
        audit=audit,
    ) if llm is not None else None
    stage = "persistence"
    stage_started = time.monotonic()
    extraction_metrics: dict[str, Any] = {}
    integrity_failures: list[dict[str, Any]] = []
    origin_name = document.get("origin_entity_name")
    if not origin_name and origin_entity:
        origin_row = connection.execute(
            "SELECT entity_name FROM frontier WHERE entity_key = ?", (origin_entity,),
        ).fetchone()
        origin_name = origin_row["entity_name"] if origin_row else None

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
                integrity_failures=integrity_failures,
                audit=audit,
            )
            summary.update(extraction_metrics)
            save_extraction(
                config,
                doc_id,
                graph,
                chunk_index,
                extraction_metrics,
                integrity_failures,
            )
            _persist_integrity_failures(config, integrity_failures)
            state.set_document_status(connection, doc_id, state.EXTRACTED)
            status = state.EXTRACTED

        # --- Prompt 2 and the facts layer ---
        if status == state.EXTRACTED:
            advance_stage("resolution")
            if graph is None:
                graph, chunk_index = load_extraction(config, doc_id)
                logger.info("doc %s: reusing stored Prompt 1 output", doc_id)
                parsed_graph = graph
                for chunk_id, index in chunk_index.items():
                    audit.emit(
                        "chunk_checkpoint_reused", stage="extraction",
                        chunk_id=chunk_id, chunk_index=index,
                        disposition="reused_document_extraction_checkpoint",
                        historical_prompt="unavailable_from_document_checkpoint",
                        historical_model_response="unavailable_from_document_checkpoint",
                    )
                resumed_outcomes: list[dict[str, Any]] = []
                graph, resumed_failures = enforce_referential_integrity(
                    graph,
                    document_id=doc_id,
                    chunk_index_by_id=chunk_index,
                    ontology=ontology,
                    recovery_outcomes=resumed_outcomes,
                )
                for outcome in resumed_outcomes:
                    event_type = outcome["event_type"]
                    payload = {
                        key: value for key, value in outcome.items()
                        if key != "event_type"
                    }
                    audit.emit(
                        event_type,
                        stage="extraction_integrity",
                        source="document_checkpoint",
                        extraction_relationship_id=(
                            "relationship-" + str(outcome["relationship_integrity_id"])
                        ),
                        **payload,
                    )
                resumed_recoveries = [
                    item for item in resumed_outcomes
                    if item["event_type"] in {
                        "endpoint_id_recovered", "endpoint_node_reconstructed",
                    }
                ]
                if resumed_failures or resumed_recoveries:
                    integrity_failures.extend(resumed_failures)
                    extraction_metrics["extraction_integrity_failures"] = len(
                        resumed_failures
                    )
                    extraction_metrics[
                        "extraction_integrity_recovered_relationships"
                    ] = len({
                        item["relationship_integrity_id"]
                        for item in resumed_recoveries
                    })
                    # Upgrade the checkpoint in place without discarding the invalid
                    # payload: the diagnostic keeps the original relationship.
                    save_extraction(
                        config,
                        doc_id,
                        graph,
                        chunk_index,
                        extraction_metrics,
                        integrity_failures,
                    )
                    _persist_integrity_failures(config, integrity_failures)
                audit.emit(
                    "parsed_extraction_graph",
                    stage="extraction",
                    source="checkpoint",
                    node_count=len(parsed_graph.nodes),
                    relationship_count=len(parsed_graph.relationships),
                    graph=(
                        parsed_graph.model_dump(mode="json")
                        if audit.capture_content
                        else None
                    ),
                )
                audit.emit(
                    "structural_integrity_checked",
                    stage="extraction_integrity",
                    source="checkpoint",
                    valid=not resumed_failures,
                    failure_count=len(resumed_failures),
                    failures=(
                        resumed_failures
                        if audit.capture_content
                        else [
                            {
                                "integrity_failure_id": failure["integrity_failure_id"],
                                "chunk_id": failure["chunk_id"],
                                "chunk_index": failure["chunk_index"],
                                "relation_type": failure["relation_type"],
                                "missing_node_ids": failure["missing_node_ids"],
                                "reason": failure["reason"],
                            }
                            for failure in resumed_failures
                        ]
                    ),
                )
                parsed_node_ids = {node.id for node in parsed_graph.nodes}
                for relationship in parsed_graph.relationships:
                    missing = [
                        endpoint for endpoint in (
                            relationship.start_node_id, relationship.end_node_id,
                        ) if endpoint not in parsed_node_ids
                    ]
                    relation_chunk = str(relationship.start_node_id).split(":", 1)[0]
                    if relation_chunk not in chunk_index:
                        relation_chunk = str(relationship.end_node_id).split(":", 1)[0]
                    relationship_id = extraction_relationship_id(doc_id, relationship)
                    integrity_id = relationship_id.removeprefix("relationship-")
                    endpoint_outcomes = [
                        item for item in resumed_outcomes
                        if item["relationship_integrity_id"] == integrity_id
                    ]
                    unresolved = any(item["event_type"] in {
                        "endpoint_recovery_ambiguous", "endpoint_unrecoverable",
                    } for item in endpoint_outcomes)
                    recovered = any(item["event_type"] in {
                        "endpoint_id_recovered", "endpoint_node_reconstructed",
                    } for item in endpoint_outcomes)
                    audit.emit(
                        "relationship_integrity_result",
                        stage="extraction_integrity",
                        extraction_relationship_id=relationship_id,
                        chunk_id=(relation_chunk if relation_chunk in chunk_index else None),
                        chunk_index=chunk_index.get(relation_chunk),
                        source="document_checkpoint",
                        source_node_exists=relationship.start_node_id in parsed_node_ids,
                        target_node_exists=relationship.end_node_id in parsed_node_ids,
                        missing_node_ids=missing,
                        action=(
                            "dropped_missing_endpoint" if unresolved
                            else "recovered_endpoint" if recovered
                            else "retained"
                        ),
                        original_relationship=(
                            relationship.model_dump(mode="json")
                            if audit.capture_content else {
                                "start_node_id": relationship.start_node_id,
                                "end_node_id": relationship.end_node_id,
                                "type": relationship.type,
                            }
                        ),
                    )
            cache = state.load_resolution_cache(connection)
            resolution, new_entries = resolve_entities(
                graph,
                llm=measured_llm,
                cache=cache,
                aliases=ontology.get("aliases"),
                document_text=document.get("text") or "",
                audit=audit,
            )
            if new_entries:
                state.save_resolution(connection, new_entries)
            _write_resolution_parquet(config, doc_id, resolution)
            audit.emit(
                "entity_resolution_completed",
                stage="resolution",
                resolved_count=len(resolution),
                new_entry_count=len(new_entries),
                results=resolution if audit.capture_content else None,
            )
            original_nodes = {node.id: node for node in graph.nodes}
            for node_id, entry in resolution.items():
                original = original_nodes.get(node_id)
                audit.emit(
                    "entity_resolution_item",
                    stage="resolution",
                    node_id=node_id,
                    original_text=(
                        (original.properties or {}).get("name")
                        if original is not None and audit.capture_content else None
                    ),
                    original_type=original.label if original is not None else None,
                    canonical_id=entry.get("canonical_id"),
                    canonical_name=(entry.get("canonical_name") if audit.capture_content else None),
                    mention=(entry.get("mention") if audit.capture_content else None),
                    aliases=(entry.get("aliases") if audit.capture_content else None),
                    method=entry.get("method"),
                    related_llm_call_ids=(
                        audit.contextual_llm_call_ids(
                            "resolution", "entity_type",
                            original.label if original is not None else "",
                        ) if "llm" in str(entry.get("method")) else []
                    ),
                    merged_with_multiple_names=len(entry.get("aliases") or []) > 1,
                    possible_duplicates_remaining="not_observable",
                )
            unresolved_relationships = [
                {
                    "relation_type": rel.type,
                    "start_node_id": rel.start_node_id,
                    "end_node_id": rel.end_node_id,
                    "missing_resolution_ids": [
                        node_id
                        for node_id in (rel.start_node_id, rel.end_node_id)
                        if node_id not in resolution
                    ],
                }
                for rel in graph.relationships
                if rel.start_node_id not in resolution or rel.end_node_id not in resolution
            ]
            audit.emit(
                "entity_resolution_integrity_checked",
                stage="resolution",
                valid=not unresolved_relationships,
                failures=unresolved_relationships,
            )
            facts = graph_to_facts(
                graph, document, chunk_index, resolution, model=model, audit=audit,
            )
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
                for list_row in list_rows:
                    candidate_id = candidate_fact_id_for_audit(
                        str(list_row.get("doc_id") or ""),
                        str(list_row.get("subject_id") or ""),
                        str(list_row.get("relation") or ""),
                        str(list_row.get("object_id") or ""),
                        str(list_row.get("evidence") or ""),
                    )
                    audit.emit(
                        "candidate_fact_created",
                        stage="fact_construction",
                        candidate_fact_id=candidate_id,
                        fact_id=list_row.get("fact_id"),
                        extraction_relationship_id=None,
                        chunk_id=None,
                        chunk_index=list_row.get("chunk_index"),
                        subject_id=list_row.get("subject_id"),
                        object_id=list_row.get("object_id"),
                        subject_mention=(
                            list_row.get("subject_mention") if audit.capture_content else None
                        ),
                        object_mention=(
                            list_row.get("object_mention") if audit.capture_content else None
                        ),
                        subject_name=(
                            list_row.get("subject_name") if audit.capture_content else None
                        ),
                        object_name=(
                            list_row.get("object_name") if audit.capture_content else None
                        ),
                        predicate=list_row.get("relation"),
                        original_evidence=(
                            list_row.get("evidence") if audit.capture_content else None
                        ),
                        originating_llm_call_ids=[],
                        creation_path="deterministic_supplier_list_extraction",
                        status="candidate",
                    )
                if list_rows:
                    facts = pd.concat([facts, pd.DataFrame(list_rows)], ignore_index=True)
            except Exception as list_error:  # noqa: BLE001
                logger.warning("structured list extraction skipped: %s", list_error)
            summary["candidate_facts"] = len(facts)
            audit.emit(
                "candidate_facts_created",
                stage="fact_construction",
                candidate_fact_ids=facts.fact_id.tolist(),
                count=len(facts),
            )
            advance_stage("validation")
            accepted, rejected = validate_facts_frame(
                facts, config=config, ontology=ontology, llm=measured_llm,
                document_text=document.get("text") or "",
                audit=audit,
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
            audit.emit(
                "validation_completed",
                stage="validation",
                accepted_fact_ids=accepted.fact_id.tolist(),
                accepted=[
                    {
                        "fact_id": row.get("fact_id"),
                        "expansion_relevant": bool(row.get("expansion_relevant")),
                        "tier_disposition": (
                            "eligible_for_quality_aggregation"
                            if bool(row.get("expansion_relevant"))
                            else "context_only_not_upstream"
                        ),
                    }
                    for row in accepted.to_dict(orient="records")
                ],
                rejected=[
                    {
                        "candidate_fact_id": row.get("candidate_fact_id"),
                        "stage": row.get("rejection_stage"),
                        "reason": row.get("rejection_reason"),
                    }
                    for row in rejected.to_dict(orient="records")
                ],
            )
            for row in accepted.to_dict(orient="records"):
                audit.emit(
                    "candidate_accepted",
                    stage="validation",
                    candidate_fact_id=row.get("candidate_fact_id"),
                    fact_id=row.get("fact_id"),
                    final_evidence=(row.get("evidence") if audit.capture_content else None),
                    validation_status=row.get("validation_status"),
                    evidence_repaired=row.get("evidence_repair_status") == "succeeded",
                    expansion_relevant=bool(row.get("expansion_relevant")),
                    related_llm_call_ids=audit.contextual_llm_call_ids(
                        "validation", "candidate_fact_id", row.get("candidate_fact_id"),
                    ) + audit.contextual_llm_call_ids(
                        "validation", "candidate_fact_ids", row.get("candidate_fact_id"),
                    ),
                )
            for row in rejected.to_dict(orient="records"):
                audit.emit(
                    "candidate_rejected",
                    stage="validation",
                    candidate_fact_id=row.get("candidate_fact_id"),
                    rejection_stage=row.get("rejection_stage"),
                    rejection_reason=row.get("rejection_reason"),
                    evidence=(row.get("evidence") if audit.capture_content else None),
                    repair_attempted=bool(row.get("repair_attempted")),
                    repair_result=row.get("repair_result"),
                    related_llm_call_ids=audit.contextual_llm_call_ids(
                        "validation", "candidate_fact_id", row.get("candidate_fact_id"),
                    ) + audit.contextual_llm_call_ids(
                        "validation", "candidate_fact_ids", row.get("candidate_fact_id"),
                    ),
                )
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
                nodes, rels = commit_facts_with_retry(
                    driver, config, connection, doc_id, facts, ontology, resolution, audit,
                )
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
        # Keep the live object attached: the finally block below adds extraction,
        # timing, and LLM counters before the exception reaches run_ingest.
        try:
            setattr(error, "_schain_document_summary", summary)
        except (AttributeError, TypeError):
            pass
        audit.emit(
            "document_failed",
            stage=stage,
            checkpoint_status=keep,
            error_type=type(error).__name__,
            error=str(error),
        )
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
        audit.emit(
            "document_finished",
            stage=stage,
            checkpoint_status=status,
            elapsed_seconds=summary["elapsed_seconds"],
            llm_calls=summary["llm_calls"],
            extraction_cache_hits=summary["extraction_cache_hits"],
            accepted_facts=summary["accepted_facts"],
            rejected_facts=summary["rejected_facts"],
        )

    summary["status"] = status
    return summary
