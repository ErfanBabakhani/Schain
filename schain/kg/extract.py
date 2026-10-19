"""Prompt 1 - split the document and extract entities and relations.

Both steps are reused from neo4j-graphrag: ``FixedSizeSplitter`` and
``LLMEntityRelationExtractor``. Only the prompt and the schema are ours.

``create_lexical_graph=False`` on purpose: the library would otherwise add Document and
Chunk nodes to Neo4j. V1 keeps Neo4j limited to the static ontology and holds the
evidence trail in the facts Parquet layer instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from neo4j_graphrag.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.components.types import Neo4jGraph, TextChunks
from neo4j_graphrag.llm import LLMInterface

from ..checkpoint import atomic_write_json
from ..async_runtime import run_async
from ..audit import llm_call_context, prompt_hash
from ..llm.provider import RuntimeDeadlineExceeded
from ..ontology import EXTRACTION_EXAMPLES, SupplyChainERTemplate, build_graph_schema
from .facts import extraction_relationship_id
from .integrity import enforce_referential_integrity

logger = logging.getLogger(__name__)


def classify_extraction_error(error: BaseException) -> str:
    """Stable first-failure categories for Prompt-1 telemetry (not retry policy).

    Categories match the observability contract:
    transport_error | timeout | invalid_json | schema_error | truncated_output |
    empty_output | parser_error | other
    """
    if isinstance(error, RuntimeDeadlineExceeded):
        return "timeout"
    text = f"{type(error).__name__}: {error}".casefold()
    if any(token in text for token in ("timeout", "timed out", "deadline")):
        return "timeout"
    if any(token in text for token in ("json", "expecting value", "jsondecode", "repair_json")):
        return "invalid_json"
    if any(token in text for token in ("schema", "validation error", "pydantic", "model_validate")):
        return "schema_error"
    if any(token in text for token in ("empty response", "empty content", "no content")):
        return "empty_output"
    if any(token in text for token in ("truncat", "max_tokens", "finish_reason", "length")):
        return "truncated_output"
    if any(token in text for token in ("parse", "parser", "decode")):
        return "parser_error"
    if any(token in text for token in (
        "connection", "connect error", "server disconnected", "remote protocol",
        "network", "bad gateway", "gateway timeout", "502", "503", "504",
        "rate limit", "429", "transport", "unavailable", "connectionerror",
    )):
        return "transport_error"
    return "other"


def _is_transient_llm_error(error: BaseException) -> bool:
    """Only transport/rate-limit failures retry. JSON/schema must not spin retries."""
    if isinstance(error, RuntimeDeadlineExceeded):
        return False
    return classify_extraction_error(error) in ("timeout", "transport_error")


def extract_graph(
    llm: LLMInterface,
    document: dict[str, Any],
    ontology: dict[str, Any],
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    on_error: str = "RAISE",
    max_concurrency: int = 4,
    structured_output: bool = False,
    checkpoint_dir: Path | None = None,
    metrics: dict[str, Any] | None = None,
    max_chunks_per_document: int | None = None,
    max_retries: int = 1,
    origin_name: str | None = None,
    evidence_policy_file: str | None = None,
    integrity_failures: list[dict[str, Any]] | None = None,
    audit=None,
) -> tuple[Neo4jGraph, dict[str, int]]:
    """Extract a graph from one normalized document record.

    Returns the extracted graph plus a ``chunk_id -> chunk index`` map. The extractor
    prefixes every node id with its chunk id, so that map is what lets a fact be traced
    back to the part of the document it came from. When ``checkpoint_dir`` is supplied,
    successful chunks are saved separately and reused after an interruption. The cache
    key includes the exact prompt, model settings and parsing policy.
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if max_chunks_per_document is not None and max_chunks_per_document < 1:
        raise ValueError("max_chunks_per_document must be positive when supplied")
    metrics = metrics if metrics is not None else {}
    splitter = FixedSizeSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = run_async(splitter.run(text=document["text"]))
    logger.info("split doc %s into %d chunk(s)", document["doc_id"], len(chunks.chunks))
    # FixedSizeSplitter normally generates random ids.  Assign provenance ids
    # before applying a cap so omitted chunks are traceable too.
    for chunk in chunks.chunks:
        chunk.uid = hashlib.sha256(
            f"{document['doc_id']}:{chunk.index}:{chunk.text}".encode("utf-8")
        ).hexdigest()[:32]
    all_chunks = list(chunks.chunks)
    available = len(chunks.chunks)
    selection = "all"
    if max_chunks_per_document is not None and available > max_chunks_per_document:
        selected_chunks = chunks.chunks
        selection = "original_order"
        if origin_name:
            from .relation_evidence import (
                classify_supply_direction, load_relation_evidence,
                UPSTREAM_EXPLICIT, DIRECTION_AMBIGUOUS, DOWNSTREAM_EXPLICIT,
            )
            policy = load_relation_evidence(evidence_policy_file)
            aliases = [alias for alias, target in (ontology.get("aliases", {}).get("Company") or {}).items()
                       if target == origin_name]
            priorities = {UPSTREAM_EXPLICIT: 0, DIRECTION_AMBIGUOUS: 1, DOWNSTREAM_EXPLICIT: 2}
            selected_chunks = sorted(chunks.chunks, key=lambda chunk: (
                priorities.get(classify_supply_direction(
                    origin_name, chunk.text, aliases=aliases, policy=policy), 1), chunk.index,
            ))
            selection = "upstream_priority_then_original_order"
        chunks.chunks = sorted(selected_chunks[:max_chunks_per_document], key=lambda chunk: chunk.index)
        logger.warning(
            "doc %s explicit chunk cap: extracting %d/%d chunks (%s); full source is "
            "retained, remaining chunks are unexamined", document["doc_id"],
            max_chunks_per_document, available, selection,
        )
    selected_ids = {chunk.chunk_id for chunk in chunks.chunks}
    if audit is not None:
        for chunk in all_chunks:
            processed = chunk.chunk_id in selected_ids
            audit.emit(
                "chunk_discovered", stage="chunking",
                chunk_id=chunk.chunk_id, chunk_index=chunk.index,
                character_count=len(chunk.text), chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                disposition="processed" if processed else "omitted",
                omission_reason=(None if processed else "max_chunks_per_document_cap"),
                selection_strategy=selection,
                source_text=chunk.text if audit.capture_content else None,
            )
    metrics.update(
        extraction_chunks_available=available, extraction_chunks=len(chunks.chunks),
        extraction_chunks_omitted=available - len(chunks.chunks),
        extraction_chunk_indices=[chunk.index for chunk in chunks.chunks],
        extraction_coverage="partial" if available > len(chunks.chunks) else "complete",
        extraction_chunk_selection=selection,
        extraction_calls=0, extraction_cache_hits=0, extraction_retries=0,
        extraction_retry_reasons={}, extraction_first_failure_reasons={},
    )

    template = SupplyChainERTemplate()
    schema = build_graph_schema(ontology)
    extractor = LLMEntityRelationExtractor(
        llm=llm,
        prompt_template=template,
        create_lexical_graph=False,
        on_error=OnError(on_error),
        max_concurrency=max_concurrency,
        use_structured_output=structured_output,
    )

    async def _run_extract() -> Neo4jGraph:
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrency)

        async def extract_chunk(chunk) -> Neo4jGraph:
            cache_file = None
            if checkpoint_dir is not None:
                signature = {
                    "version": 1,
                    "chunk_id": chunk.chunk_id,
                    "prompt": template.format(
                        text=chunk.text, schema=schema.model_dump(exclude_none=True),
                        examples=EXTRACTION_EXAMPLES,
                    ),
                    "model": getattr(llm, "model_name", None),
                    "provider": f"{type(getattr(llm, '_schain_wrapped_llm', llm)).__module__}."
                                f"{type(getattr(llm, '_schain_wrapped_llm', llm)).__qualname__}",
                    "model_params": getattr(llm, "model_params", None),
                    "structured_output": structured_output,
                    "on_error": on_error,
                }
                digest = hashlib.sha256(
                    json.dumps(signature, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
                cache_file = Path(checkpoint_dir) / f"{chunk.index}-{digest}.json"
                if cache_file.exists():
                    try:
                        cached = Neo4jGraph.model_validate_json(
                            cache_file.read_text(encoding="utf-8")
                        )
                    except (ValueError, OSError):
                        logger.warning("ignoring damaged chunk checkpoint %s", cache_file)
                    else:
                        logger.info("doc %s: reusing chunk %d", document["doc_id"], chunk.index)
                        metrics["extraction_cache_hits"] += 1
                        if audit is not None:
                            audit.emit(
                                "extraction_cache_hit",
                                stage="extraction",
                                chunk_id=chunk.chunk_id,
                                chunk_index=chunk.index,
                                checkpoint=str(cache_file),
                                cache_key=digest,
                                parsed_artifact=(cached.model_dump(mode="json")
                                                 if audit.capture_content else None),
                                node_count=len(cached.nodes),
                                relationship_count=len(cached.relationships),
                                historical_prompt="unavailable_from_chunk_checkpoint",
                                historical_model_response="unavailable_from_chunk_checkpoint",
                            )
                        return cached

            async with semaphore:
                for attempt in range(max_retries + 1):
                    attempt_id = (
                        f"extract-{str(document['doc_id'])[:12]}-"
                        f"{chunk.index}-{attempt + 1}"
                    )
                    try:
                        metrics["extraction_calls"] += 1
                        if audit is not None:
                            audit.emit(
                                "chunk_extraction_attempt_started", stage="extraction",
                                chunk_id=chunk.chunk_id, chunk_index=chunk.index,
                                extraction_attempt_id=attempt_id,
                                attempt_number=attempt + 1,
                            )
                        with llm_call_context(
                            chunk_id=chunk.chunk_id,
                            chunk_index=chunk.index,
                            extraction_attempt_id=attempt_id,
                            prompt_template_id="SupplyChainERTemplate.DEFAULT_TEMPLATE",
                            prompt_template_hash=prompt_hash(template.DEFAULT_TEMPLATE),
                            retry_attempt=attempt,
                        ):
                            result = await extractor.run(
                                chunks=TextChunks(chunks=[chunk]), schema=schema,
                                examples=EXTRACTION_EXAMPLES,
                            )
                        if audit is not None:
                            audit.emit(
                                "chunk_extraction_parsed", stage="extraction",
                                chunk_id=chunk.chunk_id, chunk_index=chunk.index,
                                extraction_attempt_id=attempt_id,
                                attempt_number=attempt + 1,
                                node_count=len(result.nodes),
                                relationship_count=len(result.relationships),
                                graph=(result.model_dump(mode="json")
                                       if audit.capture_content else None),
                                related_llm_call_ids=audit.contextual_llm_call_ids(
                                    "extraction", "extraction_attempt_id", attempt_id,
                                ),
                            )
                        break
                    except Exception as error:  # noqa: BLE001
                        category = classify_extraction_error(error)
                        reasons = metrics.setdefault("extraction_first_failure_reasons", {})
                        if attempt == 0:
                            reasons[category] = int(reasons.get(category, 0)) + 1
                        if audit is not None:
                            audit.emit(
                                "chunk_extraction_attempt_failed", stage="extraction",
                                chunk_id=chunk.chunk_id, chunk_index=chunk.index,
                                extraction_attempt_id=attempt_id,
                                attempt_number=attempt + 1,
                                error_type=type(error).__name__, error=str(error),
                                failure_kind=("parser_or_schema_error"
                                              if not _is_transient_llm_error(error)
                                              else "transient_llm_error"),
                            )
                        if attempt >= max_retries or not _is_transient_llm_error(error):
                            raise
                        metrics["extraction_retries"] += 1
                        retry_reasons = metrics.setdefault("extraction_retry_reasons", {})
                        retry_reasons[category] = int(retry_reasons.get(category, 0)) + 1
                        if audit is not None:
                            audit.emit(
                                "llm_retry",
                                stage="extraction",
                                chunk_id=chunk.chunk_id,
                                chunk_index=chunk.index,
                                attempt=attempt + 1,
                                max_retries=max_retries,
                                error_type=type(error).__name__,
                                error=str(error),
                            )
                        logger.warning(
                            "transient LLM error on doc %s chunk %d (%s/%s); retry %d/%d",
                            document["doc_id"], chunk.index, category, error,
                            attempt + 1, max_retries,
                        )
                        await asyncio.sleep(1.5)
                if cache_file is not None:
                    atomic_write_json(cache_file, result.model_dump())
                return result

        # Await every started chunk so completed work is saved even if a sibling fails.
        results = await asyncio.gather(
            *(extract_chunk(chunk) for chunk in chunks.chunks), return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return extractor.combine_chunk_graphs(None, results)

    graph = run_async(_run_extract())
    parsed_graph = graph
    chunk_index = {chunk.chunk_id: chunk.index for chunk in chunks.chunks}
    graph, failures = enforce_referential_integrity(
        graph,
        document_id=str(document["doc_id"]),
        chunk_index_by_id=chunk_index,
    )
    if integrity_failures is not None:
        integrity_failures.extend(failures)
    metrics["extraction_integrity_failures"] = len(failures)
    if failures:
        logger.warning(
            "doc %s: rejected %d dangling extracted relationship(s)",
            document["doc_id"],
            len(failures),
        )
    if audit is not None:
        node_ids = {node.id for node in parsed_graph.nodes}
        for relationship in parsed_graph.relationships:
            payload = relationship.model_dump(mode="json")
            missing = [endpoint for endpoint in (
                relationship.start_node_id, relationship.end_node_id,
            ) if endpoint not in node_ids]
            relationship_id = extraction_relationship_id(
                str(document["doc_id"]), relationship,
            )
            chunk_id = str(relationship.start_node_id).split(":", 1)[0]
            if chunk_id not in chunk_index:
                chunk_id = str(relationship.end_node_id).split(":", 1)[0]
            audit.emit(
                "relationship_integrity_result", stage="extraction_integrity",
                extraction_relationship_id=relationship_id,
                chunk_id=chunk_id if chunk_id in chunk_index else None,
                chunk_index=chunk_index.get(chunk_id),
                source_node_exists=relationship.start_node_id in node_ids,
                target_node_exists=relationship.end_node_id in node_ids,
                missing_node_ids=missing,
                action="dropped_missing_endpoint" if missing else "retained",
                original_relationship=(payload if audit.capture_content else {
                    "start_node_id": relationship.start_node_id,
                    "end_node_id": relationship.end_node_id,
                    "type": relationship.type,
                }),
            )
        audit.emit(
            "parsed_extraction_graph",
            stage="extraction",
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
            valid=not failures,
            failure_count=len(failures),
            failures=(
                failures
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
                    for failure in failures
                ]
            ),
        )

    logger.info(
        "extracted %d node(s) and %d relationship(s) from doc %s",
        len(graph.nodes),
        len(graph.relationships),
        document["doc_id"],
    )
    return graph, chunk_index
