"""Prompt 1 - split the document and extract entities and relations.

Both steps are reused from neo4j-graphrag: ``FixedSizeSplitter`` and
``LLMEntityRelationExtractor``. Only the prompt and the schema are ours.

``create_lexical_graph=False`` on purpose: the library would otherwise add Document and
Chunk nodes to Neo4j. V1 keeps Neo4j limited to the static ontology and holds the
evidence trail in the facts Parquet layer instead.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from neo4j_graphrag.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.components.types import Neo4jGraph
from neo4j_graphrag.llm import LLMInterface

from ..ontology import EXTRACTION_EXAMPLES, SupplyChainERTemplate, build_graph_schema

logger = logging.getLogger(__name__)


def extract_graph(
    llm: LLMInterface,
    document: dict[str, Any],
    ontology: dict[str, Any],
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    on_error: str = "IGNORE",
    max_concurrency: int = 4,
    structured_output: bool = False,
) -> tuple[Neo4jGraph, dict[str, int]]:
    """Extract a graph from one normalized document record.

    Returns the extracted graph plus a ``chunk_id -> chunk index`` map. The extractor
    prefixes every node id with its chunk id, so that map is what lets a fact be traced
    back to the part of the document it came from.
    """
    splitter = FixedSizeSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = asyncio.run(splitter.run(text=document["text"]))
    logger.info("split doc %s into %d chunk(s)", document["doc_id"], len(chunks.chunks))

    extractor = LLMEntityRelationExtractor(
        llm=llm,
        prompt_template=SupplyChainERTemplate(),
        create_lexical_graph=False,
        on_error=OnError(on_error),
        max_concurrency=max_concurrency,
        use_structured_output=structured_output,
    )
    graph = asyncio.run(
        extractor.run(
            chunks=chunks,
            schema=build_graph_schema(ontology),
            examples=EXTRACTION_EXAMPLES,
        )
    )
    logger.info(
        "extracted %d node(s) and %d relationship(s) from doc %s",
        len(graph.nodes),
        len(graph.relationships),
        document["doc_id"],
    )
    return graph, {chunk.chunk_id: chunk.index for chunk in chunks.chunks}
