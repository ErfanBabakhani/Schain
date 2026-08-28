"""Entry point for the supply-chain KG pipeline.

V1 mode INGEST, one document:

    python src/main.py ingest --input path/to/document.txt

The steps below are the INGEST flow from the architecture brief:

    document -> normalize/store -> Prompt 1 -> Prompt 2 -> facts -> Neo4j MERGE
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from neo4j_graphrag.exceptions import LLMGenerationError

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config, neo4j_settings, storage_path
from documents import append_raw_jsonl, load_document, upsert_normalized_parquet
from kg.extract import extract_graph
from kg.facts import graph_to_facts, upsert_facts_parquet
from kg.resolve import resolve_entities
from kg.writer import (
    ensure_constraints,
    facts_to_graph_rows,
    merge_nodes,
    merge_relationships,
)
from llm.provider import build_llm
from ontology import ENTITY_TYPES, RELATION_TYPES

logger = logging.getLogger("schain")


def ingest_document(
    input_path: Path,
    config: dict,
    source_url: str | None = None,
    title: str | None = None,
    write_to_neo4j: bool = True,
    llm=None,
) -> dict:
    """Run the full INGEST path for one document and return a summary."""
    ingest_config = config.get("ingest", {})

    logger.info("step 1/5 load and normalize the document")
    document = load_document(input_path, source_url=source_url, title=title)
    append_raw_jsonl(document, storage_path(config, "raw_documents"))
    upsert_normalized_parquet(document, storage_path(config, "normalized_documents"))

    logger.info("step 2/5 Prompt 1: entity and relation extraction")
    llm = llm or build_llm(config)  # injectable so the pipeline is testable offline
    graph, chunk_index_by_id = extract_graph(
        llm,
        document,
        chunk_size=ingest_config.get("chunk_size", 2000),
        chunk_overlap=ingest_config.get("chunk_overlap", 200),
        on_error=ingest_config.get("on_error", "IGNORE"),
    )

    logger.info("step 3/5 Prompt 2: entity disambiguation")
    resolution = resolve_entities(graph, llm=llm)
    if resolution:
        resolution_frame = pd.DataFrame(
            [{"doc_id": document["doc_id"], "node_id": node_id, **entry}
             for node_id, entry in resolution.items()]
        )
        resolution_frame["aliases"] = resolution_frame["aliases"].apply(list)
        resolution_frame.to_parquet(
            storage_path(config, "entity_resolution"), index=False
        )

    logger.info("step 4/5 write the facts table")
    facts = graph_to_facts(graph, document, chunk_index_by_id, resolution)
    total_facts = upsert_facts_parquet(facts, storage_path(config, "extracted_facts"))
    logger.info("%d fact(s) from this document, %d in total", len(facts), total_facts)

    summary = {
        "doc_id": document["doc_id"],
        "chunks": len(set(chunk_index_by_id.values())),
        "nodes_extracted": len(graph.nodes),
        "relationships_extracted": len(graph.relationships),
        "canonical_entities": len({e["canonical_id"] for e in resolution.values()}),
        "facts": len(facts),
        "facts_total": total_facts,
        "nodes_merged": 0,
        "relationships_merged": 0,
    }

    if not write_to_neo4j:
        logger.info("step 5/5 skipped: --no-neo4j")
        return summary

    logger.info("step 5/5 merge into Neo4j")
    import neo4j

    settings = neo4j_settings(config)
    nodes, relationships = facts_to_graph_rows(facts, resolution)
    with neo4j.GraphDatabase.driver(
        settings["uri"], auth=(settings["username"], settings["password"])
    ) as driver:
        driver.verify_connectivity()
        ensure_constraints(driver, settings["database"])
        summary["nodes_merged"] = merge_nodes(
            driver, settings["database"], nodes, ENTITY_TYPES
        )
        summary["relationships_merged"] = merge_relationships(
            driver, settings["database"], relationships, RELATION_TYPES
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    ingest = subparsers.add_parser("ingest", help="process documents already in hand")
    ingest.add_argument("--input", required=True, type=Path, help=".txt, .md or .pdf file")
    ingest.add_argument("--config", type=Path, default=None, help="path to a YAML config")
    ingest.add_argument("--url", default=None, help="source URL, recorded for deduplication")
    ingest.add_argument("--title", default=None, help="document title (defaults to filename)")
    ingest.add_argument(
        "--no-neo4j",
        action="store_true",
        help="stop after the facts table instead of merging into Neo4j",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s"
    )

    config = load_config(args.config)
    try:
        summary = ingest_document(
            args.input,
            config,
            source_url=args.url,
            title=args.title,
            write_to_neo4j=not args.no_neo4j,
        )
    except LLMGenerationError as error:
        # By far the most common setup failure: no model server on the configured URL.
        logger.error(
            "LLM call failed (%s). Check that the server at %s is running and that "
            "llm.model matches what it serves.",
            error,
            config.get("llm", {}).get("base_url", "the configured base_url"),
        )
        return 1
    except Exception as error:  # noqa: BLE001 - report the step that failed, then stop
        logger.error("ingest failed: %s: %s", type(error).__name__, error)
        return 1
    logger.info("done: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
