"""Command-line entry point.

    python -m schain ingest   --config config/ingest.yaml
    python -m schain expand   --config config/expansion.yaml
    python -m schain validate --config config/ingest.yaml
    python -m schain export   --config config/ingest.yaml

Resume is the default. Pass --restart for an explicit fresh run.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

from neo4j_graphrag.exceptions import LLMGenerationError

from . import state
from .config import load_config, storage_path
from .expansion.loop import run_expand
from .ingest import run_ingest
from .llm.provider import build_llm
from .ontology import load_ontology
from .pipeline import clear_generated_data, open_driver

logger = logging.getLogger("schain")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="schain", description=__doc__)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(sub: argparse.ArgumentParser, default_config: str) -> None:
        sub.add_argument("--config", type=Path, default=Path(default_config))
        sub.add_argument(
            "--restart", action="store_true",
            help="clear checkpoint state and generated data, then run from scratch",
        )
        sub.add_argument(
            "--no-neo4j", action="store_true",
            help="stop after the facts table instead of merging into Neo4j",
        )

    ingest = subparsers.add_parser("ingest", help="process documents already in hand")
    add_common(ingest, "config/ingest.yaml")
    ingest.add_argument("--input", type=Path, default=None, help="file or directory")

    expand = subparsers.add_parser("expand", help="discover suppliers from seed companies")
    add_common(expand, "config/expansion.yaml")
    expand.add_argument("--max-tier", type=int, default=None, help="override expansion.max_tier")
    expand.add_argument(
        "--max-documents", type=int, default=None, help="stop after this many documents"
    )

    validate = subparsers.add_parser(
        "validate", help="compare extracted facts with an external reference dataset"
    )
    validate.add_argument("--config", type=Path, default=Path("config/ingest.yaml"))
    validate.add_argument(
        "--reference", type=Path, default=None,
        help="supplier-customer CSV/JSON (default: first file in data/external/sectivia)",
    )

    export = subparsers.add_parser(
        "export",
        help="write frontend-ready graph JSON from persisted facts (no LLM, no Neo4j)",
    )
    export.add_argument("--config", type=Path, default=Path("config/ingest.yaml"))
    export.add_argument(
        "--output", type=Path, default=None,
        help="output JSON path (default: storage.graph_export)",
    )
    return parser


def _run_validate(args: argparse.Namespace, config: dict[str, Any]) -> int:
    import pandas as pd

    from .validation.sectivia import compare_with_facts, find_reference_file, load_reference

    reference_path = args.reference or find_reference_file()
    if reference_path is None:
        logger.error(
            "no reference dataset found. Place a supplier-customer CSV under "
            "data/external/sectivia/ or pass --reference. See docs/EXTERNAL_DATASETS.md."
        )
        return 1

    facts_path = storage_path(config, "extracted_facts")
    facts = pd.read_parquet(facts_path) if facts_path.exists() else pd.DataFrame()
    report = compare_with_facts(load_reference(Path(reference_path)), facts)
    print(json.dumps(report, indent=2))
    return 0


def _run_export(args: argparse.Namespace, config: dict[str, Any]) -> int:
    from .graph_export import export_graph

    path = export_graph(config, output=args.output, mode="export")
    print(json.dumps({"graph_export": str(path)}, indent=2))
    return 0


def _attach_graph_export(
    config: dict[str, Any], summary: dict[str, Any], mode: str
) -> dict[str, Any]:
    """Write the derived graph JSON after a successful ingest/expand run."""
    from .graph_export import export_graph

    path = export_graph(config, mode=mode, pipeline_summary=summary)
    summary = dict(summary)
    summary["graph_export"] = str(path)
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    if args.mode == "validate":
        return _run_validate(args, config)
    if args.mode == "export":
        # Operate only on persisted artifacts — no LLM, Neo4j, or retrieval.
        return _run_export(args, config)

    # CLI flags override the config file.
    if args.restart:
        config.setdefault("runtime", {})["restart_from_scratch"] = True
    if args.mode == "expand":
        if args.max_tier is not None:
            config.setdefault("expansion", {})["max_tier"] = args.max_tier
        if args.max_documents is not None:
            config.setdefault("expansion", {})["max_documents"] = args.max_documents

    if config.get("runtime", {}).get("restart_from_scratch"):
        removed = clear_generated_data(config)
        logger.warning("restart: removed %d generated file(s)", len(removed))

    ontology = load_ontology(config)
    connection = state.connect(storage_path(config, "state_db"))
    driver = None
    try:
        llm = build_llm(config)
        if not args.no_neo4j:
            driver = open_driver(config)
        if args.mode == "ingest":
            summary = run_ingest(config, connection, llm, ontology, driver, args.input)
        else:
            summary = run_expand(config, connection, llm, ontology, driver)
        summary = _attach_graph_export(config, summary, args.mode)
    except LLMGenerationError as error:
        logger.error(
            "LLM call failed (%s). Check that the server at %s is running and that "
            "llm.model matches what it serves.",
            error, config.get("llm", {}).get("base_url", "the configured base_url"),
        )
        return 1
    except Exception as error:  # noqa: BLE001 - report what failed, then stop
        logger.error("%s failed: %s: %s", args.mode, type(error).__name__, error)
        return 1
    finally:
        if driver is not None:
            driver.close()
        connection.close()

    logger.info("done")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
