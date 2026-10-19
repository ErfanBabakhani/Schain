"""Command-line entry point.

    python -m schain ingest   --config config/ingest.yaml
    python -m schain expand   --config config/expansion.yaml
    python -m schain validate --config config/ingest.yaml
    python -m schain export   --config config/ingest.yaml

Resume is the default. Pass --restart for an explicit fresh run.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Optional

from neo4j_graphrag.exceptions import LLMGenerationError

from . import state
from .config import (
    load_config, storage_path, log_effective_expansion_config, save_settings,
    validate_expansion_config,
)
from .durable import DurableCheckpointManager
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
        lifecycle = sub.add_mutually_exclusive_group()
        lifecycle.add_argument(
            "--restart", "--new-run", dest="restart", action="store_true",
            help="clear checkpoint state and generated data, then run from scratch",
        )
        lifecycle.add_argument(
            "--resume", action="store_true",
            help="explicitly resume the configured experiment (also the default)",
        )
        save_group = sub.add_mutually_exclusive_group()
        save_group.add_argument(
            "--save", dest="save_override", action="store_true", default=None,
            help="incrementally persist accepted facts to Neo4j",
        )
        save_group.add_argument(
            "--no-save", dest="save_override", action="store_false",
            help="disable Neo4j persistence without disabling local checkpoints",
        )
        save_group.add_argument(
            "--no-neo4j", action="store_true",
            help="legacy alias for --no-save",
        )
        sub.set_defaults(no_neo4j=False)
        sub.add_argument("--experiment-id", default=None)
        sub.add_argument("--checkpoint-store", type=Path, default=None)
        sub.add_argument("--force-checkpoint-unlock", action="store_true")

    ingest = subparsers.add_parser("ingest", help="process documents already in hand")
    add_common(ingest, "config/ingest.yaml")
    ingest.add_argument("--input", type=Path, default=None, help="file or directory")

    expand = subparsers.add_parser("expand", help="discover suppliers from seed companies")
    add_common(expand, "config/expansion.yaml")
    expand.add_argument("--max-tier", type=int, default=None, help="override expansion.max_tier")
    expand.add_argument(
        "--max-documents", type=int, default=None, help="stop after this many documents"
    )
    seed_override = expand.add_mutually_exclusive_group()
    seed_override.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "explicit comma-separated seed override (otherwise seeds come from "
            "config/expansion.yaml and are never replaced)"
        ),
    )
    seed_override.add_argument(
        "--seed", action="append", default=None,
        help="repeat for each explicit seed; preserves names containing commas",
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
    view = subparsers.add_parser("view", help="build an offline HTML graph explorer")
    view.add_argument("--config", type=Path, default=Path("config/ingest.yaml"))
    view.add_argument("--input", required=True, type=Path, help="existing SChain graph JSON")
    view.add_argument("--output", type=Path, default=Path("outputs/graph-explorer.html"))

    evaluate = subparsers.add_parser("evaluate", help="score directed triples against a held-out gold set")
    evaluate.add_argument("--config", type=Path, default=Path("config/ingest.yaml"))
    evaluate.add_argument("--gold", required=True, type=Path)
    evaluate.add_argument("--output", type=Path, default=None)
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
    from .audit_trace import configure_run_audit
    from .graph_export import export_graph, export_path

    graph_path = export_path(config, args.output)
    audit = configure_run_audit(
        config, mode="export", graph_path=graph_path,
        input_path=None, neo4j_enabled=False,
    )
    error = None
    report = None
    try:
        path = export_graph(config, output=args.output, mode="export")
        report = {"graph_export": str(path)}
        if audit:
            report["audit_trace"] = str(audit.output_path)
        print(json.dumps(report, indent=2))
        return 0
    except Exception as caught:
        error = caught
        raise
    finally:
        if audit:
            audit.finalize(
                complete=error is None,
                status="completed" if error is None else "failed",
                summary=report,
                error=error,
            )


def _attach_graph_export(
    config: dict[str, Any], summary: dict[str, Any], mode: str
) -> dict[str, Any]:
    """Write the derived graph JSON after a successful ingest/expand run."""
    from .graph_export import export_graph

    summary = dict(summary)
    completion = dict(summary.get("completion") or {})
    completion["export_success"] = True
    summary["completion"] = completion
    path = export_graph(config, mode=mode, pipeline_summary=summary)
    summary["graph_export"] = str(path)
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    if args.mode == "view":
        from .viewer import main as viewer_main
        return viewer_main(["--input", str(args.input), "--output", str(args.output)])
    config = load_config(args.config)
    if args.mode == "evaluate":
        from .validation.gold import evaluate_file
        report = evaluate_file(args.gold, storage_path(config, "extracted_facts"))
        payload = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            from .checkpoint import atomic_write_json
            atomic_write_json(args.output, report)
        print(payload)
        return 0
    if args.mode == "validate":
        return _run_validate(args, config)
    if args.mode == "export":
        # Operate only on persisted artifacts — no LLM, Neo4j, or retrieval.
        return _run_export(args, config)

    # CLI flags override the config file. Seeds are replaced ONLY via --seeds.
    if args.resume:
        config.setdefault("runtime", {})["restart_from_scratch"] = False
        config["runtime"]["resume"] = True
    elif args.restart:
        config.setdefault("runtime", {})["restart_from_scratch"] = True
    if args.no_neo4j:
        config.setdefault("save", {})["enabled"] = False
    elif args.save_override is not None:
        config.setdefault("save", {})["enabled"] = bool(args.save_override)
    if args.experiment_id is not None:
        config.setdefault("checkpoint_store", {})["experiment_id"] = args.experiment_id
        config["checkpoint_store"]["enabled"] = True
    if args.checkpoint_store is not None:
        config.setdefault("checkpoint_store", {})["directory"] = str(args.checkpoint_store)
        config["checkpoint_store"]["enabled"] = True
    if args.mode == "expand":
        if args.max_tier is not None:
            config.setdefault("expansion", {})["max_tier"] = args.max_tier
        if args.max_documents is not None:
            config.setdefault("expansion", {})["max_documents"] = args.max_documents
        if getattr(args, "seeds", None) or getattr(args, "seed", None):
            names = args.seed or next(csv.reader([args.seeds], skipinitialspace=True))
            names = [name.strip() for name in names if name.strip()]
            if names:
                config["seeds"] = {"cli_override": names}
                logger.warning(
                    "explicit --seeds override active (%d seed(s)); "
                    "expansion.yaml seeds were replaced intentionally",
                    len(names),
                )
        log_effective_expansion_config(config)
        validate_expansion_config(config)

    save = save_settings(config)
    checkpoint_manager = DurableCheckpointManager.from_config(config, mode=args.mode)
    checkpoint_status = None
    if checkpoint_manager is not None:
        try:
            checkpoint_status = checkpoint_manager.prepare(
                new_run=bool(config.get("runtime", {}).get("restart_from_scratch")),
                force_unlock=bool(args.force_checkpoint_unlock),
            )
        except Exception as error:  # noqa: BLE001 - configuration/recovery boundary
            logger.error("checkpoint preparation failed: %s", error)
            return 2

    from .audit_trace import configure_run_audit
    from .graph_export import export_path
    try:
        if config.get("runtime", {}).get("restart_from_scratch"):
            removed = clear_generated_data(config)
            logger.warning("restart: removed %d generated file(s)", len(removed))
        audit = configure_run_audit(
            config,
            mode=args.mode,
            graph_path=export_path(config),
            input_path=getattr(args, "input", None),
            neo4j_enabled=save["enabled"],
        )
    except Exception as error:  # noqa: BLE001 - release an acquired durable lock
        if checkpoint_manager is not None:
            checkpoint_manager.release()
        logger.error("run setup failed: %s", error)
        return 2
    connection = None
    driver = None
    summary = None
    run_error = None
    try:
        ontology = load_ontology(config)
        connection = state.connect(storage_path(config, "state_db"))
        llm = build_llm(config)
        if save["enabled"]:
            driver = open_driver(config)
        if args.mode == "ingest":
            summary = run_ingest(
                config, connection, llm, ontology, driver, args.input,
                checkpoint_manager=checkpoint_manager,
            )
        else:
            summary = run_expand(
                config, connection, llm, ontology, driver,
                checkpoint_manager=checkpoint_manager,
            )
        if checkpoint_manager is not None:
            summary["checkpoint_store"] = {
                **checkpoint_manager.status(), **(checkpoint_status or {}),
            }
        summary = _attach_graph_export(config, summary, args.mode)
    except LLMGenerationError as error:
        run_error = error
        logger.error(
            "LLM call failed (%s). Check that the server at %s is running and that "
            "llm.model matches what it serves.",
            error, config.get("llm", {}).get("base_url", "the configured base_url"),
        )
        return 1
    except Exception as error:  # noqa: BLE001 - report what failed, then stop
        run_error = error
        logger.error("%s failed: %s: %s", args.mode, type(error).__name__, error)
        return 1
    finally:
        if audit is not None:
            status = (
                "failed" if run_error is not None
                else "completed_with_document_failures"
                if summary and summary.get("documents_failed")
                else "completed"
            )
            try:
                audit.finalize(
                    complete=(
                        run_error is None
                        and not (summary and summary.get("documents_failed"))
                    ),
                    status=status,
                    summary=summary,
                    error=run_error,
                )
                if summary is not None:
                    summary["audit_trace"] = str(audit.output_path)
            except Exception:  # noqa: BLE001 - graph result must survive audit finalization
                logger.exception("could not finalize audit trace %s", audit.output_path)
        if checkpoint_manager is not None and connection is not None:
            try:
                generation = checkpoint_manager.snapshot(connection, reason="run_finalized")
                if summary is not None:
                    summary.setdefault("checkpoint_store", {}).update(
                        checkpoint_manager.status()
                    )
                    summary["checkpoint_store"]["last_snapshot"] = generation
            except Exception as error:  # noqa: BLE001 - durability failure is a run failure
                logger.error("final durable checkpoint failed: %s", error)
                run_error = run_error or error
        if driver is not None:
            driver.close()
        if connection is not None:
            connection.close()
        if checkpoint_manager is not None:
            checkpoint_manager.release()

    if run_error is not None:
        return 1

    logger.info(
        "done documents=%s facts=%s nodes=%s relationships=%s frontier=%s "
        "graph_export=%s elapsed=%ss",
        summary.get("documents_processed", summary.get("documents")),
        summary.get("facts"),
        summary.get("nodes_merged", summary.get("nodes")),
        summary.get("relationships_merged", summary.get("relationships")),
        summary.get("frontier"),
        summary.get("graph_export"),
        summary.get("elapsed_seconds"),
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
