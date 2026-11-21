"""Explicit prepare, dry-run, run, and report commands for policy comparison."""

import argparse
from contextlib import closing, contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import time
from uuid import uuid4

from . import state
from .benchmark_inputs import (
    MODES, POLICIES, PRIMARY, Replay, digest, initial_bytes, load_dataset,
    record_dataset, result_key, retrieval_fingerprint, targets_for,
)
from .benchmark_report import collect, export_comparison, read_json, write_csv
from .benchmark_selection import run_selection
from .checkpoint import atomic_write_json
from .config import REPO_ROOT, load_config, validate_expansion_config
from .documents import normalize_url
from .expansion.input_budget import budgeted_llm
from .expansion.loop import run_expand
from .llm.provider import InputTokenBudgetExceeded, build_llm, load_tokenizer
from .names import canonical_id
from .ontology import load_ontology
from .run_config import _policy_hash, _safe, _safe_config

logger = logging.getLogger(__name__)


class SessionTimeLimit(Exception):
    pass


def arm_config(config_path, mode, policy, seed=None, warmup=None, budget=None, batch_size=None):
    config = load_config(config_path)
    expansion = config["expansion"]
    selection = expansion.setdefault("selection", {})
    selection["policy"] = policy
    for key, provided, default in (
        ("total_input_tokens", budget, 200000), ("random_seed", seed, 42),
        ("warmup_candidates", warmup, 200), ("snippet_batch_size", batch_size, 10),
    ):
        selection[key] = int(selection.get(key, default) if provided is None else provided)
    selection.setdefault("target_tokens_per_candidate", 1000)
    retrieval = expansion.setdefault("retrieval", {})
    if mode == "selection-only" or policy == "adaptive_linucb":
        expansion.update(max_documents=None, max_documents_per_run=None)
        retrieval.update(target_documents_per_entity=0, max_documents_per_entity=None,
                         target_documents_by_tier={}, max_documents_by_tier={})
    if mode == "selection-only":
        selection.update(allowed_actions=["snippet"], candidate_pool_policy="title_snippet")
        retrieval.update(zero_yield_rescue=False, max_candidates_per_domain=None)
        config.setdefault("runtime", {})["max_minutes"] = 0
    config["runtime"].update(resume=True, restart_from_scratch=False)
    config["llm"].update(client_max_retries=0, rate_limit_max_attempts=1)
    config["save"] = {"enabled": False}
    config["verification"] = {"enabled": False}
    config["distributed"] = {"enabled": False}
    config["state"] = {"backend": "sqlite"}
    config["checkpoint_store"] = {"enabled": False}
    config["_benchmark_offline"] = True
    if (selection["random_seed"] < 0 or selection["warmup_candidates"] < 0
            or not 1 <= selection["snippet_batch_size"] <= 10):
        raise ValueError("seed and warmup must be nonnegative, batch size must be 1–10")
    validate_expansion_config(config)
    return config


def isolate(config, directory):
    # Snapshot, Parquet files, logs, trace and all derived caches live inside one arm.
    artifacts = {
        "raw_documents": "documents.jsonl", "normalized_documents": "documents.parquet",
        "extracted_facts": "facts.parquet", "extractions": "extractions",
        "search_results": "search_results.parquet", "entity_resolution": "resolution.parquet",
        "rejected_facts": "rejected.parquet", "edge_support": "support.parquet",
        "verification_evidence": "verification.parquet", "extraction_integrity": "integrity.json",
        "state_db": "state.db", "graph_export": "graph.json", "logs": "logs",
        "run_config": "run_config.json", "audit_trace": "audit",
    }
    config["storage"] = {key: str(directory / name) for key, name in artifacts.items()}
    config["audit"] = {"enabled": False}


def configs_for(args):
    return {policy: arm_config(args.config, args.mode, policy, getattr(args, "seed", None),
                               getattr(args, "warmup", None), getattr(args, "budget", None),
                               getattr(args, "batch_size", None)) for policy in POLICIES}


def inventory_for(targets, entries, ontology, max_tier):
    aliases = {canonical_id("Company", alias): name
               for alias, name in (ontology.get("aliases", {}).get("Company") or {}).items()}
    inventory = {(key, tier): (name, key, tier) for name, key, tier in targets}
    for entry in entries:
        tier_text, sep, name = entry.partition(":")
        if not sep or not name.strip() or not tier_text.isdecimal():
            raise ValueError("--company must use TIER:NAME")
        name, tier = name.strip(), int(tier_text)
        if not 0 <= tier < max_tier:
            raise ValueError(f"recorded inventory must use tiers below {max_tier}")
        key = canonical_id("Company", aliases.get(canonical_id("Company", name), name))
        inventory.setdefault((key, tier), (name, key, tier))
    return list(inventory.values())


def git_provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL)

    commit = git("rev-parse", "HEAD").decode().strip()
    names = set(git("ls-files", "-z", "--cached", "--others", "--exclude-standard",
                    "--", "schain", "config", "pyproject.toml").decode().split("\0"))
    hashes = {
        name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
        for name in sorted(names) if name and (REPO_ROOT / name).is_file()
    }
    return {"git_commit": commit, "working_source_sha256": digest(hashes)}


def validate_inputs(archive, mode, configs, targets):
    if archive["mode"] != mode:
        raise ValueError("dataset was prepared for a different experimental mode")
    if archive["targets"] != [list(target) for target in targets] and archive["targets"] != targets:
        raise ValueError("configured targets differ from the frozen dataset")
    if retrieval_fingerprint(configs, mode) != archive["retrieval_config_sha256"]:
        raise ValueError("retrieval settings differ from the frozen dataset")
    profiles = ("shared",) if mode == "selection-only" else POLICIES
    for profile in profiles:
        replay = Replay(archive, profile)
        for _, key, tier in targets:
            replay.entry(key, tier, PRIMARY)


@contextmanager
def run_lock(directory):
    with (directory / ".lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def event_sink(connection):
    connection.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_events (
            candidate_id TEXT, outcome TEXT, phase TEXT, doc_id TEXT, detail TEXT,
            PRIMARY KEY(candidate_id, outcome)
        )
    """)
    connection.commit()

    def log(entity, tier, result, outcome, *, phase=None, doc_id=None, **detail):
        source = normalize_url(result.canonical_source_url or result.url) or result_key(result)
        candidate_id = digest([entity, tier, source])
        with connection:
            connection.execute(
                "INSERT OR IGNORE INTO benchmark_events VALUES(?,?,?,?,?)",
                (candidate_id, outcome, phase, doc_id, json.dumps(detail)),
            )
    return log


def run_arm(args, *, llm_factory=None, tokenizer_factory=None, interrupted=lambda: False):
    configs = configs_for(args)
    config = configs[args.policy]
    archive = load_dataset(args.dataset)
    ontology = load_ontology(config)
    targets = targets_for(config, ontology)
    validate_inputs(archive, args.mode, configs, targets)
    settings = config["expansion"]["selection"]
    seed, budget = settings["random_seed"], settings["total_input_tokens"]
    model = config["llm"].get("model")
    tokenizer_model = config["llm"].get("tokenizer_model") or model
    if not tokenizer_model or not model:
        raise ValueError("configure the model and its matching tokenizer")
    run_dir = (args.workspace / args.mode / f"seed-{seed}-{args.policy}").resolve()
    isolate(config, run_dir)
    provenance = git_provenance()
    comparison_settings = {
        policy: {**_safe_config(arm), "relation_policy_sha256": _policy_hash(arm)}
        for policy, arm in configs.items()
    }
    for arm in comparison_settings.values():
        arm["expansion"]["selection"].pop("random_seed")
    manifest = {
        **provenance, "mode": args.mode, "policy": args.policy, "random_seed": seed,
        "warmup_candidates": settings["warmup_candidates"], "input_token_budget": budget,
        "dataset_version": archive["dataset_version"], "dataset_sha256": archive["dataset_sha256"],
        "initial_graph_sha256": (archive["initial_facts"] or {}).get("sha256"),
        "comparison_sha256": digest([provenance, archive["dataset_sha256"], comparison_settings]),
        "targets": targets, "model": model, "tokenizer_model": tokenizer_model,
        "effective_config": {**_safe_config(config), **_safe({
            key: config[key] for key in ("storage", "save", "verification", "distributed", "state",
                                        "checkpoint_store", "audit")
        })},
        "relation_policy_sha256": _policy_hash(config),
        "shared_preparation": archive["preparation"],
    }
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        run_dir.mkdir()
    elif not (run_dir / "experiment.json").is_file():
        raise ValueError(f"no benchmark repetition to resume in {run_dir}")
    with run_lock(run_dir):
        path = run_dir / "experiment.json"
        session_path = run_dir / "session.json"
        previous = read_json(session_path) if session_path.exists() else {}
        if path.exists():
            saved = read_json(path)
            if digest(manifest) != saved["identity_sha256"]:
                raise ValueError("benchmark inputs, configuration, or code changed since this repetition started")
            if previous.get("stop_reason") in {"candidate_exhaustion", "frontier_exhaustion",
                                               "token_budget", "document_budget"}:
                return collect(run_dir)
            run_id = saved["run_id"]
        else:
            run_id = uuid4().hex
            data = initial_bytes(archive)
            if data is not None:
                (run_dir / "initial_facts.parquet").write_bytes(data)
                (run_dir / "facts.parquet").write_bytes(data)
            atomic_write_json(path, {**manifest, "run_id": run_id, "identity_sha256": digest(manifest)})
        started = time.monotonic()
        elapsed_before = previous.get("elapsed_seconds", 0.0)
        if previous.get("started_at") and not previous.get("stop_reason"):
            stamps = [(run_dir / name).stat().st_mtime for name in ("state.db", "state.db-wal")
                      if (run_dir / name).exists()]
            elapsed_before += max(0.0, max(stamps, default=0) - previous["started_at"])
        atomic_write_json(session_path, {"elapsed_seconds": elapsed_before, "started_at": time.time()})
        stop, error_message, pipeline_summary = "error", None, None

        def time_limit_reached():
            return bool(args.session_minutes and time.monotonic() - started >= args.session_minutes * 60)

        def check_time():
            if interrupted():
                raise KeyboardInterrupt
            if time_limit_reached():
                raise SessionTimeLimit("benchmark session limit reached")

        try:
            tokenizer = (tokenizer_factory(tokenizer_model) if tokenizer_factory else
                         load_tokenizer(tokenizer_model, local_files_only=True))
            if not getattr(tokenizer, "chat_template", None):
                raise ValueError("benchmark requires a tokenizer with a chat template")
            template_path = run_dir / "tokenizer.json"
            template = {"chat_template_sha256": digest(tokenizer.chat_template),
                        "model_max_length": getattr(tokenizer, "model_max_length", None)}
            backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
            if backend_tokenizer is not None:
                template["tokenizer_sha256"] = digest(backend_tokenizer.to_str())
            elif hasattr(tokenizer, "get_vocab"):
                template["tokenizer_sha256"] = digest([tokenizer.get_vocab(), tokenizer.special_tokens_map])
            if template_path.exists() and read_json(template_path) != template:
                raise ValueError("configured tokenizer chat template or context size changed")
            atomic_write_json(template_path, template)
            llm = (llm_factory or build_llm)(config)
            with closing(state.connect(run_dir / "state.db")) as db:
                state.set_run_value(db, "run_id", run_id)
                profile = "shared" if args.mode == "selection-only" else args.policy
                replay = Replay(archive, profile, tokenizer=tokenizer, event=event_sink(db))
                if args.mode == "selection-only":
                    llm = budgeted_llm(config, db, run_id, llm, tokenizer, tokenizer_model, strict=True)
                    stop = run_selection(config, db, llm, ontology, replay, targets, run_id,
                                         check_time=check_time)
                else:
                    pipeline_summary = run_expand(
                        config, db, llm, ontology, retrieval_checkpoint=replay, benchmark=replay,
                        should_pause=lambda: interrupted() or time_limit_reached(),
                    )
                    raw_stop = pipeline_summary["stop_reason"]
                    stop = {
                        "frontier_empty": "frontier_exhaustion", "runtime_budget": "time_limit",
                        "runtime_reserve": "time_limit", "pause_requested": "time_limit",
                    }.get(raw_stop, raw_stop or "error")
                    if pipeline_summary["completion"].get("failed_frontier") and stop == "frontier_exhaustion":
                        stop = "candidate_errors"
                    if interrupted() and stop == "time_limit":
                        stop = "external_interruption"
        except InputTokenBudgetExceeded:
            stop = "token_budget"
        except SessionTimeLimit:
            stop = "time_limit"
        except KeyboardInterrupt:
            stop = "external_interruption"
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
            logger.exception("benchmark repetition failed")
        finally:
            atomic_write_json(session_path, {"stop_reason": stop, "error": error_message,
                                            "elapsed_seconds": elapsed_before + time.monotonic() - started,
                                            "pipeline_summary": pipeline_summary})
        report = collect(run_dir)
        atomic_write_json(run_dir / "report.json", report)
        write_csv(run_dir / "report.csv", [{key: value for key, value in report.items() if key != "relationships"}])
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m schain.benchmark", description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "dry-run", "run"):
        action = actions.add_parser(command)
        action.add_argument("--mode", choices=MODES, required=True)
        action.add_argument("--config", type=Path, default=REPO_ROOT / "config/selection_benchmark.yaml")
        action.add_argument("--dataset", type=Path, required=True)
        if command == "prepare":
            action.add_argument("--version", required=True)
            action.add_argument("--company", action="append", default=[], metavar="TIER:NAME")
            action.add_argument("--initial-facts", type=Path)
            action.add_argument("--allow-search-errors", action="store_true")
        else:
            action.add_argument("--seed", type=int)
            action.add_argument("--warmup", type=int)
            action.add_argument("--budget", type=int)
            action.add_argument("--batch-size", type=int)
        if command == "run":
            action.add_argument("--policy", choices=POLICIES, required=True)
            action.add_argument("--workspace", type=Path, required=True)
            action.add_argument("--resume", action="store_true")
            action.add_argument("--session-minutes", type=float, default=0)
    report_parser = actions.add_parser("report")
    report_parser.add_argument("--workspace", type=Path, required=True)
    report_parser.add_argument("--labels", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        if args.command == "report":
            print(export_comparison(args.workspace.resolve(), args.labels))
            return 0
        if args.command == "run":
            import math

            if not math.isfinite(args.session_minutes) or args.session_minutes < 0:
                raise ValueError("session minutes must be finite and nonnegative")

            requested = False

            def interrupt(*_):
                nonlocal requested
                if requested:
                    raise KeyboardInterrupt
                requested = True

            original = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                report = run_arm(args, interrupted=lambda: requested)
            finally:
                for sig, handler in original.items():
                    signal.signal(sig, handler)
            print(f"{report['policy']} seed {report['random_seed']}: {report['stop_reason']}")
            return 0 if report["complete"] else 1
        configs = configs_for(args)
        config = configs["legacy"]
        ontology = load_ontology(config)
        targets = targets_for(config, ontology)
        if not targets:
            raise ValueError("configure at least one target seed")
        if args.command == "prepare":
            inventory = inventory_for(targets, args.company, ontology, int(config["expansion"]["max_tier"]))
            if args.mode == "selection-only" and args.company:
                raise ValueError("--company is only for the full-pipeline search inventory")
            record_dataset(args.dataset.resolve(), args.mode, configs, targets, inventory, args.version,
                           initial_facts=args.initial_facts, allow_search_errors=args.allow_search_errors,
                           provenance=git_provenance())
            print(args.dataset.resolve())
        else:
            archive = load_dataset(args.dataset)
            validate_inputs(archive, args.mode, configs, targets)
            print(f"{args.mode}: {archive['dataset_sha256']}")
            print(f"{len(targets)} seed targets, {len(archive['entries'])} frozen retrieval entries")
            print(f"{config['expansion']['selection']['total_input_tokens']} input tokens per repetition")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
