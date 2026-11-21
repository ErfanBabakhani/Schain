"""Metrics and blind review exports from persisted benchmark results."""

from collections import defaultdict, deque
from contextlib import closing
import csv
import errno
import io
import json
import os
import sqlite3
from statistics import mean, stdev
import time
from uuid import uuid4

import pandas as pd

from . import state
from .benchmark_inputs import digest
from .benchmark_selection import STATE_KEY
from .checkpoint import atomic_write_json
from .kg.support import contextualize_facts, frontier_witness
from .names import canonical_id


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def company_alias_ids(ontology):
    return {canonical_id("Company", alias): canonical_id("Company", name)
            for alias, name in (ontology.get("aliases", {}).get("Company") or {}).items()}


def resolve_id(identifier, aliases):
    seen = set()
    while identifier in aliases and identifier not in seen:
        seen.add(identifier)
        identifier = aliases[identifier]
    return min(seen) if identifier in seen else identifier


def relationships(facts, aliases, *, require_witness):
    """Group accepted, directional Company→Company witnesses by canonical endpoints."""
    edges = defaultdict(list)
    for row in facts.to_dict(orient="records"):
        if (row.get("relation") != "suppliesTo" or row.get("subject_type") != "Company"
                or row.get("object_type") != "Company" or str(row.get("doc_id", "")).startswith("verify:")):
            continue
        endpoints = []
        for role in ("subject", "object"):
            identifier = str(row.get(f"{role}_id") or "")
            name = row.get(f"{role}_name")
            name_id = canonical_id("Company", name) if isinstance(name, str) else identifier
            endpoints.append(resolve_id(name_id if name_id in aliases else identifier, aliases))
        supplier, buyer = endpoints
        if supplier == buyer or not all(key.startswith("company:") and key[8:] for key in endpoints):
            continue
        if require_witness and not frontier_witness(row)[2]:
            continue
        quote = row.get("evidence")
        if require_witness and (not isinstance(quote, str) or not quote.strip()):
            continue
        edges[supplier, buyer].append(row)
    return edges


def tier_stats(edges, seeds):
    by_buyer = defaultdict(set)
    for supplier, buyer in edges:
        by_buyer[buyer].add(supplier)
    tiers = {seed: 0 for seed in seeds}
    pending = deque(sorted(seeds))
    while pending:
        buyer = pending.popleft()
        for supplier in sorted(by_buyer[buyer]):
            if supplier not in tiers:
                tiers[supplier] = tiers[buyer] + 1
                pending.append(supplier)
    counts = {str(tier): 0 for tier in range(6)}
    for tier in tiers.values():
        counts[str(tier)] = counts.get(str(tier), 0) + 1
    return {"company_counts_by_tier": counts,
            "tiers_2_to_5_present": {str(tier): counts[str(tier)] > 0 for tier in range(2, 6)},
            "tiers_2_to_5_coverage": sum(counts[str(tier)] > 0 for tier in range(2, 6)) / 4}


def session_status(run_dir, session):
    elapsed = session.get("elapsed_seconds", 0.0)
    if session.get("stop_reason"):
        return session["stop_reason"], elapsed
    if not session:
        return "not_started", elapsed
    lock_path = run_dir / ".lock"
    if lock_path.exists():
        try:
            with lock_path.open("rb") as handle:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBRLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return "active", elapsed + max(0.0, time.time() - session["started_at"])
            raise
    stamps = [(run_dir / name).stat().st_mtime for name in ("state.db", "state.db-wal")
              if (run_dir / name).exists()]
    return "external_interruption", elapsed + max(0.0, max(stamps, default=0) - session["started_at"])


def collect(run_dir):
    manifest = read_json(run_dir / "experiment.json")
    status_file = run_dir / "session.json"
    session = read_json(status_file) if status_file.exists() else {}
    database = run_dir / "state.db"
    journal, ledger, events, full_decisions = {}, {}, [], 0
    resolutions, origins = {}, {}
    if database.exists():
        with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "expansion_run_state" in tables:
                run_id = state.run_value(db, "run_id") or manifest["run_id"]
                ledger = state.run_value(db, f"input_budget:{run_id}", {})
                journal = state.run_value(db, STATE_KEY, {})
                full_decisions = state.run_value(db, f"adaptive_candidate_decisions:{run_id}", 0)
            if "benchmark_events" in tables:
                events = [dict(row) for row in db.execute("SELECT * FROM benchmark_events ORDER BY rowid")]
            if "entity_resolution" in tables:
                resolutions = {
                    canonical_id("Company", row["mention"]): row["canonical_id"] for row in db.execute(
                        "SELECT mention, canonical_id FROM entity_resolution WHERE label='Company'"
                    )
                }
            if "document_origins" in tables:
                origins = state.document_origins(db)
    config = manifest["effective_config"]
    aliases = {**resolutions, **company_alias_ids(config["ontology"])}
    seeds = {resolve_id(row[1], aliases) for row in manifest["targets"]}
    facts_path = run_dir / "facts.parquet"
    facts = pd.read_parquet(facts_path) if facts_path.exists() else pd.DataFrame()
    initial_path = run_dir / "initial_facts.parquet"
    initial = pd.read_parquet(initial_path) if initial_path.exists() else pd.DataFrame()
    existing = relationships(initial, aliases, require_witness=False)
    supported = relationships(contextualize_facts(facts, origins), aliases, require_witness=True)
    mode = manifest["mode"]
    discovered = {
        endpoints: rows for endpoints, rows in supported.items()
        if endpoints not in existing and (mode == "full-pipeline" or endpoints[1] in seeds)
    }
    doc_phases = {}
    for event in events:
        if event["outcome"] in ("selected", "processed") and event["doc_id"]:
            doc_phases.setdefault(event["doc_id"], event["phase"])
    phase_yield = defaultdict(int)
    edges = []
    for (supplier, buyer), rows in sorted(discovered.items()):
        # Parquet preserves insertion order. Attribute each edge to its first qualifying witness.
        phase_yield[doc_phases.get(rows[0].get("doc_id"), "unassigned")] += 1
        evidence = []
        for row in rows:
            witness = {field: row[field] for field in ("evidence", "source_url", "document_title", "content_source")
                       if isinstance(row.get(field), str)}
            if witness not in evidence:
                evidence.append(witness)
        edges.append({"supplier_id": supplier, "relation": "suppliesTo", "buyer_id": buyer, "evidence": evidence})
    counts = defaultdict(set)
    for event in events:
        identity = event["candidate_id"]
        counts[event["outcome"]].add(identity)
        if event["outcome"] == "fetch_failed":
            counts["selected"].add(identity)
    spent = ledger.get("spent_estimate", 0)
    covered = {buyer for _, buyer in discovered} & seeds
    adaptive = manifest["policy"] == "adaptive_linucb"
    warmup = manifest["warmup_candidates"] if adaptive else None
    decisions = journal.get("decisions", full_decisions) if adaptive else None
    phases = {}
    for phase in ("warmup", "greedy", "overall", "unassigned"):
        counts_phase = defaultdict(set)
        for event in events:
            if event["phase"] == phase:
                counts_phase[event["outcome"]].add(event["candidate_id"])
        bucket = ledger.get("phases", {}).get(phase, {})
        tokens = bucket.get("input_tokens", 0)
        label = "post_warmup" if phase == "greedy" else phase
        phases[label] = {
            "input_tokens": tokens, "requests": bucket.get("requests", 0),
            "selected": len(counts_phase["selected"] | counts_phase["fetch_failed"]),
            "processed": len(counts_phase["processed"]), "skipped": len(counts_phase["skipped"]),
            "failed": len((counts_phase["failed"] | counts_phase["fetch_failed"]) - counts_phase["processed"]),
            "new_supplier_relationships": phase_yield[phase],
            "new_relationships_per_10000_input_tokens": 10000 * phase_yield[phase] / tokens if tokens else None,
        }
    stop, elapsed = session_status(run_dir, session)
    failed = (counts["failed"] | counts["fetch_failed"]) - counts["processed"]
    processing_failed = counts["failed"] - counts["processed"]
    if stop in ("candidate_exhaustion", "frontier_exhaustion") and processing_failed:
        stop = "candidate_errors"
    complete = (stop in ("candidate_exhaustion", "token_budget", "document_budget", "frontier_exhaustion")
                and not processing_failed)
    report = {
        **manifest, "stop_reason": stop, "complete": complete,
        "error": session.get("error"),
        "elapsed_seconds": elapsed,
        "input_tokens": spent, "output_tokens": ledger.get("output_tokens"),
        "responses_with_output_usage": ledger.get("responses_with_output_usage", 0),
        "requests": ledger.get("requests", 0),
        "remaining_input_tokens": manifest["input_token_budget"] - spent,
        "first_rejected_request": ledger.get("first_rejection"),
        "candidates_considered": len(counts["considered"]), "candidates_selected": len(counts["selected"]),
        "candidates_skipped": len(counts["skipped"] | counts["ineligible"] | counts["duplicate"]),
        "candidates_successfully_processed": len(counts["processed"]), "candidates_failed": len(failed),
        "new_supplier_relationships": len(discovered),
        "new_relationships_per_10000_input_tokens": 10000 * len(discovered) / spent if spent else None,
        "target_count": len(seeds), "targets_with_supplier": len(covered),
        "target_coverage": len(covered) / len(seeds) if seeds else None,
        "candidate_decisions": decisions, "warmup_complete": decisions >= warmup if adaptive else None,
        "learned_performance_insufficient": (
            decisions < warmup or phases["post_warmup"]["selected"] == 0 if adaptive else None
        ),
        "phases": phases, "human_reviewed_precision": None,
        "relationships": edges, "pipeline_summary": session.get("pipeline_summary"),
    }
    if mode == "full-pipeline":
        report.update(tier_stats(supported, seeds))
    tokenizer_path = run_dir / "tokenizer.json"
    if tokenizer_path.exists():
        report["tokenizer_config"] = read_json(tokenizer_path)
        report["comparison_sha256"] = digest([manifest["comparison_sha256"], report["tokenizer_config"]])
    return report


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple))
                         else value for key, value in row.items()})
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="")


def labels_for(path, valid_ids):
    labels = {}
    if path is None:
        return labels
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"review_id", "label"} <= set(reader.fieldnames or []):
            raise ValueError("review labels must contain review_id,label columns")
        for row in reader:
            review_id, label = row["review_id"], row["label"].strip().lower()
            if review_id not in valid_ids or review_id in labels:
                raise ValueError(f"unknown or repeated review ID: {review_id}")
            if label not in {"", "unsure", "true", "false"}:
                raise ValueError(f"invalid review label for {review_id}: {label}")
            labels[review_id] = {"true": True, "false": False}.get(label)
    return labels


def aggregate(reports):
    groups = defaultdict(list)
    for report in reports:
        key = (report["comparison_sha256"], report["mode"], report["policy"])
        groups[key].append(report)
    output = []
    for (fingerprint, mode, policy), group in sorted(groups.items()):
        complete = [report for report in group if report["complete"]]
        metrics = {}
        for name in ("new_supplier_relationships", "new_relationships_per_10000_input_tokens",
                     "target_coverage", "input_tokens", "tiers_2_to_5_coverage"):
            values = [report[name] for report in complete if report.get(name) is not None]
            if values:
                metrics[name] = {"mean": mean(values), "sample_stddev": stdev(values) if len(values) > 1 else None,
                                 "repetitions": len(values)}
        output.append({
            "comparison_sha256": fingerprint, "mode": mode, "policy": policy,
            "repetitions": len(group), "completed_repetitions": len(complete),
            "seeds": sorted({run["random_seed"] for run in group}), "metrics": metrics,
        })
    return output


def paired_differences(reports):
    comparisons = defaultdict(dict)
    for report in reports:
        if report["complete"]:
            key = report["comparison_sha256"], report["mode"]
            comparisons[key].setdefault(report["random_seed"], {})[report["policy"]] = report
    output = []
    for (fingerprint, mode), by_seed in sorted(comparisons.items()):
        pairs = [(seed, arms["adaptive_linucb"], arms["legacy"])
                 for seed, arms in sorted(by_seed.items()) if {"adaptive_linucb", "legacy"} <= arms.keys()]
        metrics = {}
        for name in ("new_supplier_relationships", "new_relationships_per_10000_input_tokens",
                     "target_coverage", "input_tokens"):
            deltas = [a[name] - b[name] for _, a, b in pairs
                      if a.get(name) is not None and b.get(name) is not None]
            if deltas:
                metrics[name] = {"mean": mean(deltas), "sample_stddev": stdev(deltas) if len(deltas) > 1 else None,
                                 "pairs": len(deltas)}
        output.append({"comparison_sha256": fingerprint, "mode": mode,
                       "matched_seeds": [seed for seed, _, _ in pairs], "adaptive_minus_legacy": metrics})
    return output


def export_comparison(root, labels_path=None):
    runs = sorted(path.parent for path in root.glob("*/*/experiment.json"))
    if not runs:
        raise ValueError(f"no benchmark repetitions in {root}")
    reports = [collect(run) for run in runs]
    directory = root / "reports"
    directory.mkdir(exist_ok=True)
    mapping_path = directory / "review_mapping.private.json"
    previous = read_json(mapping_path) if mapping_path.exists() else []
    prior_ids = {(row["supplier_id"], row["buyer_id"]): row["review_id"] for row in previous}
    review = {}
    mapping = {}
    memberships = defaultdict(set)
    for report in reports:
        for edge in report["relationships"]:
            key = edge["supplier_id"], edge["buyer_id"]
            review_id = prior_ids.setdefault(key, uuid4().hex)
            entry = review.setdefault(review_id, {"review_id": review_id, "supplier_id": key[0],
                                                  "relation": "suppliesTo", "buyer_id": key[1], "evidence": []})
            for evidence in edge["evidence"]:
                if evidence not in entry["evidence"]:
                    entry["evidence"].append(evidence)
            mapping[review_id] = {"review_id": review_id, "supplier_id": key[0], "buyer_id": key[1]}
            memberships[review_id].add(report["run_id"])
    labels = labels_for(labels_path, set(review))
    for report, run in zip(reports, runs):
        reviewed = [labels[review_id] for review_id, members in memberships.items()
                    if report["run_id"] in members and labels.get(review_id) is not None]
        report["human_reviewed_precision"] = sum(reviewed) / len(reviewed) if reviewed else None
        report["human_labeled_relationships"] = len(reviewed)
        report["human_correct_relationships"] = sum(reviewed)
        atomic_write_json(run / "report.json", report)
        write_csv(run / "report.csv", [{key: value for key, value in report.items() if key != "relationships"}])
    reviewer_rows = [review[key] for key in sorted(review)]
    for row in reviewer_rows:
        row["evidence"].sort(key=lambda item: json.dumps(item, sort_keys=True))
    atomic_write_json(mapping_path, [{**mapping[key], "run_ids": sorted(memberships[key])}
                                     for key in sorted(mapping)])
    atomic_write_json(directory / "review.json", reviewer_rows)
    write_csv(directory / "review.csv", [{**row, "label": ""} for row in reviewer_rows]
              or [{"review_id": "", "supplier_id": "", "relation": "", "buyer_id": "", "evidence": "", "label": ""}])
    summary = aggregate(reports)
    paired = paired_differences(reports)
    atomic_write_json(directory / "comparison.json", {
        "repetitions": reports, "summary": summary, "paired_differences": paired,
    })
    write_csv(directory / "repetitions.csv", [{key: value for key, value in row.items() if key != "relationships"}
                                               for row in reports])
    write_csv(directory / "summary.csv", summary)
    write_csv(directory / "paired_differences.csv", paired)
    return directory
