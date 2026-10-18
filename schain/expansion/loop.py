"""The EXPAND loop.

    seeds -> frontier -> tier queries -> retrieval -> origin filter -> rank
          -> process multiple docs (per-entity + global budgets)
          -> Prompt 1 -> Prompt 2 -> validate -> facts -> edge_support
          -> frontier from expansion-relevant suppliesTo -> next tier

The frontier lives in SQLite, so an interrupted run continues where it stopped instead of
restarting from the seeds.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from .. import state
from ..config import REPO_ROOT, seed_entities, storage_path
from ..run_config import ensure_run_snapshot
from ..kg.relation_evidence import (
    classify_supply_direction,
    UPSTREAM_EXPLICIT,
    DOWNSTREAM_EXPLICIT,
    DIRECTION_AMBIGUOUS,
    load_relation_evidence,
)
from .budget import (
    entity_document_limits,
    breadth_caps,
    remaining_target_reserve,
    documents_per_run_limit,
    scheduling_batch_sizes,
)
from .telemetry import (
    empty_prompt1_drop_counts,
    graph_growth_snapshot,
    low_discovery_yield_warning,
    premature_discovery_exhaustion_warning,
    rejection_category_counts,
)
from ..documents import make_document
from ..kg.facts import FACT_COLUMNS
from ..kg.support import load_edge_support, refresh_edge_support
from ..pipeline import process_document
from ..sources import manifest as mf
from ..sources.retrieve import document_kwargs_from_result, retrieve_for_entity
from ..sources.ranking import discovery_candidate_is_meaningful
from ..sources.origin import origin_name_variants
from .frontier import discover_upstream, seed_frontier

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONSECUTIVE_FAILURES = 5


def _facts_for_entity(config: dict[str, Any], connection, entity_key: str) -> pd.DataFrame:
    """Facts extracted from the documents retrieved for this entity."""
    path = storage_path(config, "extracted_facts")
    if not path.exists():
        return pd.DataFrame(columns=FACT_COLUMNS)
    doc_ids = state.document_ids_for_entity(connection, entity_key)
    if not doc_ids:
        return pd.DataFrame(columns=FACT_COLUMNS)
    frame = pd.read_parquet(path)
    return frame[frame.doc_id.isin(doc_ids)].reset_index(drop=True)


def _entity_document_limits(expansion: dict[str, Any], tier: int = 0) -> tuple[int, int]:
    """Compatibility wrapper around the tier-aware policy."""
    return entity_document_limits(expansion, tier)


def _quality_summary(config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Build the CLI quality block from persisted artifacts."""
    facts_path = storage_path(config, "extracted_facts")
    rejected_path = storage_path(config, "rejected_facts")
    support = load_edge_support(config)
    facts_n = len(pd.read_parquet(facts_path)) if facts_path.exists() else 0
    rejected_n = len(pd.read_parquet(rejected_path)) if rejected_path.exists() else 0
    status_counts: dict[str, int] = {}
    frontier_eligible = 0
    expansion_edges = 0
    if not support.empty and "quality_status" in support.columns:
        for status, count in support.quality_status.value_counts().items():
            status_counts[str(status)] = int(count)
        frontier_eligible = int(support.frontier_eligible.fillna(False).sum())
        if "expansion_relevant" in support.columns:
            expansion_edges = int(support.expansion_relevant.fillna(False).sum())

    known = unknown = full_text = snippet = 0
    if facts_path.exists() and facts_n:
        facts = pd.read_parquet(facts_path)
        if "source_known" in facts.columns:
            known_flags = facts.source_known.eq(True)
            known = int(known_flags.sum())
            unknown = int((~known_flags).sum())
        if "content_source" in facts.columns:
            full_text = int((facts.content_source == "full_text").sum())
            snippet = int((facts.content_source == "title_snippet").sum())

    return {
        "discovery_documents": summary.get("documents_processed_cumulative", summary.get("documents_processed", 0)),
        "discovery_document_budget": documents_per_run_limit(config.get("expansion") or {}) or 0,
        "documents_processed_this_run": summary.get("documents_processed_this_run", summary.get("documents_processed", 0)),
        "lifetime_documents_processed": summary.get("lifetime_documents_processed",
                                                   summary.get("documents_processed_cumulative", 0)),
        "facts_accepted": facts_n,
        "facts_rejected": rejected_n,
        "accepted_unique_facts": facts_n,
        "rejected_unique_facts": rejected_n,
        "semantic_edges": len(support) if support is not None and not support.empty else 0,
        "quality_status": status_counts,
        "frontier_eligible_edges": frontier_eligible,
        "expansion_relevant_edges": expansion_edges,
        "known_source_evidence_items": known,
        "unknown_source_evidence_items": unknown,
        "full_text_evidence": full_text,
        "snippet_only_evidence": snippet,
        "documents_by_origin_entity": summary.get("documents_by_origin") or {},
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "stop_reason": summary.get("stop_reason"),
    }


def _frontier_completion(
    connection,
    max_tier: int,
    stop_reason: Optional[str],
    *,
    lifetime_documents: int = 0,
    documents_this_run: int = 0,
) -> dict[str, Any]:
    """Experiment is complete only when no expandable PENDING/PROCESSING work remains."""
    counts = state.frontier_counts(connection)
    pending = int(state.pending_frontier_count(connection, max_tier))
    processing = int(counts.get(state.PROCESSING, 0) or 0)
    done = int(counts.get(state.DONE, 0) or 0)
    failed = int(counts.get(state.FAILED, 0) or 0)
    # Terminal rows (tier >= max_tier or expandable=0) must not block completion.
    terminal = int(connection.execute(
        "SELECT COUNT(*) FROM frontier WHERE expandable=0 OR tier>=?",
        (max_tier,),
    ).fetchone()[0])
    tiers = state.frontier_tier_counts(connection)
    highest = max((t for t in tiers if t is not None), default=0)
    complete = pending == 0 and processing == 0
    # Runtime / document ceilings pause work — they never mean experiment complete.
    if stop_reason in (
        "runtime_budget", "runtime_reserve", "max_documents", "document_budget",
    ):
        complete = False if (pending > 0 or processing > 0) else complete
    return {
        "complete": complete,
        "stop_reason": stop_reason,
        "pending_frontier": pending,
        "pending_expandable_frontier": pending,
        "terminal_frontier": terminal,
        "processing_frontier": processing,
        "done_frontier": done,
        "failed_frontier": failed,
        "highest_reached_tier": highest,
        "highest_discovery_tier": highest,
        "lifetime_documents_processed": lifetime_documents,
        "documents_processed_this_run": documents_this_run,
    }


def _persisted_graph_counts(config: dict[str, Any], connection) -> dict[str, int]:
    """Accurate graph-size signals from persisted facts/frontier (works with --no-neo4j)."""
    facts_path = storage_path(config, "extracted_facts")
    support = load_edge_support(config)
    accepted_facts = 0
    subject_object_ids: set[str] = set()
    if facts_path.exists():
        facts = pd.read_parquet(facts_path)
        accepted_facts = len(facts)
        if not facts.empty:
            for col in ("subject_id", "object_id"):
                if col in facts.columns:
                    subject_object_ids.update(str(v) for v in facts[col].dropna().tolist())
    tiered = sum(state.frontier_tier_counts(connection).values())
    frontier_ids = {
        str(row["entity_key"])
        for row in connection.execute("SELECT entity_key FROM frontier").fetchall()
    }
    total_nodes = len(subject_object_ids | frontier_ids) if subject_object_ids or frontier_ids else tiered
    context_nodes = max(0, total_nodes - tiered)
    expansion_edges = 0
    supply_core = 0
    total_edges = 0
    if support is not None and not support.empty:
        total_edges = len(support)
        if "expansion_relevant" in support.columns:
            expansion_edges = int(support.expansion_relevant.fillna(False).sum())
        if "frontier_eligible" in support.columns:
            supply_core = int(support.frontier_eligible.fillna(False).sum())
        elif expansion_edges:
            supply_core = expansion_edges
    return {
        "total_nodes": total_nodes,
        "tiered_nodes": tiered,
        "context_nodes": context_nodes,
        "total_edges": total_edges,
        "supply_chain_core_edges": supply_core,
        "expansion_relevant_edges": expansion_edges,
        "accepted_facts": accepted_facts,
    }


def print_quality_summary(quality: dict[str, Any]) -> None:
    status = quality.get("quality_status") or {}
    lines = [
        "SChain Quality Summary",
        "----------------------",
        f"Discovery documents       : {quality.get('discovery_documents')} / "
        f"{quality.get('discovery_document_budget') or '∞'}",
        f"Facts accepted            : {quality.get('facts_accepted')}",
        f"Facts rejected            : {quality.get('facts_rejected')}",
        f"Semantic edges            : {quality.get('semantic_edges')}",
        f"Accepted                  : {status.get('accepted', 0)}",
        f"Accepted (high reputation): {status.get('accepted_high_reputation', 0)}",
        f"Contested                 : {status.get('contested', 0)}",
        f"Frontier eligible edges   : {quality.get('frontier_eligible_edges')}",
        f"Expansion-relevant edges  : {quality.get('expansion_relevant_edges')}",
        f"Known-source evidence     : {quality.get('known_source_evidence_items')}",
        f"Unknown-source evidence   : {quality.get('unknown_source_evidence_items')}",
        f"Full-text evidence        : {quality.get('full_text_evidence')}",
        f"Snippet-only evidence     : {quality.get('snippet_only_evidence')}",
        f"Elapsed time              : {quality.get('elapsed_seconds')}s",
        f"Stop reason               : {quality.get('stop_reason')}",
    ]
    for line in lines:
        logger.info(line)


def _origin_edge_keys(config, entity_key: str) -> set[str]:
    support = load_edge_support(config)
    if support.empty:
        return set()
    mask = (support.object_id == entity_key) & (support.frontier_eligible == True)
    return set(support.loc[mask, "edge_key"].tolist())


def _set_rescue_mode(connection, entity_key: str, enabled: bool) -> None:
    connection.execute(
        "UPDATE frontier SET rescue_mode=? WHERE entity_key=?",
        (1 if enabled else 0, entity_key),
    )
    connection.commit()


def _admit_above_target(
    *,
    used: int,
    target: int,
    hard: int,
    direction: str,
    existing_keys: set[str],
    zero_gain: int,
    patience: int,
    min_docs: int,
    rescue_enabled: bool,
    connection,
    config: dict[str, Any],
    max_documents: int,
) -> tuple[bool, Optional[str], bool]:
    """Return (admit, stop_reason, rescue_active).

    TARGET is a normal allocation checkpoint, not an unconditional stop.
    MAX is the useful spend ceiling. Zero-yield entities escalate to MAX.
    Productive entities may continue above target when strong candidates remain
    and deeper-tier reservation allows it.
    """
    if not target or used < target:
        return True, None, False
    if hard and used >= hard:
        return False, "entity_document_cap", bool(not existing_keys)
    if max_documents and connection is not None:
        reserve = remaining_target_reserve(connection, config)
        # max_documents here is the remaining allowance for THIS run (caller-adjusted).
        if max_documents <= reserve:
            return False, "deeper_tier_budget_reserve", False

    zero_yield = not existing_keys
    if zero_yield and rescue_enabled:
        # Rescue: keep spending competitive (non-downstream) candidates up to hard.
        return True, None, True

    # CASE D / A: productive entity with remaining high-quality upstream candidates.
    if existing_keys and direction == UPSTREAM_EXPLICIT and (not hard or used < hard):
        if used >= max(target, min_docs) and zero_gain >= patience:
            return False, "diminishing_returns", False
        return True, None, False

    if direction != UPSTREAM_EXPLICIT:
        # CASE B: one supplier already found — allow bounded ambiguous continuation.
        if len(existing_keys) == 1 and direction == DIRECTION_AMBIGUOUS and (not hard or used < hard):
            return True, None, False
        return False, "target_met_no_strong_upstream_candidate", False

    if used >= max(target, min_docs) and zero_gain >= patience:
        return False, "diminishing_returns", False
    return True, None, False


_ESCALATION_LADDER = (
    ("primary", ["directional", "broad"]),
    ("rescue", ["directional", "broad", "component"]),
    ("llm_planned", ["directional", "broad", "component", "llm_planned"]),
    ("official", ["directional", "broad", "component", "llm_planned", "official_pdf"]),
)


def _escalation_stages_for(connection, entity_key: str) -> list[str]:
    row = connection.execute(
        "SELECT escalation_stage FROM frontier WHERE entity_key=?", (entity_key,),
    ).fetchone()
    label = (row["escalation_stage"] if row and "escalation_stage" in row.keys() else None) or "primary"
    for name, stages in _ESCALATION_LADDER:
        if name == label:
            return list(stages)
    return list(_ESCALATION_LADDER[0][1])


def _advance_escalation(connection, entity_key: str) -> Optional[str]:
    row = connection.execute(
        "SELECT escalation_stage FROM frontier WHERE entity_key=?", (entity_key,),
    ).fetchone()
    current = (row["escalation_stage"] if row and "escalation_stage" in row.keys() else None) or "primary"
    labels = [name for name, _ in _ESCALATION_LADDER]
    if current not in labels:
        current = "primary"
    idx = labels.index(current)
    if idx >= len(labels) - 1:
        return None
    nxt = labels[idx + 1]
    connection.execute(
        "UPDATE frontier SET escalation_stage=?, rescue_mode=1 WHERE entity_key=?",
        (nxt, entity_key),
    )
    connection.commit()
    return nxt


def _usage_metrics(connection, config=None) -> dict[str, Any]:
    rows = connection.execute("SELECT * FROM expansion_document_usage").fetchall()
    metrics = {"documents_selected": len({r["doc_id"] for r in rows}),
               "documents_completed": len({r["doc_id"] for r in rows if r["completed"]}),
               "document_attempts": sum(r["attempts"] for r in rows),
               "document_seconds": sum(r["elapsed_seconds"] for r in rows)}
    for key in ("extraction_chunks", "extraction_calls", "extraction_cache_hits",
                "extraction_retries", "candidate_facts", "accepted_facts", "rejected_facts"):
        metrics[key] = sum((json.loads(r["metrics_json"] or "{}") or {}).get(key, 0) or 0 for r in rows)
    attempts = metrics["document_attempts"]
    metrics["average_seconds_per_attempt"] = metrics["document_seconds"] / attempts if attempts else None
    if config is not None:
        # Per-document meters include failed calls and cache-hit resumes. Their
        # cumulative totals are more accurate than adding the latest return values.
        meter_dir = storage_path(config, "extractions") / "metrics"
        meters = []
        for doc_id in {row["doc_id"] for row in rows}:
            path = meter_dir / f"{doc_id}.json"
            if path.exists():
                try:
                    meters.append(json.loads(path.read_text(encoding="utf-8")))
                except (ValueError, OSError):
                    logger.warning("unreadable document metrics: %s", path)
        metrics["llm_calls"] = sum(int(m.get("total_llm_calls") or 0) for m in meters)
        metrics["documents_with_llm_metrics"] = len(meters)
        metrics["average_llm_calls_per_metered_document"] = metrics["llm_calls"] / len(meters) if meters else None
    return metrics


def run_expand(config: dict[str, Any], connection, llm, ontology: dict[str, Any], driver=None) -> dict[str, Any]:
    """Expand under persisted budgets; all accepted facts survive scheduling stops.

    Runtime is enforced at cancellable unit boundaries. An in-flight model call can
    finish after the deadline; observed timings prevent starting likely overruns.
    """
    ensure_run_snapshot(config)
    expansion, runtime = config.get("expansion", {}), config.get("runtime", {})
    max_tier = int(expansion.get("max_tier", 3))
    # Per-run document ceiling pauses the run; lifetime usage remains in SQLite.
    max_documents = documents_per_run_limit(expansion)
    max_seconds = float(runtime.get("max_minutes") or 0) * 60
    max_failures = int(runtime.get("max_consecutive_failures", DEFAULT_MAX_CONSECUTIVE_FAILURES))
    max_attempts = int(runtime.get("max_document_attempts", 2))
    retrieval_cfg = expansion.get("retrieval") or {}
    relation_policy = load_relation_evidence(
        ((config.get("validation") or {}).get("evidence") or {}).get("policy_file"))
    patience = int(retrieval_cfg.get("zero_gain_patience", 3))
    min_docs = int(retrieval_cfg.get("min_documents_before_diminishing_returns", 3))
    rescue_enabled = bool(retrieval_cfg.get("zero_yield_rescue", True))
    prompt1_drops = empty_prompt1_drop_counts()
    entity_stop_reasons: dict[str, int] = {}
    zero_yield_origins = 0
    graph_growth: list[dict[str, Any]] = []
    started = time.monotonic()
    elapsed_before = float(state.run_value(connection, "elapsed_seconds", 0))
    docs_at_run_start = state.document_usage_count(connection)
    run_new_claims_by_tier: dict[int, int] = {}
    run_id = state.run_value(connection, "run_id") or datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%f")
    state.set_run_value(connection, "run_id", run_id)
    manifest = mf.RetrievalManifest(REPO_ROOT / "logs" / f"retrieval_manifest_{run_id}.csv", run_id)
    requeued = state.requeue_stale_processing(connection)
    if requeued:
        logger.info("returned %d interrupted frontier entries to pending", requeued)
    seed_frontier(connection, config, ontology)
    # Legacy permanent beam-exclusion markers must not block resume.
    connection.execute(
        "UPDATE frontier SET stop_reason=NULL WHERE status=? AND stop_reason='frontier_beam_cap'",
        (state.PENDING,),
    )
    connection.commit()
    # A crash may occur after pipeline persistence and before usage bookkeeping.
    for doc in connection.execute("SELECT doc_id FROM documents WHERE status IN (?, ?)",
                                  (state.COMMITTED, state.RESOLVED if driver is None else state.COMMITTED)).fetchall():
        state.reconcile_completed_usage(connection, doc["doc_id"])
    # Existing legacy checkpoints may include terminal nodes as pending Companies.
    connection.execute("UPDATE frontier SET status=?, expandable=0, stop_reason='terminal_tier' WHERE tier>=?", (state.DONE, max_tier))
    connection.commit()
    # Repair the crash window between accepted fact persistence and frontier update,
    # even if the resumed run has no document/runtime budget left for more work.
    if storage_path(config, "extracted_facts").exists():
        refresh_edge_support(config)
        for known_tier in range(max_tier):
            parents = connection.execute("SELECT * FROM frontier WHERE tier=?", (known_tier,)).fetchall()
            for parent in parents:
                discover_upstream(connection, config, ontology, parent["entity_name"],
                    parent["entity_key"], known_tier, pd.DataFrame(columns=FACT_COLUMNS))
    summary = {"entities_expanded": 0, "documents_processed": 0, "documents_skipped": 0,
               "documents_failed": 0, "facts": 0, "nodes_merged": 0,
               "relationships_merged": 0, "discovered": {}, "documents_by_origin": {}, "stop_reason": None}
    consecutive_failures = 0

    def elapsed():
        return elapsed_before + time.monotonic() - started

    def checkpoint_time():
        state.set_run_value(connection, "elapsed_seconds", elapsed())

    def time_reason(*, estimate=False):
        remaining = max_seconds - elapsed()
        if not max_seconds:
            return None
        if remaining <= 0:
            return "runtime_budget"
        if estimate:
            observed = _usage_metrics(connection).get("average_seconds_per_attempt")
            if observed and observed * 1.15 > remaining:
                return "runtime_reserve"
        return None

    def finalize_entity(entity_name, entity_key, tier):
        # Required even after the final admitted document or a deadline: the graph
        # must retain every defensible supplier discovered by completed extraction.
        refresh_edge_support(config)
        facts = _facts_for_entity(config, connection, entity_key)
        found = discover_upstream(connection, config, ontology, entity_name, entity_key, tier, facts)
        if found:
            summary["discovered"].setdefault(entity_name, [])
            summary["discovered"][entity_name] = list(dict.fromkeys(summary["discovered"][entity_name] + found))
        checkpoint_time()
        manifest.flush()

    try:
        while True:
            stop = time_reason(estimate=True)
            if stop:
                summary["stop_reason"] = stop
                break
            allowed = None
            docs_this_run = state.document_usage_count(connection) - docs_at_run_start
            if max_documents and docs_this_run >= max_documents:
                allowed = state.incomplete_usage_entities(connection, max_attempts)
                if not allowed:
                    summary["stop_reason"] = "document_budget"
                    break
            row = state.next_frontier_entity(
                connection, max_tier,
                max_expandable_entities_by_tier=scheduling_batch_sizes(expansion),
                allowed_entity_keys=allowed,
                run_new_claims_by_tier=run_new_claims_by_tier,
            )
            if row is None:
                summary["stop_reason"] = "frontier_empty"
                break
            entity_name, entity_key, tier = row["entity_name"], row["entity_key"], int(row["tier"])
            if int(row["attempts"] or 0) == 1:
                run_new_claims_by_tier[tier] = int(run_new_claims_by_tier.get(tier, 0)) + 1
            target, hard = _entity_document_limits(expansion, tier)
            used = state.document_usage_count(connection, entity_key)
            zero_gain = int(row["zero_gain_streak"])
            existing_keys = _origin_edge_keys(config, entity_key)
            logger.info("entity=%s type=%s tier=%s used=%s target=%s max=%s global=%s/%s",
                        entity_name, row["entity_type"], tier, used, target, hard,
                        state.document_usage_count(connection), max_documents or "unlimited")
            if hard and used >= hard:
                retryable = entity_key in state.incomplete_usage_entities(connection, max_attempts)
                if not retryable:
                    finalize_entity(entity_name, entity_key, tier)
                    state.set_frontier_status(connection, entity_key, state.DONE, stop_reason="entity_document_cap")
                    continue
            try:
                stages = _escalation_stages_for(connection, entity_key)
                results = retrieve_for_entity(
                    config, entity_name, entity_key, tier, connection, manifest,
                    llm=llm, escalation_stages=stages,
                )
            except Exception as error:
                checkpoint_time()
                retriable = int(row["attempts"]) + 1 < int(runtime.get("max_retrieval_attempts", 2))
                state.set_frontier_status(connection, entity_key, state.PENDING if retriable else state.FAILED,
                                          str(error), stop_reason="retrieval_failed")
                continue
            checkpoint_time()
            reason, retry_pending = "candidate_pool_exhausted", False
            # Incomplete admitted documents must be retried before unseen candidates,
            # even when the global hard ceiling has already been reached.
            prepared = []
            entity_aliases = origin_name_variants(entity_name, (ontology.get("aliases") or {}).get("Company"))
            for result in results:
                if not discovery_candidate_is_meaningful(result, entity_name,
                        aliases=entity_aliases, policy=relation_policy):
                    prompt1_drops["candidate_unusable"] += 1
                    continue
                direction = (getattr(result, "discovery_direction", None)
                             or getattr(result, "_discovery_direction", None)
                             or classify_supply_direction(entity_name, result.document_text(),
                                                          aliases=entity_aliases, policy=relation_policy))
                if direction == DOWNSTREAM_EXPLICIT:
                    prompt1_drops["downstream_only"] += 1
                    continue
                try:
                    document = make_document(result.document_text(), **document_kwargs_from_result(result))
                except ValueError:
                    prompt1_drops["candidate_unusable"] += 1
                    continue
                known = state.find_document(connection, document["content_hash"], document.get("source_url"))
                doc_id = known["doc_id"] if known else document["doc_id"]
                usage = state.document_usage(connection, doc_id, entity_key)
                unfinished = usage is not None and not usage["completed"] and state.document_attempt_count(connection, doc_id) < max_attempts
                prepared.append((not unfinished, result, direction, document, known, usage, doc_id))
            prepared.sort(key=lambda item: item[0])
            rescue_active = bool(int(row["rescue_mode"] or 0)) if "rescue_mode" in row.keys() else False
            for _, result, direction, document, known, usage, doc_id in prepared:
                # Another result in this same batch may just have persisted identical
                # content. Refresh before reserving so duplicates never spend twice.
                known = state.find_document(connection, document["content_hash"], document.get("source_url"))
                doc_id = known["doc_id"] if known else document["doc_id"]
                usage = state.document_usage(connection, doc_id, entity_key)
                stop = time_reason(estimate=True)
                if stop:
                    summary["stop_reason"], reason = stop, stop
                    prompt1_drops["runtime_reserve" if stop == "runtime_reserve" else "other"] += 1
                    break
                if known and known["status"] in (state.COMMITTED, state.RESOLVED) and (driver is None or known["status"] == state.COMMITTED):
                    state.add_document_origin(connection, doc_id, entity_key, tier)
                    state.reconcile_completed_usage(connection, doc_id)
                    manifest.update_by_url(result.url, llm_status=mf.SKIPPED, dedup_status=mf.DUPLICATE_CONTENT)
                    summary["documents_skipped"] += 1
                    prompt1_drops["duplicate_content"] += 1
                    continue
                if state.document_attempt_count(connection, doc_id) >= max_attempts:
                    prompt1_drops["already_processed"] += 1
                    continue
                is_new_spend = usage is None
                if is_new_spend:
                    used = state.document_usage_count(connection, entity_key)
                    if hard and used >= hard:
                        reason = "entity_document_cap"
                        prompt1_drops["entity_max_reached"] += 1
                        break
                    globally_known = connection.execute(
                        "SELECT 1 FROM expansion_document_usage WHERE doc_id=? LIMIT 1", (doc_id,),
                    ).fetchone()
                    docs_this_run = state.document_usage_count(connection) - docs_at_run_start
                    run_remaining = (max_documents - docs_this_run) if max_documents else 0
                    if max_documents and not globally_known and docs_this_run >= max_documents:
                        summary["stop_reason"], reason = "document_budget", "document_budget"
                        prompt1_drops["global_cap"] += 1
                        break
                    admit, stop_reason, entered_rescue = _admit_above_target(
                        used=used, target=target, hard=hard, direction=direction,
                        existing_keys=existing_keys, zero_gain=zero_gain,
                        patience=patience, min_docs=min_docs,
                        rescue_enabled=rescue_enabled, connection=connection,
                        config=config, max_documents=run_remaining if max_documents else 0,
                    )
                    if entered_rescue and not rescue_active:
                        rescue_active = True
                        _set_rescue_mode(connection, entity_key, True)
                        logger.info(
                            "entity=%s zero-yield rescue activated used=%s target=%s max=%s",
                            entity_name, used, target, hard,
                        )
                    if not admit:
                        reason = stop_reason or "target_reached"
                        if stop_reason == "diminishing_returns":
                            prompt1_drops["diminishing_returns"] += 1
                        elif stop_reason == "entity_document_cap":
                            prompt1_drops["entity_max_reached"] += 1
                        else:
                            prompt1_drops["target_reached"] += 1
                        break
                # Reserve first; a crash/partial extraction cannot reset document spend.
                state.reserve_document(connection, doc_id, entity_key)
                if max_seconds:
                    config["_runtime_deadline_monotonic"] = time.monotonic() + max(0, max_seconds - elapsed())
                else:
                    config.pop("_runtime_deadline_monotonic", None)
                manifest.mark_selected(result)
                doc_started = time.monotonic()
                try:
                    document_summary = process_document(document, config, connection, llm, ontology, driver,
                        origin_entity=entity_key, tier=tier,
                        document_index=state.document_usage_count(connection), document_budget=max_documents or None)
                except Exception as error:
                    doc_elapsed = time.monotonic() - doc_started
                    state.finish_document_usage(connection, doc_id, entity_key, completed=False, elapsed_seconds=doc_elapsed)
                    checkpoint_time()
                    manifest.record_document(result, doc_id, mf.LLM_FAILED)
                    summary["documents_failed"] += 1
                    if time_reason() or getattr(error, "stop_reason", None) == "runtime_deadline":
                        summary["stop_reason"], reason = "runtime_budget", "runtime_budget"
                        break
                    consecutive_failures += 1
                    retry_pending = True
                    if consecutive_failures >= max_failures:
                        finalize_entity(entity_name, entity_key, tier)
                        state.set_frontier_status(connection, entity_key, state.PENDING, stop_reason="document_failures")
                        raise RuntimeError(f"{consecutive_failures} documents failed in a row; last error: {error}") from error
                    continue
                consecutive_failures = 0
                # Pipeline refreshed support; reused documents add only context, not LLM spend.
                now_keys = _origin_edge_keys(config, entity_key)
                gained_keys = now_keys - existing_keys
                gained = len(gained_keys)
                existing_keys |= now_keys
                if existing_keys and rescue_active:
                    rescue_active = False
                    _set_rescue_mode(connection, entity_key, False)
                zero_gain = 0 if gained else zero_gain + 1
                connection.execute("UPDATE frontier SET zero_gain_streak=? WHERE entity_key=?", (zero_gain, entity_key))
                connection.commit()
                state.finish_document_usage(connection, doc_id, entity_key, completed=True,
                    elapsed_seconds=time.monotonic() - doc_started, expansion_edges=gained, metrics=document_summary)
                checkpoint_time()
                if document_summary.get("skipped"):
                    summary["documents_skipped"] += 1
                    manifest.record_document(result, doc_id, mf.SKIPPED)
                    manifest.update_by_url(result.url, dedup_status=mf.DUPLICATE_CONTENT)
                    prompt1_drops["duplicate_content"] += 1
                    continue
                manifest.record_document(result, doc_id, mf.PROCESSED, accepted_expansion_edges=gained_keys)
                summary["documents_processed"] += 1
                summary["documents_by_origin"][entity_name] = summary["documents_by_origin"].get(entity_name, 0) + 1
                for key in ("facts", "nodes_merged", "relationships_merged"):
                    summary[key] += document_summary.get(key, 0)
                used = state.document_usage_count(connection, entity_key)
                # Fairness soft-yield removed as an unconditional TARGET stop.
                # Productive entities continue under _admit_above_target / diminishing returns.
                # Zero-yield entities continue via rescue/escalation up to hard max.
            if not existing_keys and reason == "candidate_pool_exhausted" and rescue_enabled:
                advanced = _advance_escalation(connection, entity_key)
                if advanced and (not hard or used < hard):
                    reason = f"escalation_{advanced}"
                    pending_escalation = True
                else:
                    reason = "zero_yield_rescue_exhausted"
                    pending_escalation = False
            else:
                pending_escalation = False
            if not existing_keys and used >= (target or 0):
                zero_yield_origins += 1
            finalize_entity(entity_name, entity_key, tier)
            unfinished = entity_key in state.incomplete_usage_entities(connection, max_attempts)
            pending = bool(
                summary["stop_reason"]
                or reason == "target_yield"
                or (retry_pending and unfinished)
                or pending_escalation
            )
            # Interrupt during zero-yield rescue: resume with remaining max capacity.
            if (
                summary["stop_reason"] in ("runtime_budget", "runtime_reserve", "max_documents", "document_budget")
                and rescue_enabled
                and not existing_keys
                and hard
                and used < hard
            ):
                pending = True
                _set_rescue_mode(connection, entity_key, True)
                reason = "zero_yield_rescue"
            elif not existing_keys and not pending:
                _set_rescue_mode(connection, entity_key, False)
            state.set_frontier_status(
                connection, entity_key,
                state.PENDING if pending else state.DONE,
                stop_reason=reason,
            )
            entity_stop_reasons[reason] = entity_stop_reasons.get(reason, 0) + 1
            if not pending:
                summary["entities_expanded"] += 1
            support = load_edge_support(config)
            counts = _persisted_graph_counts(config, connection)
            discovered_n = len(summary.get("discovered", {}).get(entity_name) or [])
            graph_growth.append(graph_growth_snapshot(
                total_nodes=counts["total_nodes"],
                tiered_nodes=counts["tiered_nodes"],
                context_nodes=counts["context_nodes"],
                expansion_edges=counts["expansion_relevant_edges"],
                new_supplier_entities=discovered_n,
                frontier_size=state.pending_frontier_count(connection, max_tier),
                documents_processed=summary["documents_processed"],
                accepted_facts=counts["accepted_facts"],
                new_suppliers_by_seed={entity_name: discovered_n} if discovered_n else {},
                total_edges=counts["total_edges"],
                supply_chain_core_edges=counts["supply_chain_core_edges"],
                nodes_merged_neo4j=summary.get("nodes_merged", 0),
            ))
            if summary["stop_reason"]:
                break
    finally:
        checkpoint_time()
        manifest.flush()
    summary["frontier"] = state.frontier_counts(connection)
    summary["frontier_by_tier"] = state.frontier_tier_counts(connection)
    summary["retrieval"] = manifest.summary()
    manifest.log_summary()
    summary["retrieval_manifest"] = str(manifest.path)
    summary["elapsed_seconds"] = round(time.monotonic() - started, 1)
    summary["elapsed_seconds_cumulative"] = round(elapsed(), 1)
    summary["usage"] = _usage_metrics(connection, config)
    # Clarify: extraction_retries counts only transient chunk LLM retries, not
    # validation/repair/disambiguation calls (those live in llm_calls meters).
    summary["usage"]["extraction_retries_note"] = (
        "Transient Prompt-1 chunk LLM retries only; not validation/repair/disambiguation."
    )
    summary["documents_selected_cumulative"] = summary["usage"]["documents_selected"]
    summary["documents_processed_cumulative"] = summary["usage"]["documents_completed"]
    summary["lifetime_documents_processed"] = summary["documents_processed_cumulative"]
    summary["documents_processed_this_run"] = summary["documents_processed"]
    summary["max_documents_per_run"] = max_documents
    effective_stop = summary.get("stop_reason") or (
        "frontier_empty" if not state.pending_frontier_count(connection, max_tier) else None
    )
    if not summary.get("stop_reason") and effective_stop:
        summary["stop_reason"] = effective_stop
    summary["completion"] = _frontier_completion(
        connection, max_tier, summary.get("stop_reason"),
        lifetime_documents=summary["lifetime_documents_processed"],
        documents_this_run=summary["documents_processed_this_run"],
    )
    summary["experiment_complete"] = bool(summary["completion"]["complete"])
    summary["quality"] = _quality_summary(config, summary)
    rejected_path = storage_path(config, "rejected_facts")
    rejected = pd.read_parquet(rejected_path) if rejected_path.exists() else pd.DataFrame()
    facts_path = storage_path(config, "extracted_facts")
    accepted = pd.read_parquet(facts_path) if facts_path.exists() else pd.DataFrame()
    # Clarify metric names: parquet lengths are unique persisted rows, not event streams.
    summary["accepted_unique_facts"] = len(accepted)
    summary["rejected_unique_facts"] = len(rejected)
    summary["rejected_facts"] = summary["rejected_unique_facts"]
    summary["rejection_categories"] = rejection_category_counts(rejected)
    summary["prompt1_drop_reasons"] = prompt1_drops
    summary["entity_stop_reasons"] = entity_stop_reasons
    summary["zero_yield_origins"] = zero_yield_origins
    summary["graph_growth"] = graph_growth
    support = load_edge_support(config)
    expansion_n = (
        int(support.expansion_relevant.fillna(False).sum())
        if not support.empty and "expansion_relevant" in support.columns else 0
    )
    tiered = sum(state.frontier_tier_counts(connection).values())
    summary["discovery_yield_warning"] = low_discovery_yield_warning(
        seed_count=len(seed_entities(config)),
        documents_processed=summary["documents_processed_cumulative"],
        tiered_nodes=tiered,
        expansion_edges=expansion_n,
        zero_yield_origins=zero_yield_origins,
    )
    # Premature-exhaustion diagnostic uses THIS RUN's spend vs per-run budget,
    # not lifetime docs against a per-run ceiling (which false-alarms on resume).
    summary["premature_discovery_warning"] = premature_discovery_exhaustion_warning(
        global_cap=max_documents,
        documents_processed=summary["documents_processed_this_run"],
        runtime_max_minutes=float(runtime.get("max_minutes") or 0),
        elapsed_seconds=float(summary.get("elapsed_seconds") or 0),
        stop_reason=summary.get("stop_reason") or (
            "frontier_empty" if not state.pending_frontier_count(connection, max_tier) else None
        ),
    )
    metrics_path = storage_path(config, "state_db").with_name("run_metrics.json")
    temporary = metrics_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    temporary.replace(metrics_path)
    print_quality_summary(summary["quality"])
    return summary
