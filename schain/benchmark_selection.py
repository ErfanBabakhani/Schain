"""Run the fixed target experiment through SChain's existing document pipeline."""

import logging
import math
import json
from dataclasses import replace

import numpy as np
import pandas as pd
from neo4j_graphrag.components.types import Neo4jGraph

from . import state
from .benchmark_inputs import PRIMARY, digest, result_key
from .checkpoint import atomic_write_json
from .config import storage_dir, storage_path
from .documents import make_document
from .expansion.adaptive import LinUCB, build_document_contexts, choose_action, normalize_reward
from .expansion.frontier import seed_frontier
from .expansion.loop import _origin_edge_keys
from .kg.extract import build_snippet_batch_prompt, extract_snippet_batch
from .kg.relation_evidence import DOWNSTREAM_EXPLICIT, classify_supply_direction, load_relation_evidence
from .kg.support import refresh_edge_support
from .llm.provider import InputTokenBudgetExceeded
from .pipeline import process_document
from .sources.origin import origin_name_variants
from .sources.ranking import discovery_candidate_is_meaningful, sort_discovery_results
from .sources.retrieve import document_kwargs_from_result

logger = logging.getLogger(__name__)
STATE_KEY = "selection_benchmark"


def candidate_pool(config, ontology, replay, name, entity_id, tier):
    aliases = origin_name_variants(name, (ontology.get("aliases") or {}).get("Company"))
    evidence = (config.get("validation") or {}).get("evidence") or {}
    policy = load_relation_evidence(evidence.get("policy_file"))
    results = [replace(result, text=None, cached_full_text=None, fetched_text=None)
               for result in replay.load(entity_id, tier, stages=PRIMARY)]
    candidates = {}
    content, urls = set(), set()
    for result in sort_discovery_results(results, name, aliases=aliases, policy=policy):
        key = result_key(result)
        if (not discovery_candidate_is_meaningful(result, name, aliases=aliases, policy=policy)
                or classify_supply_direction(name, result.document_text(), aliases=aliases,
                                             policy=policy) == DOWNSTREAM_EXPLICIT):
            replay.event(entity_id, tier, result, "ineligible")
            continue
        document = make_document(result.document_text(), **document_kwargs_from_result(result))
        document.update(processing_action="snippet", origin_entity_name=name)
        url = document["source_url"]
        if document["content_hash"] in content or (url and url in urls):
            replay.event(entity_id, tier, result, "duplicate")
            continue
        content.add(document["content_hash"])
        if url:
            urls.add(url)
        candidates[key] = (result, document)
    return candidates, aliases


def known_companies(config, connection):
    names = {row[0] for row in connection.execute("SELECT entity_name FROM frontier")}
    path = storage_path(config, "extracted_facts")
    if path.exists():
        facts = pd.read_parquet(path)
        for prefix in ("subject", "object"):
            names.update(facts.loc[facts[f"{prefix}_type"].eq("Company"),
                                   f"{prefix}_name"].dropna().astype(str))
    path = storage_path(config, "entity_resolution")
    if path.exists():
        resolution = pd.read_parquet(path)
        if "aliases" in resolution:
            for aliases in resolution["aliases"].dropna():
                if hasattr(aliases, "__iter__") and not isinstance(aliases, str):
                    names.update(map(str, aliases))
    return names


def pick_batch(config, connection, llm, learner, journal, name, aliases, entity_id, eligible):
    selection = config["expansion"]["selection"]
    limit = selection["snippet_batch_size"]
    if learner is None:
        return [{"key": key, "action": "snippet", "phase": "overall"} for key in list(eligible)[:limit]]
    count = journal["decisions"]
    warmup = count < selection["warmup_candidates"]
    keys = list(eligible)
    if warmup:
        keys = keys[:min(limit, selection["warmup_candidates"] - count)]
    contexts, embeddings = build_document_contexts(
        [eligible[key][0] for key in keys], name, aliases=aliases,
        known_companies=known_companies(config, connection),
        history=journal["history"].get(entity_id, []),
        official_domains=list(selection.get("official_domains") or []), config=config,
    )
    expected = limit
    if warmup:
        prior = float((selection.get("initial_action_prior") or {}).get("snippet", .60))
        expected = min(limit, len(keys), 1 / max(1 - prior, 1e-9))
    overhead = int(selection.get("prompt_overhead_tokens_estimate", 2000))
    decisions = []
    for index, key in enumerate(keys):
        candidate = eligible[key][0]
        snippet = "\n".join(part for part in (candidate.title, candidate.snippet) if part)
        cost = llm.count_text_tokens(snippet) + math.ceil(overhead / expected)
        action, details = choose_action(
            learner, contexts[index], estimated_costs={"snippet": cost},
            virtual_price=journal["price"],
            cost_reference=float(selection.get("cost_reference_tokens", 1000)),
            priors=selection.get("initial_action_prior"), warmup=warmup,
            temperature=float(selection.get("warmup_temperature", 1.0)),
            reward_weight=float(selection.get("primal_reward_weight_V", 1.0)),
            rng=np.random.default_rng(selection["random_seed"] + count + index),
        )
        decisions.append({"key": key, "action": action, "phase": "warmup" if warmup else "greedy",
                          "context": contexts[index].tolist(), "embedding": embeddings[index].tolist(),
                          **details})
    if warmup:
        return decisions
    positive = [item for item in decisions if item["action"] == "snippet"]
    positive.sort(key=lambda item: -item["scores"]["snippet"])
    return positive[:limit] if positive else decisions


def cache_batch(config, llm, ontology, connection, budget_key, entity_id, plan, candidates):
    document_ids = [candidates[item["key"]][1]["doc_id"] for item in plan if item["action"] == "snippet"]
    if len(document_ids) < 2:
        return {}, 0
    batch_id = digest([entity_id, plan])
    path = storage_dir(config, "extractions") / "benchmark_batches" / f"{batch_id}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        for data in cached["extractions"].values():
            data["graph"] = Neo4jGraph.model_validate(data["graph"])
        return cached["extractions"], cached["input_tokens"]
    docs, before = [], state.run_value(connection, budget_key)["spent_estimate"]
    limit = state.run_value(connection, budget_key)["limit"] - before
    context = llm.input_context_limit_tokens()
    for item in plan:
        if item["action"] != "snippet":
            continue
        result, document = candidates[item["key"]]
        candidate = {**document, "snippet": result.snippet}
        proposed = [*docs, candidate]
        tokens = llm.count_request_tokens((build_snippet_batch_prompt(proposed, ontology),), {})
        if tokens > limit or (context is not None and tokens > context):
            break
        docs = proposed
    extracted = {}
    if len(docs) >= 2:
        try:
            extracted = extract_snippet_batch(llm, docs, ontology)
        except InputTokenBudgetExceeded:
            raise
        except Exception as error:
            logger.warning("snippet batch failed, retrying through document extraction: %s", error)
    spent = state.run_value(connection, budget_key)["spent_estimate"] - before
    atomic_write_json(path, {"input_tokens": spent, "extractions": {
        doc_id: {**data, "graph": data["graph"].model_dump(mode="json")}
        for doc_id, data in extracted.items()
    }})
    return extracted, spent


def run_selection(config, connection, llm, ontology, replay, targets, run_id, *, check_time):
    """Decisions and their rewards share a checkpoint. The token ledger charges before I/O."""
    selection = config["expansion"]["selection"]
    adaptive = selection["policy"] == "adaptive_linucb"
    budget_key = f"input_budget:{run_id}"
    seed_frontier(connection, config, ontology)
    if storage_path(config, "extracted_facts").exists():
        refresh_edge_support(config)
    journal = state.run_value(connection, STATE_KEY) or {
        "target": 0, "done": [], "pending": [], "decisions": 0, "history": {},
        "price": float(selection.get("initial_virtual_queue", 0.0)), "model": None,
    }
    learner = None
    if adaptive:
        learner = (LinUCB.from_dict(journal["model"]) if journal["model"] else
                   LinUCB(float(selection.get("ridge_lambda", 1.0)), float(selection.get("linucb_alpha", .5))))

    def checkpoint():
        if learner is not None:
            journal["model"] = learner.to_dict()
        state.set_run_value(connection, STATE_KEY, journal)

    def spent():
        return state.run_value(connection, budget_key)["spent_estimate"]

    def settle(cost):
        journal["price"] = max(0.0, journal["price"] + (
            cost - float(selection["target_tokens_per_candidate"])
        ) / float(selection.get("cost_reference_tokens", 1000)))

    checkpoint()
    while journal["target"] < len(targets):
        check_time()
        name, entity_id, tier = targets[journal["target"]]
        candidates, aliases = candidate_pool(config, ontology, replay, name, entity_id, tier)

        def record_duplicate(known, result, phase=None):
            if known["status"] in (state.RESOLVED, state.COMMITTED):
                state.add_document_origin(connection, known["doc_id"], entity_id, tier)
            replay.event(entity_id, tier, result, "duplicate", phase=phase)

        while True:
            check_time()
            if not journal["pending"]:
                done = set(journal["done"])
                eligible = {}
                for key, (result, doc) in candidates.items():
                    if key in done:
                        continue
                    known = state.find_document(connection, doc["content_hash"], doc.get("source_url"))
                    if known:
                        record_duplicate(known, result)
                    else:
                        eligible[key] = (result, doc)
                if not eligible:
                    break
                journal["pending"] = pick_batch(
                    config, connection, llm, learner, journal, name, aliases, entity_id, eligible,
                )
                journal["batch_plan"] = [dict(item) for item in journal["pending"]]
                checkpoint()
            plan = journal["batch_plan"]
            llm.benchmark_phase = plan[0]["phase"]
            extracted, batch_tokens = cache_batch(config, llm, ontology, connection, budget_key,
                                                  entity_id, plan, candidates)
            selected = [item["key"] for item in plan if item["action"] == "snippet"]
            recipients = [key for key in selected if candidates[key][1]["doc_id"] in extracted] or selected
            shares = {key: batch_tokens // len(recipients) + int(index < batch_tokens % len(recipients))
                      for index, key in enumerate(recipients)}
            while journal["pending"]:
                check_time()
                item = journal["pending"][0]
                key, phase = item["key"], item["phase"]
                result, doc = candidates[key]
                if item["action"] == "skip":
                    replay.event(entity_id, tier, result, "skipped", phase=phase)
                    settle(0)
                else:
                    doc_id = doc["doc_id"]
                    if "spent_before" not in item:
                        duplicate = state.find_document(connection, doc["content_hash"], doc.get("source_url"))
                        if duplicate:
                            record_duplicate(duplicate, result, phase)
                            if adaptive:
                                settle(shares.get(key, 0))
                            journal["decisions"] += 1
                            journal["done"].append(key)
                            journal["pending"].pop(0)
                            checkpoint()
                            continue
                        item.update(spent_before=spent(), edges_before=sorted(_origin_edge_keys(config, entity_id)))
                        checkpoint()
                    replay.event(entity_id, tier, result, "selected", doc_id=doc_id, phase=phase)
                    try:
                        process_document(
                            doc, config, connection, llm, ontology, origin_entity=entity_id, tier=tier,
                            precomputed_extraction=extracted.get(doc_id),
                        )
                        if state.run_value(connection, budget_key).get("first_rejection"):
                            raise InputTokenBudgetExceeded("a request reached the input-token limit")
                    except InputTokenBudgetExceeded:
                        raise
                    except Exception as error:
                        if state.run_value(connection, budget_key).get("first_rejection"):
                            raise InputTokenBudgetExceeded("a request reached the input-token limit") from error
                        replay.event(entity_id, tier, result, "failed", doc_id=doc_id,
                                     phase=phase, error=str(error))
                    else:
                        gained = len(_origin_edge_keys(config, entity_id) - set(item["edges_before"]))
                        replay.event(entity_id, tier, result, "processed", doc_id=doc_id,
                                     phase=phase, gained=gained)
                        if adaptive:
                            weights = selection.get("reward_weights") or {}
                            reward = normalize_reward(gained * float(weights.get("new_supply_edge", 1)),
                                                      float(weights.get("clip_max", 5)))
                            learner.update("snippet", np.asarray(item["context"]), reward)
                            journal["history"].setdefault(entity_id, []).append(item["embedding"])
                    if adaptive:
                        settle(spent() - item["spent_before"] + shares.get(key, 0))
                journal["decisions"] += 1
                journal["done"].append(key)
                journal["pending"].pop(0)
                checkpoint()
        journal.update(target=journal["target"] + 1, done=[])
        checkpoint()
    if storage_path(config, "extracted_facts").exists():
        refresh_edge_support(config)
    return "candidate_exhaustion"
