import csv
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from schain import benchmark, benchmark_inputs, benchmark_selection, state
from schain.benchmark_report import export_comparison, relationships, session_status, tier_stats
from schain.sources.base import SearchResult
from .conftest import DOCUMENT_TEXT, OfflineTokenizer, StubLLM
from .test_snippet_batch import make_entry


class FakeModel(StubLLM):
    def _response(self, prompt):
        if "never combine evidence across document IDs" in prompt:
            self.prompts.append(prompt)
            ids = re.findall(r'"document_id":\s*"([^"]+)"', prompt)
            data = {"documents": [make_entry(
                doc_id, "CATL", "BMW", DOCUMENT_TEXT.splitlines()[0],
            ) for doc_id in dict.fromkeys(ids)]}
            content = json.dumps(data)
        else:
            content = self._answer(prompt)
        return SimpleNamespace(content=content, usage=SimpleNamespace(response_tokens=3))

    def invoke(self, input, **kwargs):
        return self._response(input)

    async def ainvoke(self, input, **kwargs):
        return self._response(input)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        "include: [config/selection_benchmark.yaml]\n"
        "seeds: {ev_manufacturers: [BMW]}\n"
        "expansion:\n"
        "  max_tier: 1\n"
        "  retrieval: {zero_yield_rescue: false}\n"
        "  selection:\n"
        "    target_tokens_per_candidate: 100000\n"
        "    initial_action_prior: {snippet: 1, skip: 0.000000000001, full_text: 0, both: 0}\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(config=config, workspace=tmp_path / "arms", dataset=tmp_path / "inputs.json",
                           mode="selection-only", policy="legacy", seed=42, budget=200000,
                           warmup=2, batch_size=2, resume=False, session_minutes=0)
    monkeypatch.setattr(benchmark, "git_provenance",
                        lambda: {"git_commit": "test-commit", "working_source_sha256": "test-source"})
    return args


def fake_contexts(candidates, *args, **kwargs):
    return np.tile(np.array([1., .5, .1, 1., .2, .5, .6, .6]), (len(candidates), 1)), np.ones((len(candidates), 384))


def freeze(args, monkeypatch, *, count=2, downloader=None, initial_facts=None):
    configs = benchmark.configs_for(args)
    native = configs["legacy"]
    ontology = benchmark.load_ontology(native)
    targets = benchmark.targets_for(native, ontology)

    def fake_retrieve(config, name, key, tier, connection, *, escalation_stages, record_snapshot):
        query = f'"{name}" suppliers'
        found = [SearchResult(
            "general_web", query, f"CATL supplies battery cells to {name}: report {index}",
            f"https://fixture.test/{key}/{index}", DOCUMENT_TEXT + f" Report {index}.",
            entity_key=key, tier=tier, search_id=f"{key}-{index}", provider_rank=index + 1,
        ) for index in range(count)]
        plan = [{"query": query, "max_results": count, "providers": ["general_web"]}]
        record_snapshot(plan, [{"query": query, "status": state.DONE}], [item.as_row() for item in found], found)
        return found

    def fake_download(result, _config):
        result.full_text_attempted = True
        result.text = result.snippet + " Archived full text."
        return result

    monkeypatch.setattr(benchmark_inputs, "retrieve_for_entity", fake_retrieve)
    monkeypatch.setattr(benchmark_inputs, "fetch_selected_full_text", downloader or fake_download)
    monkeypatch.setattr(benchmark_selection, "build_document_contexts", fake_contexts)
    from schain.expansion import loop
    monkeypatch.setattr(loop, "build_document_contexts", fake_contexts)
    benchmark_inputs.record_dataset(args.dataset, args.mode, configs, targets, targets, "test-v1",
                                    initial_facts=initial_facts)
    archive = benchmark_inputs.load_dataset(args.dataset)
    assert archive["preparation"]["llm_requests"] == 0
    if count:
        assert next(iter(archive["entries"].values()))["provider_results"][0]["result"]["provider_rank"] == 1

    def no_network(*args, **kwargs):
        pytest.fail("replay entered a live retrieval or download")

    for module in (loop, benchmark_inputs):
        for name in ("retrieve_for_entity", "fetch_selected_full_text"):
            monkeypatch.setattr(module, name, no_network)
    return archive


def run(args, model=None, tokenizer=OfflineTokenizer):
    return benchmark.run_arm(args, llm_factory=lambda _: model or FakeModel(),
                             tokenizer_factory=lambda _: tokenizer())


def change(args, **kwargs):
    return SimpleNamespace(**{**vars(args), **kwargs})


def test_controlled_parity_isolation_blinded_reports_and_no_network(experiment, monkeypatch):
    archive = freeze(experiment, monkeypatch)
    legacy, adaptive = FakeModel(), FakeModel()
    a = run(experiment, legacy)
    b = run(change(experiment, policy="adaptive_linucb"), adaptive)
    assert a["complete"] and b["complete"], (a["error"], b["error"])
    assert a["input_tokens"] == b["input_tokens"]
    assert a["requests"] == b["requests"] == len(legacy.prompts)
    assert legacy.prompts == adaptive.prompts
    assert a["output_tokens"] == b["output_tokens"] == 3 * a["requests"]
    assert a["new_supplier_relationships"] == b["new_supplier_relationships"] == 1
    assert a["dataset_sha256"] == b["dataset_sha256"] == archive["dataset_sha256"]
    assert a["comparison_sha256"] == b["comparison_sha256"]
    assert a["target_coverage"] == b["target_coverage"] == 1
    assert a["effective_config"]["llm"] == b["effective_config"]["llm"]
    assert a["effective_config"]["storage"] != b["effective_config"]["storage"]
    assert a["run_id"] != b["run_id"]
    assert b["warmup_complete"] and b["learned_performance_insufficient"]
    assert b["phases"]["warmup"]["input_tokens"] == b["input_tokens"]
    assert b["phases"]["warmup"]["new_supplier_relationships"] == 1
    assert b["human_reviewed_precision"] is None
    directory = export_comparison(experiment.workspace)
    review = json.loads((directory / "review.json").read_text(encoding="utf-8"))
    assert len(review) == 1 and "policy" not in review[0] and "run_id" not in review[0]
    labels = directory / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([["review_id", "label"], [review[0]["review_id"], "true"]])
    export_comparison(experiment.workspace, labels)
    comparison = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    assert [row["human_reviewed_precision"] for row in comparison["repetitions"]] == [1, 1]


@pytest.mark.parametrize("mode", benchmark_inputs.MODES)
@pytest.mark.parametrize("policy", benchmark_inputs.POLICIES)
def test_budget_rejection_inside_validation_and_resume(experiment, monkeypatch, mode, policy):
    args = change(experiment, mode=mode, policy=policy, warmup=0, budget=10000, batch_size=1)
    freeze(args, monkeypatch, count=1)

    class DenyValidation(OfflineTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            text = messages[-1]["content"].lower()
            if "for each candidate, decide" in text or "evidence entails" in text:
                return range(12000)
            return super().apply_chat_template(messages, **kwargs)

    model = FakeModel()
    report = run(args, model, DenyValidation)
    assert report["stop_reason"] == "token_budget", report["error"]
    assert report["input_tokens"] > 0
    assert report["first_rejected_request"] == {
        "request_input_tokens": 12000, "remaining_input_tokens": report["remaining_input_tokens"],
    }
    original_calls = len(model.prompts)
    assert run(change(args, resume=True), model, DenyValidation)["input_tokens"] == report["input_tokens"]
    assert len(model.prompts) == original_calls


def test_resume_retains_learner_spend_and_pending_batch(experiment, monkeypatch):
    args = change(experiment, policy="adaptive_linucb", warmup=1, batch_size=1)
    freeze(args, monkeypatch, count=3)
    real = benchmark_selection.process_document
    processed = 0

    def interrupt(*args, **kwargs):
        nonlocal processed
        real(*args, **kwargs)
        processed += 1
        if processed == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(benchmark_selection, "process_document", interrupt)
    first_model = FakeModel()
    first = run(args, first_model)
    assert first["stop_reason"] == "external_interruption" and not first["complete"]
    assert first["input_tokens"] > 0
    path = first["effective_config"]["storage"]["state_db"]
    with state.connect(Path(path)) as db:
        assert state.run_value(db, benchmark_selection.STATE_KEY)["model"]["counts"]["snippet"] == 1
    resumed_model = FakeModel()
    second = run(change(args, resume=True), resumed_model)
    assert second["complete"] and second["run_id"] == first["run_id"]
    assert second["requests"] == first["requests"] + len(resumed_model.prompts)
    assert second["candidate_decisions"] == 3 and second["warmup_complete"]
    assert second["phases"]["post_warmup"]["selected"] == 2
    path = second["effective_config"]["storage"]["state_db"]
    with state.connect(Path(path)) as db:
        counts = state.run_value(db, benchmark_selection.STATE_KEY)["model"]["counts"]
        assert counts == {"snippet": 3, "full_text": 0, "both": 0}
    third = run(change(args, seed=7))
    assert third["run_id"] != first["run_id"] and third["candidate_decisions"] == 3
    assert third["requests"] == second["requests"]


def test_full_replay_requires_frozen_retrieval_and_downloads(experiment, monkeypatch):
    args = change(experiment, mode="full-pipeline", warmup=0, policy="adaptive_linucb")
    archive = freeze(args, monkeypatch, count=1)
    replay = benchmark_inputs.Replay(archive, "adaptive_linucb")
    key = archive["targets"][0][1]
    candidate = replay.load(key, 0, stages=benchmark_inputs.PRIMARY)[0]
    replay.fetch(key, 0, benchmark_inputs.PRIMARY, candidate)
    assert candidate.text.endswith("Archived full text.")
    with pytest.raises(ValueError, match="missing frozen full-text"):
        replay.fetch(key, 0, benchmark_inputs.PRIMARY, replace(candidate, url="https://missing.test"))
    with pytest.raises(ValueError, match="missing frozen retrieval"):
        replay.load("company:missing", 1, stages=benchmark_inputs.PRIMARY)


def test_full_pipeline_adaptive_really_uses_frozen_body(experiment, monkeypatch):
    args = change(experiment, mode="full-pipeline", policy="adaptive_linucb")
    with args.config.open("a", encoding="utf-8") as handle:
        handle.write("    allowed_actions: [full_text]\n")
    cfg = args.config.read_text(encoding="utf-8")
    args.config.write_text(cfg.replace("snippet: 1", "snippet: 0").replace("full_text: 0", "full_text: 1"),
                           encoding="utf-8")
    freeze(args, monkeypatch, count=1)
    model = FakeModel()
    report = run(args, model)
    assert report["complete"] and report["new_supplier_relationships"] == 1, report["error"]
    assert report["pipeline_summary"]["adaptive_selection"]["model_action_counts"]["full_text"] == 1
    assert any("Archived full text." in prompt for prompt in model.prompts)


def test_existing_edges_are_excluded_even_when_extracted_again(experiment, monkeypatch, tmp_path):
    freeze(experiment, monkeypatch, count=1)
    first = run(experiment)
    assert first["new_supplier_relationships"] == 1
    initial = Path(first["effective_config"]["storage"]["extracted_facts"])
    args = change(experiment, dataset=tmp_path / "with-baseline.json", workspace=tmp_path / "second")
    archive = freeze(args, monkeypatch, count=1, initial_facts=initial)
    second = run(args)
    assert second["complete"] and second["candidates_successfully_processed"] == 1
    assert second["new_supplier_relationships"] == 0 and second["targets_with_supplier"] == 0
    assert second["initial_graph_sha256"] == archive["initial_facts"]["sha256"]


def test_full_pipeline_reports_success_and_time_limit(experiment, monkeypatch):
    args = change(experiment, mode="full-pipeline", warmup=0)
    freeze(args, monkeypatch, count=1)
    report = run(args)
    assert report["complete"] and report["new_supplier_relationships"] == 1, report["error"]
    assert report["company_counts_by_tier"]["1"] == 1
    timed = run(change(args, seed=43, session_minutes=1e-12))
    assert timed["stop_reason"] == "time_limit" and not timed["complete"] and not timed["requests"]


def test_full_pipeline_reports_missing_next_tier_as_incomplete(experiment, monkeypatch):
    args = change(experiment, mode="full-pipeline")
    config = args.config.read_text(encoding="utf-8").replace("max_tier: 1", "max_tier: 2")
    args.config.write_text(config, encoding="utf-8")
    freeze(args, monkeypatch, count=1)
    report = run(args)
    assert report["stop_reason"] == "error" and not report["complete"]
    assert "missing frozen retrieval" in report["error"] and "tier 1" in report["error"]
    assert report["new_supplier_relationships"] == 1


def test_unclosed_session_is_incomplete_after_lock_is_released(tmp_path):
    session = {"started_at": time.time(), "elapsed_seconds": 0.0}
    with benchmark.run_lock(tmp_path):
        assert session_status(tmp_path, session)[0] == "active"
    assert session_status(tmp_path, session)[0] == "external_interruption"


def test_relationship_dedup_with_aliases_and_seed_connected_tiers():
    def fact(supplier, buyer, **overrides):
        return {
            "subject_id": f"company:{supplier}", "subject_name": supplier, "subject_type": "Company",
            "object_id": f"company:{buyer}", "object_name": buyer, "object_type": "Company",
            "relation": "suppliesTo", "evidence": "Grounded evidence", "expansion_relevant": True,
            "ontology_valid": True, "direction_explicit": True, "evidence_validated": True,
            "evidence_quote_grounded": True, **overrides,
        }

    facts = pd.DataFrame([fact("alpha", "bmw"), fact("alias", "bmw"), fact("beta", "alpha"),
                          fact("gamma", "beta"), fact("delta", "gamma"), fact("epsilon", "delta"),
                          fact("stray", "orphan"), fact("mine", "bmw", subject_type="Mine"),
                          fact("aux", "bmw", expansion_relevant=False),
                          fact("wrong", "bmw", direction_explicit=False),
                          fact("ungrounded", "bmw", evidence_quote_grounded=False)])
    edges = relationships(facts, {"company:alias": "company:alpha"}, require_witness=True)
    assert len(edges["company:alpha", "company:bmw"]) == 2 and len(edges) == 6
    tiers = tier_stats(edges, {"company:bmw"})
    assert tiers["company_counts_by_tier"] == {str(index): 1 for index in range(6)}
    assert tiers["tiers_2_to_5_coverage"] == 1


def test_dry_run_never_builds_model_or_tokenizer(experiment, monkeypatch, capsys):
    freeze(experiment, monkeypatch, count=0)

    def unexpected(*args, **kwargs):
        pytest.fail("dry-run initialized a model or tokenizer")

    monkeypatch.setattr(benchmark, "build_llm", unexpected)
    monkeypatch.setattr(benchmark, "load_tokenizer", unexpected)
    assert benchmark.main(["dry-run", "--mode", experiment.mode, "--dataset", str(experiment.dataset),
                           "--config", str(experiment.config)]) == 0
    assert "200000 input tokens" in capsys.readouterr().out
    assert not experiment.workspace.exists()
