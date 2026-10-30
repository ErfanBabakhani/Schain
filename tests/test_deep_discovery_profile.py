from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from schain import state
from schain.config import load_config
from schain.distributed.benchmark import compare_snapshots
from schain.distributed.expand import DistributedExpandProcessor, RetrievalCheckpoint
from schain.distributed.maintenance import build_parser
from schain.distributed.models import TaskLease
from schain.distributed.profile import (
    DEEP_MAXIMA, DEEP_TARGETS, validate_deep_discovery,
)
from schain.expansion.budget import entity_document_limits
from schain.expansion.queries import generate_queries
from schain.expansion.escalation import build_escalation_queries
from schain.sources.base import SearchResult
from schain.sources.ranking import discovery_candidate_is_meaningful


ROOT = Path(__file__).resolve().parents[1]


def test_deep_profile_effective_limits_are_not_silently_capped():
    config = load_config(ROOT / "config" / "distributed_neon_deep_discovery.yaml")
    snapshot = validate_deep_discovery(config)
    assert snapshot["target_documents_by_tier"] == DEEP_TARGETS
    assert snapshot["max_documents_by_tier"] == DEEP_MAXIMA
    assert snapshot["document_limit"] == 20000
    assert snapshot["max_full_text_fetches_per_entity"] == 250
    assert snapshot["max_title_lookups_per_entity"] == 100
    assert [entity_document_limits(config["expansion"], tier) for tier in range(6)] == [
        (80, 200), (80, 160), (60, 120), (50, 100), (40, 80), (0, 0),
    ]
    assert 15 <= len(generate_queries(config, "Toyota", 0)) <= 30
    assert config["distributed"]["resume_existing_only"] is True
    assert config["save"]["enabled"] is False
    planned = build_escalation_queries(
        config, "BMW", 0, stages=["directional", "broad"],
        aliases=["BMW", "BMW AG", "Bayerische Motoren Werke AG"],
    )
    assert len(planned) == 30
    assert any("BMW AG" in item["query"] for item in planned)


def test_maintenance_cli_contract_parses_required_commands():
    parser = build_parser()
    status = parser.parse_args(["status", "--experiment-id", "bmw-supply-chain-v2", "--json"])
    assert status.command == "status" and status.json
    plan = parser.parse_args([
        "extend", "--experiment-id", "bmw-supply-chain-v2", "--new-max-tier", "5",
        "--topup-tier", "1", "--plan",
    ])
    assert plan.plan and not plan.apply
    apply = parser.parse_args([
        "extend", "--experiment-id", "bmw-supply-chain-v2", "--new-max-tier", "5",
        "--topup-tier", "1", "--apply", "--confirm-id", "bmw-supply-chain-v2",
    ])
    assert apply.apply and apply.confirm_id == apply.experiment_id


def test_retrieval_checkpoint_isolated_by_topup_generation_and_recovery_stage():
    class Repository:
        def __init__(self):
            self.rows = {}
        def save_retrieval_checkpoint(self, lease, rows, *, checkpoint_entity_key=None):
            self.rows[(checkpoint_entity_key, lease.tier)] = rows
        def load_retrieval_checkpoint(self, experiment, key, tier):
            return self.rows.get((key, tier))

    lease = TaskLease(
        task_id="topup", experiment_id="same", task_key="topup:v1:toyota",
        worker_id="w", lease_token=1, tier=0, entity_key="company:toyota",
        entity_name="Toyota", payload={"generation": "deep_discovery_v1"},
    )
    checkpoint = RetrievalCheckpoint(Repository(), lease)
    result = SearchResult(provider="web", query="q", title="A supplies Toyota", url="https://a.test")
    checkpoint.save(lease.entity_key, 0, [result], stages=["directional", "broad"])
    assert checkpoint.load(lease.entity_key, 0, stages=["directional", "broad"])[0].url == result.url
    assert checkpoint.load(lease.entity_key, 0, stages=["directional", "broad", "component"]) is None


def test_noisy_store_and_faq_results_do_not_spend_budget_but_upstream_evidence_survives():
    noisy = SearchResult(
        provider="web", query='"ROHM" suppliers', title="ROHM products FAQ and shopping directory",
        snippet="Browse ROHM products and frequently asked questions.", url="https://store.test/faq",
    )
    assert not discovery_candidate_is_meaningful(noisy, "ROHM")
    valid = SearchResult(
        provider="web", query='"Samsung Electronics" component supplier',
        title="Meridian supplies Samsung Electronics",
        snippet="Meridian supplies components to Samsung Electronics under a procurement contract.",
        url="https://meridian.test/release",
    )
    assert discovery_candidate_is_meaningful(valid, "Samsung Electronics")


def test_benchmark_reports_observed_metrics_and_leaves_unlabelled_precision_unknown():
    before = {"documents": [{"content_source": "title_snippet"}], "facts": []}
    after = {
        "documents": [{"content_source": "full_text", "origin_relevant": True}],
        "facts": [{"relation": "suppliesTo", "subject_id": "company:a", "object_id": "company:b"}],
        "runtime_seconds": 12.5, "cost": 0.2,
    }
    report = compare_snapshots(before, after)
    assert report["after"]["unique_verified_suppliers"] == 1
    assert report["after"]["true_upstream_edges"] == 1
    assert report["after"]["full_text_ratio"] == 1.0
    assert report["after"]["precision"] is None
    labelled = compare_snapshots(before, after, labelled_edges={("company:a", "company:b")})
    assert labelled["after"]["precision"] == 1.0
    assert labelled["after"]["false_negatives"] == 0


def test_safe_resume_notebook_uses_profile_and_streaming_auto_continue():
    notebook = json.loads((
        ROOT / "notebooks" / "SChain_Expand.ipynb"
    ).read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source") or []) for cell in notebook["cells"])
    for required in (
        "bmw-supply-chain-v2", "distributed_neon_deep_discovery.yaml",
        "resume_existing_only", "TASKS_THIS_SESSION = 40", "AUTO_CONTINUE = True",
        "APPLY_ONE_TIME_MIGRATION = False", "SAVE_TO_NEO4J = False",
    ):
        assert required in source


def test_topup_result_emits_only_new_generation_artifacts(tmp_path):
    storage = {
        "state_db": str(tmp_path / "state.db"),
        "extracted_facts": str(tmp_path / "facts.parquet"),
        "rejected_facts": str(tmp_path / "rejected.parquet"),
        "normalized_documents": str(tmp_path / "documents.parquet"),
        "extractions": str(tmp_path / "extractions"),
    }
    Path(storage["extractions"]).mkdir()
    pd.DataFrame([
        {"doc_id": "old", "content_hash": "old-hash", "text": "historical"},
        {"doc_id": "new", "content_hash": "new-hash", "text": "new evidence"},
    ]).to_parquet(storage["normalized_documents"], index=False)
    pd.DataFrame([
        {"fact_id": "old-fact", "doc_id": "old", "relation": "suppliesTo",
         "subject_id": "company:old", "object_id": "company:toyota"},
        {"fact_id": "new-fact", "doc_id": "new", "relation": "suppliesTo",
         "subject_id": "company:new", "object_id": "company:toyota"},
    ]).to_parquet(storage["extracted_facts"], index=False)
    connection = state.connect(Path(storage["state_db"]))
    try:
        state.enqueue_entity(connection, "Toyota", 0, entity_id="company:toyota")
        state.enqueue_entity(
            connection, "New", 1, entity_id="company:new", discovered_from="company:toyota",
        )
        lease = TaskLease(
            task_id="topup", experiment_id="same", task_key="topup:v1:toyota",
            worker_id="w", lease_token=1, tier=0, entity_key="company:toyota",
            entity_name="Toyota", payload={"generation": "deep_discovery_v1"},
        )
        result = DistributedExpandProcessor(
            repository=None, config={}, llm=None, ontology={}, worker_id="w",
        )._collect_result(
            {"storage": storage}, connection, lease, {}, completed_before={"old"},
            existing_aliases=set(),
        )
    finally:
        connection.close()
    assert [item["document"]["doc_id"] for item in result.documents] == ["new"]
    assert [item["fact_id"] for item in result.accepted_facts] == ["new-fact"]
    assert [item["entity_key"] for item in result.discoveries] == ["company:new"]
