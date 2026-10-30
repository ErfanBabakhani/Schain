"""Offline regressions for evidence witnesses, contexts and persisted export semantics."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from schain import state
from schain.graph_export import _build_nodes_and_edges, build_graph_bundle
from schain.kg.facts import upstream_companies
from schain.kg.support import (
    aggregate_edge_support, contextualize_facts, frontier_supplier_records,
    refresh_edge_support,
)
from schain.ontology import load_ontology
from schain.run_config import ensure_run_snapshot


def fact(**changes):
    return {
        "fact_id": "f1", "doc_id": "d1", "chunk_index": 0,
        "subject_id": "company:catl", "subject_type": "Company",
        "subject_name": "CATL", "subject_mention": "CATL",
        "object_id": "company:bmw", "object_type": "Company",
        "object_name": "BMW", "object_mention": "BMW", "relation": "suppliesTo",
        "evidence": "CATL supplies BMW.", "ontology_valid": True,
        "evidence_validated": True, "direction_explicit": True,
        "evidence_quote_grounded": True, "expansion_relevant": True,
        "source_effective_score": 0, "content_source": "title_snippet",
        "origin_entity": "company:bmw", "provider": "google_news",
        **changes,
    }


def graph(config, rows, support=None, ontology=None):
    return _build_nodes_and_edges(
        pd.DataFrame(rows), {}, {}, ["BMW"], {}, ontology or load_ontology(config),
        edge_support=support, max_tier=5,
    )


def test_runtime_cannot_combine_gates_from_different_witnesses(ingest_config):
    rows = [fact(evidence_validated=False),
            fact(fact_id="f2", doc_id="d2", expansion_relevant=False)]
    support = aggregate_edge_support(pd.DataFrame(rows), config=ingest_config)
    assert not bool(support.iloc[0].frontier_eligible)
    assert support.iloc[0].frontier_reason["eligible_witness_fact_ids"] == []
    nodes, edges, *_ = graph(ingest_config, rows, support)
    assert edges[0]["quality"]["frontier_eligible"] is False
    assert next(n for n in nodes if n["id"] == "company:catl")["discovery_tier"] is None


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA, "False"])
def test_runtime_missing_validation_never_becomes_true(ingest_config, missing):
    support = aggregate_edge_support(pd.DataFrame([fact(ontology_valid=missing)]), config=ingest_config)
    assert not bool(support.iloc[0].frontier_eligible)


def test_explicitly_ungrounded_witness_cannot_advance(ingest_config):
    support = aggregate_edge_support(pd.DataFrame([fact(evidence_quote_grounded=False)]), config=ingest_config)
    assert not bool(support.iloc[0].frontier_eligible)
    assert support.iloc[0].frontier_rejection_reason == "evidence_quote_not_grounded"


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA])
def test_unchecked_legacy_grounding_stays_global_but_never_core(ingest_config, missing):
    rows = [fact(evidence_quote_grounded=missing)]
    support = aggregate_edge_support(pd.DataFrame(rows), config=ingest_config)
    assert not bool(support.iloc[0].frontier_eligible)
    assert support.iloc[0].frontier_rejection_reason == "evidence_quote_not_checked"
    nodes, edges, *_ = graph(ingest_config, rows, support)
    assert len(edges) == 1
    assert edges[0]["quality"]["evidence_quote_grounded"] is None
    assert edges[0]["is_supply_chain_edge"] is False
    assert next(n for n in nodes if n["id"] == "company:catl")["discovery_tier"] is None


def test_one_valid_source_survives_unrelated_weak_witness(ingest_config):
    rows = [fact(), fact(fact_id="f2", doc_id="d2", direction_explicit=False)]
    support = aggregate_edge_support(pd.DataFrame(rows), config=ingest_config)
    assert bool(support.iloc[0].frontier_eligible)
    assert support.iloc[0].frontier_reason["eligible_witness_doc_ids"] == ["d1"]
    assert support.iloc[0].max_source_score == 0


def test_typed_mine_record_preserves_canonical_id(ingest_config):
    support = aggregate_edge_support(pd.DataFrame([
        fact(subject_id="mine:greenbushes", subject_name="Greenbushes", subject_type="Mine")
    ]), config=ingest_config)
    records = frontier_supplier_records(support, {"company:bmw"}, ["suppliesTo"])
    assert records[0]["entity_id"] == "mine:greenbushes"
    assert records[0]["entity_type"] == "Mine"
    assert records[0]["priority_reason"]["explicit_valid_witness"]


def test_reputation_does_not_change_frontier_priority(ingest_config):
    priorities = []
    for score in (0, 50, 100):
        support = aggregate_edge_support(pd.DataFrame([fact(source_effective_score=score)]), config=ingest_config)
        priorities.append(frontier_supplier_records(support, {"company:bmw"}, ["suppliesTo"])[0]["priority"])
    assert len(set(priorities)) == 1


def test_fulltext_counts_unique_documents_not_quotes(ingest_config):
    rows = [fact(content_source="full_text"), fact(fact_id="f2", content_source="full_text")]
    support = aggregate_edge_support(pd.DataFrame(rows), config=ingest_config)
    assert support.iloc[0].document_count == support.iloc[0].full_text_support_count == 1
    _, edges, *_ = graph(ingest_config, rows, support)
    assert edges[0]["evidence_count"] == 2
    assert edges[0]["quality"]["full_text_support_count"] == 1


def test_fresh_valid_facts_override_stale_false_cache(ingest_config):
    support = aggregate_edge_support(pd.DataFrame([fact(direction_explicit=False)]), config=ingest_config)
    _, edges, *_ = graph(ingest_config, [fact()], support)
    assert edges[0]["quality"]["frontier_eligible"] is True
    assert edges[0]["quality"]["frontier_rejection_reason"] is None
    assert edges[0]["quality"]["frontier_reason"]["parent_tier_valid"] is True
    assert edges[0]["is_supply_chain_edge"] is True


def test_all_false_relevance_never_falls_back_to_all_facts():
    rows = pd.DataFrame([fact(expansion_relevant=False)])
    assert upstream_companies(rows, {"company:bmw"}, ["suppliesTo"]) == []


def test_origin_projection_requires_exact_object_and_keeps_provenance():
    rows = pd.DataFrame([
        fact(expansion_relevant=False, origin_entity="company:tesla"),
        fact(fact_id="f2", expansion_relevant=False, object_id="company:stellantis"),
    ])
    projected = contextualize_facts(rows, {"d1": {"company:bmw"}})
    assert projected.expansion_relevant.tolist() == [True, False]
    assert projected.iloc[0].origin_entity == "company:tesla"
    assert projected.iloc[0].expansion_origin_ids == ["company:bmw"]
    assert rows.expansion_relevant.tolist() == [False, False]


def test_shared_document_origin_reuse_exports_without_mutating_facts(ingest_config):
    config = deepcopy(ingest_config)
    config["seeds"] = {"seeds": ["BMW", "Tesla Inc."]}
    path = Path(config["storage"]["extracted_facts"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([fact(origin_entity="company:tesla", expansion_relevant=False)]).to_parquet(path, index=False)
    conn = state.connect(Path(config["storage"]["state_db"]))
    state.register_document(conn, {"doc_id": "d1", "content_hash": "h1"}, "company:tesla", 0)
    state.add_document_origin(conn, "d1", "company:bmw", 0)
    conn.close()
    support = refresh_edge_support(config)
    assert bool(support.iloc[0].frontier_eligible)
    bundle = build_graph_bundle(config)
    edge = bundle["edges"][0]
    assert edge["quality"]["frontier_eligible"] is True
    assert edge["evidence"][0]["origin_entity"] == "company:tesla"
    assert edge["evidence"][0]["expansion_origin_ids"] == ["company:bmw"]
    assert not bool(pd.read_parquet(path).iloc[0].expansion_relevant)


def test_valid_global_downstream_fact_stays_out_of_supply_core(ingest_config):
    _, edges, *_ = graph(ingest_config, [fact(expansion_relevant=False)])
    assert len(edges) == 1
    assert edges[0]["quality"]["frontier_eligible"] is False
    assert edges[0]["is_supply_chain_edge"] is False


def test_mine_is_terminal_unless_explicitly_expandable(ingest_config):
    rows = [fact(subject_id="mine:greenbushes", subject_type="Mine", subject_name="Greenbushes")]
    nodes, *_ = graph(ingest_config, rows)
    mine = next(n for n in nodes if n["type"] == "Mine")
    assert mine["discovery_tier"] == 1
    assert mine["is_expandable"] is False
    ontology = load_ontology(ingest_config)
    ontology["expandable_types"] = ["Company", "Mine"]
    nodes, *_ = graph(ingest_config, rows, ontology=ontology)
    assert next(n for n in nodes if n["type"] == "Mine")["is_expandable"] is True


def test_export_uses_expansion_snapshot_for_seed_nodes_and_depth(ingest_config):
    config = deepcopy(ingest_config)
    config["seeds"] = {"experiment": ["BMW", "Toyota"]}
    config["expansion"] = {"max_tier": 5, "max_documents": 150}
    ensure_run_snapshot(config)
    later = deepcopy(config)
    later["seeds"] = {"accidental": ["Tesla Inc."]}
    later["expansion"]["max_tier"] = 2
    bundle = build_graph_bundle(later)
    assert bundle["metadata"]["seeds"] == ["BMW", "Toyota"]
    assert bundle["metadata"]["max_tier"] == 5
    assert bundle["metadata"]["run_config_provenance"] == "expansion_snapshot"
    assert {n["name"] for n in bundle["nodes"] if n["is_seed"]} == {"BMW", "Toyota"}


def test_no_neo4j_resolved_documents_count_as_processed(ingest_config):
    conn = state.connect(Path(ingest_config["storage"]["state_db"]))
    for doc_id, status in (("d1", state.RESOLVED), ("d2", state.COMMITTED), ("d3", state.EXTRACTED)):
        state.register_document(conn, {"doc_id": doc_id, "content_hash": doc_id})
        state.set_document_status(conn, doc_id, status)
    conn.close()
    stats = build_graph_bundle(ingest_config)["statistics"]
    assert stats["documents_processed"] == 2
    assert stats["documents_committed_to_neo4j"] == 1


def test_zero_fact_selected_docs_and_publisher_are_counted(ingest_config):
    docs = [
        {"doc_id": "d1", "provider": "google_news", "content_source": "full_text", "publisher_name": "Publisher", "publisher_domain": "publisher.example"},
        {"doc_id": "empty", "provider": "general_web", "content_source": "title_snippet"},
    ]
    path = Path(ingest_config["storage"]["normalized_documents"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(docs).to_parquet(path, index=False)
    pd.DataFrame([fact(), fact(fact_id="f2")]).to_parquet(ingest_config["storage"]["extracted_facts"], index=False)
    bundle = build_graph_bundle(ingest_config)
    stats = bundle["statistics"]
    assert stats["selected_documents_by_provider"] == {"google_news": 1, "general_web": 1}
    assert stats["accepted_expansion_edges_by_provider"] == {"google_news": 1}
    assert bundle["documents"][0]["publisher_name"] == "Publisher"
    assert bundle["documents"][0]["publisher_domain"] == "publisher.example"


def test_legacy_verification_pseudofacts_never_export_as_evidence(ingest_config):
    path = Path(ingest_config["storage"]["extracted_facts"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([fact(), fact(fact_id="fverify", doc_id="verify:legacy")]).to_parquet(path, index=False)
    bundle = build_graph_bundle(ingest_config)
    assert bundle["statistics"]["facts"] == 1
    assert bundle["edges"][0]["support_count"] == 1


def test_export_exposes_partial_extraction_coverage(ingest_config):
    import json
    from schain.config import storage_dir

    docs = Path(ingest_config["storage"]["normalized_documents"])
    docs.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"doc_id": "d1", "provider": "general_web"}]).to_parquet(docs, index=False)
    pd.DataFrame([fact()]).to_parquet(ingest_config["storage"]["extracted_facts"], index=False)
    metric = storage_dir(ingest_config, "extractions") / "metrics" / "d1.json"
    metric.parent.mkdir(parents=True, exist_ok=True)
    metric.write_text(json.dumps({
        "last_attempt": {"extraction_chunks_available": 9, "extraction_chunks": 4,
                         "extraction_chunks_omitted": 5, "extraction_coverage": "partial"},
        "total_llm_calls": 7,
    }), encoding="utf-8")
    exported = build_graph_bundle(ingest_config)["documents"][0]["extraction_metrics"]
    assert exported["extraction_coverage"] == "partial"
    assert exported["extraction_chunks_omitted"] == 5
    assert exported["total_llm_calls"] == 7
    metrics = json.loads(metric.read_text(encoding="utf-8"))
    metrics["extraction_metrics"] = metrics["last_attempt"]
    metrics["last_attempt"] = {"extraction_chunks": 0}
    metric.write_text(json.dumps(metrics), encoding="utf-8")
    resumed = build_graph_bundle(ingest_config)["documents"][0]["extraction_metrics"]
    assert resumed["extraction_coverage"] == "partial"
    assert resumed["extraction_chunks"] == 4
