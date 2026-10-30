"""Regression cases for grounded graph metadata, seed aliases and supply paths."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from neo4j_graphrag.components.types import Neo4jGraph, Neo4jRelationship

from schain import state
from schain.graph_export import (
    _apply_tier_semantics,
    _build_nodes_and_edges,
    _recompute_shortest_tiers,
    _statistics,
    build_graph_bundle,
)
from schain.kg.writer import facts_to_graph_rows, merge_nodes, merge_relationships
from schain.kg.facts import fact_id, graph_to_facts, upsert_facts_parquet
from schain.names import canonical_id
from schain.ontology import load_ontology


def fact(**changes):
    row = {
        "fact_id": "f1", "doc_id": "d1", "chunk_index": 0,
        "subject_id": "company:catl", "subject_type": "Company",
        "subject_name": "CATL", "subject_mention": "CATL",
        "object_id": "company:bmw", "object_type": "Company",
        "object_name": "BMW", "object_mention": "BMW",
        "relation": "suppliesTo", "evidence": "CATL supplies BMW.",
        "ontology_valid": True, "evidence_validated": True,
        "evidence_quote_grounded": True,
        "direction_explicit": True, "expansion_relevant": True,
    }
    return {**row, **changes}


def make_graph(config, rows, *, seeds=None, support=None, aliases=None, max_tier=None):
    return _build_nodes_and_edges(
        pd.DataFrame(rows), aliases or {}, {}, ["BMW"] if seeds is None else seeds,
        {}, load_ontology(config), edge_support=support, max_tier=max_tier,
    )


def test_facts_determine_support_count_and_missing_cache_flags(ingest_config):
    rows = [fact(), fact(fact_id="f2", doc_id="d2")]
    stale_support = pd.DataFrame([{
        "subject_id": "company:catl", "object_id": "company:bmw",
        "relation": "suppliesTo", "document_count": 999,
        "has_supporting_conflict": float("nan"),
    }])
    nodes, edges, _, _ = make_graph(ingest_config, rows, support=stale_support)
    assert edges[0]["support_count"] == 2
    assert edges[0]["quality"]["contested"] is False
    assert edges[0]["quality"]["frontier_eligible"] is True
    assert edges[0]["expansion_relevant"] is True
    assert next(n for n in nodes if n["id"] == "company:catl")["discovery_tier"] == 1


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA])
def test_legacy_validation_is_unknown_and_false_stays_false(ingest_config, missing):
    row = fact(ontology_valid=missing, evidence_validated=missing,
               evidence_quote_grounded=missing, direction_explicit=missing,
               expansion_relevant=missing)
    _, edges, _, _ = make_graph(ingest_config, [row])
    quality = edges[0]["quality"]
    for key in ("ontology_valid", "evidence_validated", "direction_explicit",
                "evidence_quote_grounded", "frontier_eligible", "expansion_relevant"):
        assert quality[key] is None
    _, edges, _, _ = make_graph(ingest_config, [fact(evidence_validated=False)])
    assert edges[0]["quality"]["evidence_validated"] is False
    assert edges[0]["quality"]["frontier_eligible"] is False


def test_frontier_gates_must_hold_on_the_same_witness(ingest_config):
    rows = [fact(evidence_validated=False),
            fact(fact_id="f2", doc_id="d2", expansion_relevant=False)]
    nodes, edges, _, _ = make_graph(ingest_config, rows)
    assert edges[0]["quality"]["evidence_validated"] is True
    assert edges[0]["expansion_relevant"] is True
    assert edges[0]["quality"]["frontier_eligible"] is False
    assert next(n for n in nodes if n["id"] == "company:catl")["discovery_tier"] is None


def test_quote_grounding_is_exported_per_witness_and_aggregated(ingest_config):
    _, edges, _, _ = make_graph(ingest_config, [fact(evidence_quote_grounded=False)])
    edge = edges[0]
    assert edge["evidence"][0]["evidence_quote_grounded"] is False
    assert edge["quality"]["evidence_quote_grounded"] is False
    assert edge["quality"]["frontier_eligible"] is False


def test_seed_alias_does_not_create_a_duplicate_isolated_seed(ingest_config):
    resolved_id = canonical_id("Company", "Contemporary Amperex Technology")
    rows = [fact(subject_id=resolved_id,
                 subject_name="Contemporary Amperex Technology",
                 subject_mention="Contemporary Amperex Technology")]
    nodes, _, _, _ = make_graph(
        ingest_config, rows, seeds=["CATL"], aliases={resolved_id: {"CATL"}},
    )
    assert "company:catl" not in {n["id"] for n in nodes}
    assert [n["id"] for n in nodes if n["is_seed"]] == [resolved_id]
    assert next(n for n in nodes if n["id"] == resolved_id)["discovery_tier"] == 0


def test_ambiguous_seed_alias_does_not_silently_merge_entities(ingest_config):
    rows = [fact(subject_id="company:one", subject_mention="One"),
            fact(fact_id="f2", subject_id="company:two", subject_mention="Two")]
    nodes, _, _, _ = make_graph(
        ingest_config, rows, seeds=["Ambiguous"],
        aliases={"company:one": {"Ambiguous"}, "company:two": {"Ambiguous"}},
    )
    assert {n["id"] for n in nodes if n["is_seed"]} == {"company:ambiguous"}


def test_tiers_discard_frontier_paths_beyond_boundary(ingest_config):
    nodes = [
        {"id": "company:bmw", "type": "Company", "discovery_tier": 0, "is_seed": True},
        {"id": "company:catl", "type": "Company", "discovery_tier": 1,
         "tier_parent_ids": ["company:bmw"], "is_seed": False},
        {"id": "company:raw", "type": "Company", "discovery_tier": 2,
         "tier_parent_ids": ["company:catl"], "is_seed": False},
    ]
    edges = [
        {"source": "company:catl", "target": "company:bmw", "relation": "suppliesTo"},
        {"source": "company:raw", "target": "company:catl", "relation": "suppliesTo"},
    ]
    _recompute_shortest_tiers(nodes, edges, {"company:bmw"}, max_tier=1)
    _apply_tier_semantics(nodes, edges, max_tier=1)
    assert nodes[1]["discovery_tier"] == 1
    assert nodes[1]["is_expandable"] is False
    assert nodes[2]["discovery_tier"] is None
    assert nodes[2]["is_expandable"] is False


def test_invariant_check_rejects_phantom_parent_pointers_in_any_order():
    nodes = [
        {"id": "company:child", "type": "Company", "discovery_tier": 2,
         "tier_parent_ids": ["company:parent"]},
        {"id": "company:parent", "type": "Company", "discovery_tier": 1,
         "tier_parent_ids": ["company:bmw"]},
        {"id": "company:bmw", "type": "Company", "discovery_tier": 0, "is_seed": True},
    ]
    edges = [{"source": "company:child", "target": "company:parent", "relation": "suppliesTo"}]
    stats = _apply_tier_semantics(nodes, edges)
    assert nodes[0]["discovery_tier"] is None
    assert nodes[1]["discovery_tier"] is None
    assert stats["tier_invariant_violations"] == 2


def test_tier_statistics_cover_configured_depth(ingest_config):
    nodes, edges, _, tier_stats = make_graph(ingest_config, [fact()])
    stats = _statistics(nodes, edges, [], 1,
                        config={"expansion": {"max_tier": 6}}, tier_stats=tier_stats)
    assert stats["tier_6_nodes"] == 0


class RecordingDriver:
    def __init__(self):
        self.calls = []

    def execute_query(self, query, **params):
        self.calls.append((query, params))


def test_writer_retains_quote_document_mapping_and_stable_records():
    frame = pd.DataFrame([fact(), fact(fact_id="f2", doc_id="d2", evidence="BMW buys CATL cells.")])
    _, relationships = facts_to_graph_rows(frame)
    driver = RecordingDriver()
    merge_relationships(driver, "neo4j", relationships, ["suppliesTo"])
    query, params = driver.calls[0]
    records = [json.loads(row["evidence_record"]) for row in params["rows"]]
    assert {(r["doc_id"], r["evidence"]) for r in records} == {
        ("d1", "CATL supplies BMW."), ("d2", "BMW buys CATL cells."),
    }
    assert records[0]["fact_id"] == "f1"
    assert records[0]["evidence_quote_grounded"] is True
    assert "r.evidence_records" in query and "WHERE v <> row.evidence_record" in query
    assert "r.evidence = coalesce(r.evidence, row.evidence)" in query
    merge_relationships(driver, "neo4j", relationships, ["suppliesTo"])
    assert driver.calls[0] == driver.calls[1]


def test_writer_handles_nullable_legacy_metadata():
    _, rows = facts_to_graph_rows(pd.DataFrame([fact(chunk_index=float("nan"), model=pd.NA)]))
    driver = RecordingDriver()
    merge_relationships(driver, "neo4j", rows, ["suppliesTo"])
    record = json.loads(driver.calls[0][1]["rows"][0]["evidence_record"])
    assert record["chunk_index"] is None and record["model"] is None


def test_writer_validates_all_types_before_any_write():
    nodes, rows = facts_to_graph_rows(pd.DataFrame([fact()]))
    driver = RecordingDriver()
    nodes.append({**nodes[0], "label": "Unsupported"})
    with pytest.raises(ValueError):
        merge_nodes(driver, "neo4j", nodes, ["Company"])
    assert driver.calls == []
    rows.append({**rows[0], "type": "Unsupported"})
    with pytest.raises(ValueError):
        merge_relationships(driver, "neo4j", rows, ["suppliesTo"])
    assert driver.calls == []


def test_writer_accepts_single_use_ontology_iterables():
    nodes, _ = facts_to_graph_rows(pd.DataFrame([fact(object_type="Location")]))
    driver = RecordingDriver()
    merge_nodes(driver, "neo4j", nodes, iter(["Company", "Location"]))
    assert len(driver.calls) == 2


def test_distinct_quotes_in_one_document_survive_resume_upserts(tmp_path):
    quotes = ["CATL supplies BMW.", "BMW purchases cells from CATL."]
    graph = Neo4jGraph(nodes=[], relationships=[
        Neo4jRelationship(start_node_id="chunk0:s", end_node_id="chunk0:t",
                          type="suppliesTo", properties={"evidence": quote})
        for quote in quotes
    ])
    resolution = {
        "chunk0:s": {"canonical_id": "company:catl", "label": "Company",
                     "canonical_name": "CATL", "mention": "CATL"},
        "chunk0:t": {"canonical_id": "company:bmw", "label": "Company",
                     "canonical_name": "BMW", "mention": "BMW"},
    }
    facts = graph_to_facts(graph, {"doc_id": "d1"}, {"chunk0": 0}, resolution)
    assert len(set(facts.fact_id)) == 2
    path = tmp_path / "facts.parquet"
    assert upsert_facts_parquet(facts, path) == 2
    assert upsert_facts_parquet(facts, path) == 2
    assert set(pd.read_parquet(path).evidence) == set(quotes)
    _, rows = facts_to_graph_rows(facts)
    assert len(rows) == 2
    assert len({(r["start_id"], r["type"], r["end_id"]) for r in rows}) == 1


def test_legacy_fact_id_api_stays_compatible():
    import hashlib

    expected = hashlib.sha256(b"d1|company:catl|suppliesTo|company:bmw").hexdigest()[:16]
    assert fact_id("d1", "company:catl", "suppliesTo", "company:bmw") == expected
    assert fact_id("d1", "company:catl", "suppliesTo", "company:bmw", "q", 0) != expected


def test_document_counts_include_completed_zero_fact_documents(ingest_config):
    path = Path(ingest_config["storage"]["normalized_documents"])
    path.parent.mkdir(parents=True, exist_ok=True)
    docs = [{"doc_id": "empty", "content_hash": "empty"},
            {"doc_id": "pending", "content_hash": "pending"},
            {"doc_id": "d1", "content_hash": "d1"}]
    pd.DataFrame(docs).to_parquet(path, index=False)
    pd.DataFrame([fact()]).to_parquet(ingest_config["storage"]["extracted_facts"], index=False)
    connection = state.connect(Path(ingest_config["storage"]["state_db"]))
    try:
        for doc in docs:
            state.register_document(connection, doc)
        state.set_document_status(connection, "empty", state.COMMITTED)
        state.set_document_status(connection, "d1", state.COMMITTED)
    finally:
        connection.close()
    stats = build_graph_bundle(ingest_config)["statistics"]
    assert stats["documents_processed"] == 2
    assert stats["documents_available"] == 3
    assert stats["documents_with_facts"] == 1
    assert stats["document_states"]["PENDING"] == 1


def test_document_processing_count_is_unknown_without_checkpoint(ingest_config):
    Path(ingest_config["storage"]["state_db"]).unlink(missing_ok=True)
    stats = build_graph_bundle(ingest_config)["statistics"]
    assert stats["documents_processed"] is None
    assert stats["document_states"] is None
