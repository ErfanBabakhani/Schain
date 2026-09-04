"""Visualization layer: aggregated directed graphs from facts.parquet."""

from __future__ import annotations

import pandas as pd

from schain.kg.facts import FACT_COLUMNS
from schain.visualization import (
    build_document_graph,
    render_all_graph,
    render_document_graph,
)


def _facts_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=FACT_COLUMNS)


SMOKE_ROWS = [
    {
        "fact_id": "a1", "doc_id": "smoke", "chunk_index": 0, "source_url": None,
        "subject_id": "company:ganfeng", "subject_type": "Company",
        "subject_name": "Ganfeng Lithium", "subject_mention": "Ganfeng Lithium",
        "relation": "suppliesTo",
        "object_id": "company:catl", "object_type": "Company",
        "object_name": "CATL", "object_mention": "CATL",
        "evidence": "Ganfeng Lithium supplies lithium materials to CATL.",
        "model": "Qwen/Qwen3-4B-Instruct-2507", "extracted_at": "2026-01-01T00:00:00+00:00",
    },
    {
        "fact_id": "a2", "doc_id": "smoke", "chunk_index": 0, "source_url": None,
        "subject_id": "company:catl", "subject_type": "Company",
        "subject_name": "CATL", "subject_mention": "CATL",
        "relation": "suppliesTo",
        "object_id": "company:bmw", "object_type": "Company",
        "object_name": "BMW", "object_mention": "BMW",
        "evidence": "CATL supplies lithium-ion batteries to BMW.",
        "model": "Qwen/Qwen3-4B-Instruct-2507", "extracted_at": "2026-01-01T00:00:00+00:00",
    },
    {
        "fact_id": "a3", "doc_id": "smoke", "chunk_index": 0, "source_url": None,
        "subject_id": "company:catl", "subject_type": "Company",
        "subject_name": "CATL", "subject_mention": "CATL",
        "relation": "locatedIn",
        "object_id": "location:china", "object_type": "Location",
        "object_name": "China", "object_mention": "China",
        "evidence": "CATL is located in China.",
        "model": "Qwen/Qwen3-4B-Instruct-2507", "extracted_at": "2026-01-01T00:00:00+00:00",
    },
    {
        "fact_id": "a4", "doc_id": "smoke", "chunk_index": 0, "source_url": None,
        "subject_id": "company:ganfeng", "subject_type": "Company",
        "subject_name": "Ganfeng Lithium", "subject_mention": "Ganfeng Lithium",
        "relation": "owns",
        "object_id": "mine:mariana", "object_type": "Mine",
        "object_name": "Mariana lithium mine", "object_mention": "Mariana lithium mine",
        "evidence": "Ganfeng Lithium owns the Mariana lithium mine.",
        "model": "Qwen/Qwen3-4B-Instruct-2507", "extracted_at": "2026-01-01T00:00:00+00:00",
    },
]


def test_one_canonical_entity_produces_one_node():
    facts = _facts_frame(SMOKE_ROWS)
    graph = build_document_graph(facts)
    names = [node.name for node in graph.nodes]
    assert names.count("CATL") == 1
    assert len({node.id for node in graph.nodes}) == len(graph.nodes)


def test_duplicate_fact_rows_do_not_duplicate_nodes():
    rows = SMOKE_ROWS + [
        {**SMOKE_ROWS[1], "fact_id": "dup", "evidence": "Again: CATL supplies BMW."},
    ]
    graph = build_document_graph(_facts_frame(rows))
    assert sum(1 for node in graph.nodes if node.id == "company:catl") == 1
    assert sum(1 for node in graph.nodes if node.id == "company:bmw") == 1


def test_duplicate_semantic_edges_aggregate_with_evidence():
    rows = [
        SMOKE_ROWS[1],
        {**SMOKE_ROWS[1], "fact_id": "b2", "evidence": "Second source: CATL supplies BMW."},
        {**SMOKE_ROWS[1], "fact_id": "b3", "evidence": "Third confirmation."},
    ]
    graph = build_document_graph(_facts_frame(rows))
    supplies = [e for e in graph.edges if e.relation == "suppliesTo" and e.target_id == "company:bmw"]
    assert len(supplies) == 1
    assert supplies[0].support_count == 3
    assert [s.evidence for s in supplies[0].supports] == [
        "CATL supplies lithium-ion batteries to BMW.",
        "Second source: CATL supplies BMW.",
        "Third confirmation.",
    ]


def test_supplies_to_direction_is_supplier_to_customer():
    graph = build_document_graph(_facts_frame(SMOKE_ROWS))
    supplies = {(e.source_name, e.target_name) for e in graph.edges if e.relation == "suppliesTo"}
    assert ("CATL", "BMW") in supplies
    assert ("Ganfeng Lithium", "CATL") in supplies
    assert ("BMW", "CATL") not in supplies


def test_relation_direction_preserved_for_all_edges():
    graph = build_document_graph(_facts_frame(SMOKE_ROWS))
    owns = next(e for e in graph.edges if e.relation == "owns")
    assert owns.source_name == "Ganfeng Lithium"
    assert owns.target_name == "Mariana lithium mine"
    located = next(e for e in graph.edges if e.relation == "locatedIn")
    assert located.source_id == "company:catl"
    assert located.target_id == "location:china"


def test_evidence_and_provenance_retained_on_edge():
    graph = build_document_graph(_facts_frame(SMOKE_ROWS))
    edge = next(e for e in graph.edges if e.relation == "suppliesTo" and e.source_id == "company:catl")
    support = edge.supports[0]
    assert "lithium-ion batteries" in support.evidence
    assert support.doc_id == "smoke"
    assert support.model == "Qwen/Qwen3-4B-Instruct-2507"
    assert support.chunk_index == 0
    assert support.extracted_at.startswith("2026-01-01")


def test_empty_facts_produce_friendly_empty_graph():
    graph = build_document_graph(_facts_frame([]))
    assert graph.entity_count == 0
    assert graph.edge_count == 0
    assert graph.fact_count == 0


def test_html_output_is_created(tmp_path):
    facts = _facts_frame(SMOKE_ROWS)
    path = render_document_graph(facts, tmp_path / "smoke.html", doc_id="smoke")
    assert path.exists()
    body = path.read_text(encoding="utf-8")
    assert "vis-network" in body
    assert "CATL" in body
    assert "suppliesTo" in body
    assert "Ganfeng Lithium" in body
    assert "supportCount" in body or "support count" in body.lower() or "support" in body


def test_render_all_writes_aggregate_and_per_document(tmp_path):
    facts = _facts_frame(SMOKE_ROWS + [
        {**SMOKE_ROWS[0], "fact_id": "other", "doc_id": "other-doc"},
    ])
    written = render_all_graph(facts, tmp_path / "viz")
    assert "all" in written
    assert written["all"].exists()
    assert "smoke" in written
    assert "other-doc" in written


def test_visualization_does_not_mutate_source_facts(tmp_path):
    facts = _facts_frame(SMOKE_ROWS)
    before = facts.copy(deep=True)
    build_document_graph(facts)
    render_document_graph(facts, tmp_path / "out.html")
    pd.testing.assert_frame_equal(facts, before)


def test_future_relation_type_is_handled_generically():
    rows = [{
        **SMOKE_ROWS[0],
        "fact_id": "x1",
        "relation": "jointVenturesWith",
        "object_id": "company:bmw",
        "object_name": "BMW",
        "object_type": "Company",
        "object_mention": "BMW",
        "evidence": "Hypothetical future relation.",
    }]
    graph = build_document_graph(_facts_frame(rows))
    assert any(e.relation == "jointVenturesWith" for e in graph.edges)
