"""Frontend-ready semantic graph export from persisted facts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from schain import state
from schain.config import load_config
from schain.graph_export import (
    build_graph_bundle,
    export_graph,
    graph_fingerprint,
    semantic_edge_id,
)
from schain.kg.facts import FACT_COLUMNS
from schain.names import canonical_id


def _facts(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=FACT_COLUMNS)


def _write_facts(config: dict, rows: list[dict]) -> None:
    path = Path(config["storage"]["extracted_facts"])
    path.parent.mkdir(parents=True, exist_ok=True)
    _facts(rows).to_parquet(path, index=False)


def _base_rows() -> list[dict]:
    catl = canonical_id("Company", "CATL")
    bmw = canonical_id("Company", "BMW")
    ganfeng = canonical_id("Company", "Ganfeng Lithium")
    china = canonical_id("Location", "China")
    return [
        {
            "fact_id": "f1", "doc_id": "docA", "chunk_index": 0,
            "source_url": "https://news.example/a",
            "subject_id": catl, "subject_type": "Company",
            "subject_name": "CATL", "subject_mention": "CATL",
            "relation": "suppliesTo",
            "object_id": bmw, "object_type": "Company",
            "object_name": "BMW", "object_mention": "BMW",
            "evidence": "CATL supplies battery cells to BMW.",
            "model": "stub-model", "extracted_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "fact_id": "f2", "doc_id": "docB", "chunk_index": 0,
            "source_url": "https://news.example/b",
            "subject_id": catl, "subject_type": "Company",
            "subject_name": "CATL", "subject_mention": "Contemporary Amperex Technology",
            "relation": "suppliesTo",
            "object_id": bmw, "object_type": "Company",
            "object_name": "BMW", "object_mention": "BMW AG",
            "evidence": "CATL supplies lithium-ion batteries to BMW.",
            "model": "stub-model", "extracted_at": "2026-01-02T00:00:00+00:00",
        },
        {
            "fact_id": "f3", "doc_id": "docA", "chunk_index": 0,
            "source_url": "https://news.example/a",
            "subject_id": ganfeng, "subject_type": "Company",
            "subject_name": "Ganfeng Lithium", "subject_mention": "Ganfeng Lithium",
            "relation": "suppliesTo",
            "object_id": catl, "object_type": "Company",
            "object_name": "CATL", "object_mention": "CATL",
            "evidence": "Ganfeng Lithium supplies lithium materials to CATL.",
            "model": "stub-model", "extracted_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "fact_id": "f4", "doc_id": "docA", "chunk_index": 0,
            "source_url": "https://news.example/a",
            "subject_id": catl, "subject_type": "Company",
            "subject_name": "CATL", "subject_mention": "CATL",
            "relation": "locatedIn",
            "object_id": china, "object_type": "Location",
            "object_name": "China", "object_mention": "China",
            "evidence": "CATL is located in China.",
            "model": "stub-model", "extracted_at": "2026-01-01T00:00:00+00:00",
        },
    ]


def test_canonical_entities_become_one_node_each(ingest_config):
    _write_facts(ingest_config, _base_rows())
    bundle = build_graph_bundle(ingest_config, mode="export")
    catl = canonical_id("Company", "CATL")
    ids = [node["id"] for node in bundle["nodes"]]
    assert ids.count(catl) == 1
    assert len(ids) == len(set(ids))


def test_duplicate_semantic_edges_aggregate_with_two_documents(ingest_config):
    _write_facts(ingest_config, _base_rows())
    bundle = build_graph_bundle(ingest_config, mode="export")
    catl = canonical_id("Company", "CATL")
    bmw = canonical_id("Company", "BMW")
    edge = next(
        e for e in bundle["edges"]
        if e["source"] == catl and e["target"] == bmw and e["relation"] == "suppliesTo"
    )
    assert edge["support_count"] == 2
    assert edge["evidence_count"] == 2
    assert len(edge["evidence"]) == 2
    assert set(edge["source_doc_ids"]) == {"docA", "docB"}
    assert edge["id"] == semantic_edge_id(catl, "suppliesTo", bmw)


def test_supplies_to_direction_is_supplier_to_customer(ingest_config):
    _write_facts(ingest_config, _base_rows())
    bundle = build_graph_bundle(ingest_config, mode="export")
    supplies = {
        (e["source"], e["target"])
        for e in bundle["edges"]
        if e["relation"] == "suppliesTo"
    }
    catl = canonical_id("Company", "CATL")
    bmw = canonical_id("Company", "BMW")
    ganfeng = canonical_id("Company", "Ganfeng Lithium")
    assert (catl, bmw) in supplies
    assert (ganfeng, catl) in supplies
    assert (bmw, catl) not in supplies


def test_aliases_aggregate_from_mentions_and_resolution(ingest_config):
    _write_facts(ingest_config, _base_rows())
    resolution = Path(ingest_config["storage"]["entity_resolution"])
    resolution.parent.mkdir(parents=True, exist_ok=True)
    catl = canonical_id("Company", "CATL")
    pd.DataFrame(
        [{
            "doc_id": "docA",
            "node_id": "0",
            "label": "Company",
            "canonical_id": catl,
            "canonical_name": "CATL",
            "aliases": ["CATL", "Contemporary Amperex Technology Co. Limited"],
            "mention": "CATL",
            "method": "llm",
        }]
    ).to_parquet(resolution, index=False)

    bundle = build_graph_bundle(ingest_config, mode="export")
    node = next(n for n in bundle["nodes"] if n["id"] == catl)
    assert "CATL" in node["aliases"]
    assert "Contemporary Amperex Technology Co. Limited" in node["aliases"]
    assert "Contemporary Amperex Technology" in node["aliases"]
    assert node["aliases"] == sorted(node["aliases"], key=str.casefold)


def test_missing_resolution_and_state_do_not_break_export(ingest_config):
    _write_facts(ingest_config, _base_rows())
    # No entity_resolution.parquet, no state.db — still works.
    Path(ingest_config["storage"]["entity_resolution"]).unlink(missing_ok=True)
    Path(ingest_config["storage"]["state_db"]).unlink(missing_ok=True)
    bundle = build_graph_bundle(ingest_config, mode="export")
    assert bundle["statistics"]["nodes"] >= 1
    assert all(node["discovery_tier"] is None for node in bundle["nodes"])


def test_frontier_tier_enriches_company_nodes(expand_config):
    rows = _base_rows()
    for row in rows:
        row.update(ontology_valid=True, evidence_validated=True,
                   evidence_quote_grounded=True, direction_explicit=True,
                   expansion_relevant=row["relation"] == "suppliesTo")
    _write_facts(expand_config, rows)
    connection = state.connect(Path(expand_config["storage"]["state_db"]))
    try:
        state.enqueue_entity(connection, "BMW", 0)
        state.enqueue_entity(connection, "CATL", 1, discovered_from=canonical_id("Company", "BMW"))
    finally:
        connection.close()

    bundle = build_graph_bundle(expand_config, mode="export")
    by_id = {node["id"]: node for node in bundle["nodes"]}
    bmw = by_id[canonical_id("Company", "BMW")]
    catl = by_id[canonical_id("Company", "CATL")]
    assert bmw["discovery_tier"] == 0 and bmw["is_seed"] is True
    assert catl["discovery_tier"] == 1 and catl["is_seed"] is False
    assert "BMW" in bundle["metadata"]["seeds"]


def test_documents_enriched_without_full_text_or_absolute_paths(ingest_config, tmp_path):
    rows = _base_rows()
    _write_facts(ingest_config, rows)
    docs_path = Path(ingest_config["storage"]["normalized_documents"])
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    absolute = str((tmp_path / "reports" / "supply.pdf").resolve())
    pd.DataFrame(
        [{
            "doc_id": "docA",
            "content_hash": "abc",
            "source_path": absolute,
            "source_url": "https://news.example/a",
            "title": "Supply news",
            "text": "FULL DOCUMENT BODY THAT MUST NOT BE EXPORTED " * 20,
            "char_count": 100,
            "content_source": "full_text",
            "ingested_at": "2026-01-01T00:00:00+00:00",
        }]
    ).to_parquet(docs_path, index=False)

    search_path = Path(ingest_config["storage"]["search_results"])
    pd.DataFrame(
        [{
            "provider": "google_news",
            "query": "CATL suppliers",
            "title": "Supply news",
            "url": "https://news.example/a",
            "snippet": "…",
            "text": None,
            "published": "2026-01-01",
            "language": "en",
            "entity_key": None,
            "tier": None,
            "search_id": "s1",
            "retrieved_at": "2026-01-01T00:00:00+00:00",
            "content_source": "title_snippet",
        }]
    ).to_parquet(search_path, index=False)

    bundle = build_graph_bundle(ingest_config, mode="export")
    doc = next(d for d in bundle["documents"] if d["doc_id"] == "docA")
    assert doc["source_file"] == "supply.pdf"
    assert absolute not in json.dumps(bundle)
    assert "FULL DOCUMENT BODY" not in json.dumps(bundle)
    assert "text" not in doc
    assert doc["provider"] == "google_news"
    assert doc["published_at"] == "2026-01-01"


def test_output_is_valid_json_without_nan(ingest_config, tmp_path):
    rows = _base_rows()
    rows[0]["source_url"] = float("nan")
    _write_facts(ingest_config, rows)
    path = export_graph(ingest_config, output=tmp_path / "graph.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    dumped = json.dumps(loaded)
    assert "NaN" not in dumped and "Infinity" not in dumped
    assert loaded["schema_version"] == "1.0"


def test_edge_endpoints_exist_in_nodes(ingest_config):
    _write_facts(ingest_config, _base_rows())
    bundle = build_graph_bundle(ingest_config, mode="export")
    ids = {node["id"] for node in bundle["nodes"]}
    for edge in bundle["edges"]:
        assert edge["source"] in ids and edge["target"] in ids


def test_empty_facts_produce_valid_empty_bundle(ingest_config):
    _write_facts(ingest_config, [])
    bundle = build_graph_bundle(ingest_config, mode="export")
    assert bundle["nodes"] == []
    assert bundle["edges"] == []
    assert bundle["documents"] == []
    assert bundle["statistics"]["facts"] == 0
    assert bundle["statistics"]["nodes"] == 0


def test_missing_facts_file_produces_empty_bundle(ingest_config):
    Path(ingest_config["storage"]["extracted_facts"]).unlink(missing_ok=True)
    bundle = build_graph_bundle(ingest_config, mode="export")
    assert bundle["nodes"] == [] and bundle["edges"] == []


def test_export_is_deterministic_apart_from_generated_at(ingest_config):
    _write_facts(ingest_config, _base_rows())
    first = build_graph_bundle(ingest_config, mode="export")
    second = build_graph_bundle(ingest_config, mode="export")
    assert first["metadata"]["graph_fingerprint"] == second["metadata"]["graph_fingerprint"]
    a = copy.deepcopy(first)
    b = copy.deepcopy(second)
    a["metadata"].pop("generated_at")
    b["metadata"].pop("generated_at")
    assert a == b


def test_fingerprint_ignores_generated_at(ingest_config):
    _write_facts(ingest_config, _base_rows())
    bundle = build_graph_bundle(ingest_config, mode="export")
    again = graph_fingerprint(
        [n["id"] for n in bundle["nodes"]],
        [e["id"] for e in bundle["edges"]],
        [fid for e in bundle["edges"] for fid in e["fact_ids"]],
    )
    assert again == bundle["metadata"]["graph_fingerprint"]
