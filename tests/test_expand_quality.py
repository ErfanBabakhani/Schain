"""EXPAND quality revision: origin relevance, conservative resolve, tiers, budgets."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from neo4j_graphrag.components.types import Neo4jGraph, Neo4jNode

from schain.config import load_config
from schain.expansion.loop import _entity_document_limits
from schain.graph_export import _apply_tier_semantics, build_graph_bundle
from schain.kg.facts import FACT_COLUMNS, is_expansion_relevant_fact, upstream_companies
from schain.kg.resolve import identity_plausible, looks_like_facility, resolve_entities, sanitize_llm_groups
from schain.kg.support import aggregate_edge_support, assign_quality_status
from schain.kg.validate import is_generic_company_phrase, validate_facts_frame
from schain.names import canonical_id
from schain.ontology import load_ontology
from schain.sources.base import SearchResult
from schain.sources.origin import classify_origin_relevance


REPO_ROOT = __file__  # unused; fixtures come from conftest


def _fact(**kwargs):
    row = {column: None for column in FACT_COLUMNS}
    row.update(kwargs)
    return row


def test_byd_does_not_merge_with_catl():
    assert not identity_plausible("BYD", "CATL")
    groups = sanitize_llm_groups([["BYD", "CATL"]])
    assert sorted(len(g) for g in groups) == [1, 1]


def test_blade_batteries_does_not_merge_with_ultium():
    assert not identity_plausible("Blade Batteries", "Ultium battery pack")
    groups = sanitize_llm_groups([["Blade Batteries", "Ultium battery pack"]])
    assert all(len(g) == 1 for g in groups)


def test_ventilated_rotors_do_not_merge_with_carbon_sleeved():
    assert not identity_plausible("ventilated rotors", "carbon-sleeved rotors")


def test_bmw_plant_landshut_does_not_become_bmw(ontology):
    assert looks_like_facility("BMW Group Plant Landshut")
    assert not identity_plausible("BMW Group Plant Landshut", "BMW")
    graph = Neo4jGraph(
        nodes=[
            Neo4jNode(id="0", label="Company", properties={"name": "BMW Group Plant Landshut"}),
            Neo4jNode(id="1", label="Company", properties={"name": "BMW"}),
        ],
        relationships=[],
    )
    # Even if the LLM wrongly merges them, sanitize keeps them apart.
    class BadLLM:
        def invoke(self, prompt):
            class R:
                content = '{"groups": [["BMW Group Plant Landshut", "BMW"]]}'

            return R()

    resolution, _ = resolve_entities(
        graph, llm=BadLLM(), aliases=ontology.get("aliases"),
    )
    assert resolution["0"]["canonical_id"] != resolution["1"]["canonical_id"]


def test_uncertain_pairs_remain_separate():
    # Empty/disjoint → refuse
    assert not identity_plausible("Acme Motors", "Globex Corp")


def test_uses_battery_pack_does_not_imply_produces(ingest_config, ontology):
    facts = pd.DataFrame([
        _fact(
            fact_id="p1", doc_id="d",
            subject_id=canonical_id("Company", "Lucid"),
            subject_type="Company", subject_name="Lucid",
            relation="produces",
            object_id=canonical_id("Product", "battery pack"),
            object_type="Product", object_name="battery pack",
            evidence="Lucid Gravity uses a 120-kWh battery pack.",
        ),
    ])
    from schain.kg import validate as validate_mod

    def unclear(llm, rows):
        return {row["candidate_fact_id"]: "unclear" for row in rows}

    with patch.object(validate_mod, "_batch_evidence_validate", side_effect=unclear):
        accepted, rejected = validate_facts_frame(
            facts, config=ingest_config, ontology=ontology, llm=object(),
        )
    assert accepted.empty
    assert rejected.iloc[0]["rejection_stage"] == "evidence"


def test_features_rotors_does_not_imply_produces(ingest_config, ontology):
    facts = pd.DataFrame([
        _fact(
            fact_id="p2", doc_id="d",
            subject_id=canonical_id("Company", "Brembo"),
            subject_type="Company", subject_name="Brembo",
            relation="produces",
            object_id=canonical_id("Product", "ventilated rotors"),
            object_type="Product", object_name="ventilated rotors",
            evidence="The car features ventilated rotors from a partner brand.",
        ),
    ])
    from schain.kg import validate as validate_mod

    def unclear(llm, rows):
        return {row["candidate_fact_id"]: "unclear" for row in rows}

    with patch.object(validate_mod, "_batch_evidence_validate", side_effect=unclear):
        accepted, rejected = validate_facts_frame(
            facts, config=ingest_config, ontology=ontology, llm=object(),
        )
    assert accepted.empty


def test_invests_in_mining_does_not_imply_supplies(ingest_config, ontology):
    facts = pd.DataFrame([
        _fact(
            fact_id="s1", doc_id="d",
            subject_id=canonical_id("Mine", "Lithium Mine"),
            subject_type="Mine", subject_name="Lithium Mine",
            relation="suppliesTo",
            object_id=canonical_id("Company", "BYD"),
            object_type="Company", object_name="BYD",
            evidence="BYD invests in lithium mining operations to extract raw materials directly.",
        ),
    ])
    from schain.kg import validate as validate_mod

    def unclear(llm, rows):
        return {row["candidate_fact_id"]: "unclear" for row in rows}

    with patch.object(validate_mod, "_batch_evidence_validate", side_effect=unclear):
        accepted, rejected = validate_facts_frame(
            facts, config=ingest_config, ontology=ontology, llm=object(),
        )
    assert accepted.empty


def test_explicit_supplies_accepted(ingest_config, ontology):
    facts = pd.DataFrame([
        _fact(
            fact_id="s2", doc_id="d",
            subject_id=canonical_id("Company", "Qualcomm"),
            subject_type="Company", subject_name="Qualcomm",
            relation="suppliesTo",
            object_id=canonical_id("Company", "BMW"),
            object_type="Company", object_name="BMW",
            evidence="BMW names Qualcomm lead chip supplier for next-gen platforms.",
            expansion_relevant=True,
            origin_entity=canonical_id("Company", "BMW"),
        ),
    ])
    from schain.kg import validate as validate_mod

    def supports(llm, rows):
        return {row["candidate_fact_id"]: "supports" for row in rows}

    with patch.object(validate_mod, "_batch_evidence_validate", side_effect=supports):
        accepted, rejected = validate_facts_frame(
            facts, config=ingest_config, ontology=ontology, llm=object(),
        )
    assert len(accepted) == 1
    assert rejected.empty


def test_brembo_result_without_brembo_is_origin_irrelevant():
    result = SearchResult(
        provider="general_web",
        query='"Brembo" suppliers',
        title="Tesla Lucid Rivian compare EV SUV specifications",
        url="https://example.com/ev-suv",
        snippet="Tesla, Lucid and Rivian battle for SUV market share.",
        text=(
            "Tesla Lucid Rivian Norton dominate the EV SUV segment with new platforms. "
            "Supplier portals and sourcing software are also discussed without naming Brembo."
        ),
    )
    ok, reason = classify_origin_relevance(result, "Brembo")
    assert ok is False
    assert reason in ("no_entity_mention", "generic_result", "weak_context")


def test_unrelated_global_fact_cannot_advance_brembo_frontier(ingest_config):
    brembo = canonical_id("Company", "Brembo")
    facts = pd.DataFrame([
        _fact(
            fact_id="g1", doc_id="d1",
            subject_id=canonical_id("Company", "Tesla"),
            subject_type="Company", subject_name="Tesla",
            relation="produces",
            object_id=canonical_id("Product", "Model X"),
            object_type="Product", object_name="Model X",
            evidence="Tesla produces the Model X.",
            ontology_valid=True,
            evidence_validated=True,
            expansion_relevant=False,
            origin_entity=brembo,
        ),
        _fact(
            fact_id="g2", doc_id="d1",
            subject_id=canonical_id("Company", "SupplierX"),
            subject_type="Company", subject_name="SupplierX",
            relation="suppliesTo",
            object_id=brembo,
            object_type="Company", object_name="Brembo",
            evidence="SupplierX supplies brake calipers to Brembo.",
            ontology_valid=True,
            evidence_validated=True,
            expansion_relevant=True,
            direction_explicit=True,
            evidence_quote_grounded=True,
            origin_entity=brembo,
        ),
    ])
    support = aggregate_edge_support(facts, config=ingest_config)
    tesla_rows = support[support.subject_name == "Tesla"]
    if not tesla_rows.empty:
        assert not bool(tesla_rows.iloc[0]["frontier_eligible"])
    sx = support[support.subject_name == "SupplierX"]
    assert bool(sx.iloc[0]["frontier_eligible"])
    assert "SupplierX" in upstream_companies(facts, {brembo}, ["suppliesTo"])
    assert "Tesla" not in upstream_companies(facts, {brembo}, ["suppliesTo"])


def test_supplier_of_brembo_becomes_tier2_semantics():
    nodes = [
        {
            "id": "company:bmw", "type": "Company", "name": "BMW",
            "discovery_tier": 0, "is_seed": True, "tier_parent_ids": [],
            "is_supply_chain_node": True,
        },
        {
            "id": "company:brembo", "type": "Company", "name": "Brembo",
            "discovery_tier": 1, "is_seed": False,
            "tier_parent_ids": ["company:bmw"], "tier_parent_id": "company:bmw",
            "is_supply_chain_node": True,
        },
        {
            "id": "company:supplier-x", "type": "Company", "name": "SupplierX",
            "discovery_tier": 2, "is_seed": False,
            "tier_parent_ids": ["company:brembo"], "tier_parent_id": "company:brembo",
            "is_supply_chain_node": True,
        },
        {
            "id": "product:caliper", "type": "Product", "name": "caliper",
            "discovery_tier": 2, "is_seed": False, "tier_parent_ids": [],
            "is_supply_chain_node": False,
        },
    ]
    edges = [
        {
            "source": "company:brembo", "target": "company:bmw",
            "relation": "suppliesTo", "expansion_relevant": True,
            "quality": {"expansion_relevant": True, "frontier_eligible": True},
        },
        {
            "source": "company:supplier-x", "target": "company:brembo",
            "relation": "suppliesTo", "expansion_relevant": True,
            "quality": {"expansion_relevant": True, "frontier_eligible": True},
        },
    ]
    stats = _apply_tier_semantics(nodes, edges)
    assert stats["tier_invariant_violations"] == 0
    by_id = {n["id"]: n for n in nodes}
    assert by_id["company:supplier-x"]["discovery_tier"] == 2
    assert by_id["product:caliper"]["discovery_tier"] is None


def test_context_types_never_keep_supply_tier():
    nodes = [
        {
            "id": "location:germany", "type": "Location", "name": "Germany",
            "discovery_tier": 1, "is_seed": False, "tier_parent_ids": [],
        },
        {
            "id": "person:ceo", "type": "Person", "name": "CEO",
            "discovery_tier": 1, "is_seed": False, "tier_parent_ids": [],
        },
    ]
    stats = _apply_tier_semantics(nodes, [])
    assert nodes[0]["discovery_tier"] is None
    assert nodes[1]["discovery_tier"] is None
    assert stats["tier_invariant_violations"] == 0


def test_tier_invariant_violation_demotes_node():
    nodes = [
        {
            "id": "company:orphan", "type": "Company", "name": "Orphan",
            "discovery_tier": 2, "is_seed": False, "tier_parent_ids": [],
            "is_supply_chain_node": True,
        },
    ]
    stats = _apply_tier_semantics(nodes, [])
    assert stats["tier_invariant_violations"] == 1
    assert nodes[0]["discovery_tier"] is None
    assert nodes[0]["is_supply_chain_node"] is False


def test_generic_chip_suppliers_still_blocked():
    assert is_generic_company_phrase("chip suppliers")


def test_expansion_relevant_helper():
    assert is_expansion_relevant_fact(
        relation="suppliesTo",
        object_id="company:bmw",
        subject_type="Company",
        origin_entity="company:bmw",
    )
    assert not is_expansion_relevant_fact(
        relation="produces",
        object_id="product:x",
        subject_type="Company",
        origin_entity="company:bmw",
    )


def test_source_score_still_does_not_gate():
    _, eligible, _, _ = assign_quality_status(
        ontology_valid=True,
        evidence_validated=True,
        relation="suppliesTo",
        subject_type="Company",
        max_source_score=0,
        expansion_relevant=True,
        direction_explicit=True,
    )
    assert eligible


def test_document_budget_defaults(expand_config):
    target, hard = _entity_document_limits(expand_config.get("expansion") or {})
    assert expand_config["expansion"]["max_documents"] == 320
    assert target == 24
    assert hard == 36


def test_search_breadth_defaults(expand_config):
    defaults = expand_config["expansion"]["search"]["defaults"]
    assert defaults["max_results"] == 14
    assert defaults["max_results_by_provider"]["wikipedia"] == 4
    assert defaults["max_results_by_provider"]["general_web"] == 18
    assert defaults["max_results_by_provider"]["google_news"] == 10
    assert defaults["max_results_by_provider"]["gdelt"] == 16
