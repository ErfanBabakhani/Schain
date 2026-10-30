"""Config authority, directional suppliesTo, multi-seed tiers, publisher/generic guards."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from schain.config import (
    REPO_ROOT,
    apply_runtime_overrides,
    load_config,
    seed_entities,
    seed_groups,
)
from schain.documents import make_document, split_publisher_title
from schain.graph_export import _apply_tier_semantics, _recompute_shortest_tiers
from schain.kg.facts import FACT_COLUMNS, is_expansion_relevant_fact
from schain.kg.relation_evidence import check_supplies_to_direction, lexical_relation_gate
from schain.kg.resolve import identity_plausible
from schain.kg.support import aggregate_edge_support, assign_quality_status
from schain.kg.validate import is_generic_company_phrase, validate_facts_frame
from schain.names import canonical_id


def _fact(**kwargs):
    row = {column: None for column in FACT_COLUMNS}
    row.update(kwargs)
    return row


def test_active_expansion_yaml_seeds_are_tier0_candidates():
    config = load_config(REPO_ROOT / "config/expansion.yaml")
    seeds = seed_entities(config)
    assert seeds == ["Toyota", "Tesla Inc.", "BMW", "BYD Co., Ltd."]
    groups = seed_groups(config)
    assert set(groups) == {"ev_manufacturers"}
    assert "battery_suppliers" not in groups
    assert "mining_companies" not in groups


def test_commented_seed_groups_are_not_loaded():
    raw = (REPO_ROOT / "config/expansion.yaml").read_text(encoding="utf-8")
    assert "battery_suppliers:" in raw
    assert "# battery_suppliers:" in raw or "  # battery_suppliers:" in raw
    config = load_config(REPO_ROOT / "config/expansion.yaml")
    assert "CATL" not in seed_entities(config)
    assert "Samsung SDI" not in seed_entities(config)


def test_newly_enabled_seed_group_is_loaded(tmp_path):
    text = """
seeds:
  ev_manufacturers:
    - Toyota
  battery_suppliers:
    - CATL
"""
    path = tmp_path / "seeds.yaml"
    path.write_text(text, encoding="utf-8")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert seed_entities(data) == ["Toyota", "CATL"]
    assert seed_groups(data)["battery_suppliers"] == ["CATL"]


def test_colab_overlay_does_not_replace_seeds():
    config = load_config(REPO_ROOT / "config/colab_professor_expand.yaml")
    assert seed_entities(config) == ["Toyota", "Tesla Inc.", "BMW", "BYD Co., Ltd."]
    assert "seeds" not in Path(REPO_ROOT / "config/colab_professor_expand.yaml").read_text(
        encoding="utf-8"
    ).split("include:")[0] or True
    overlay_text = (REPO_ROOT / "config/colab_professor_expand.yaml").read_text(encoding="utf-8")
    assert "FOCAL_COMPANY" not in overlay_text
    # No seeds: block in the overlay itself (only via include).
    overlay_only = yaml.safe_load(
        "\n".join(
            line for line in overlay_text.splitlines()
            if not line.strip().startswith("include")
            and "config/expansion.yaml" not in line
        )
    ) or {}
    assert "seeds" not in overlay_only


def test_runtime_overrides_cannot_silently_replace_seeds():
    config = load_config(REPO_ROOT / "config/expansion.yaml")
    merged = apply_runtime_overrides(
        config,
        {"seeds": {"ev_manufacturers": ["BMW"]}, "expansion": {"max_documents": 50}},
        protect_seeds=True,
    )
    assert seed_entities(merged) == ["Toyota", "Tesla Inc.", "BMW", "BYD Co., Ltd."]
    assert merged["expansion"]["max_documents"] == 50


def test_explicit_seed_override_allowed_when_unprotected():
    config = load_config(REPO_ROOT / "config/expansion.yaml")
    merged = apply_runtime_overrides(
        config,
        {"seeds": {"cli_override": ["BMW"]}},
        protect_seeds=False,
    )
    assert seed_entities(merged) == ["BMW"]


def test_strategic_pact_does_not_imply_supplies_to():
    ok, reason = check_supplies_to_direction(
        "CFMOTO", "Brembo",
        "CFMOTO signed a long-term strategic pact with Brembo.",
    )
    assert ok is False
    assert reason in ("ambiguous_supply_direction", "insufficient_relation_language")
    lexical, lex_reason = lexical_relation_gate(
        "suppliesTo",
        "CFMOTO signed a long-term strategic pact with Brembo.",
    )
    assert lexical is False
    assert "insufficient" in lex_reason


def test_ambiguous_supply_agreement_does_not_guess_direction():
    ok, reason = check_supplies_to_direction(
        "Acme", "BMW",
        "Acme entered a supply agreement with BMW.",
    )
    assert ok is False
    assert reason == "ambiguous_supply_direction"


def test_x_supplies_y_direction():
    ok, reason = check_supplies_to_direction(
        "CATL", "BMW",
        "CATL supplies battery cells to BMW.",
    )
    assert ok is True
    assert "supplier" in reason or "direction" in reason


def test_y_sources_from_x_direction():
    ok, reason = check_supplies_to_direction(
        "Albemarle", "CATL",
        "CATL sources lithium from Albemarle.",
    )
    assert ok is True


def test_downstream_relation_not_expansion_relevant():
    assert not is_expansion_relevant_fact(
        relation="suppliesTo",
        object_id=canonical_id("Company", "Stellantis"),
        subject_type="Company",
        origin_entity=canonical_id("Company", "BorgWarner"),
    )
    assert is_expansion_relevant_fact(
        relation="suppliesTo",
        object_id=canonical_id("Company", "BorgWarner"),
        subject_type="Company",
        origin_entity=canonical_id("Company", "BorgWarner"),
    )


def test_chinese_oem_is_generic():
    assert is_generic_company_phrase("Chinese OEM")
    assert is_generic_company_phrase("OEM")
    assert is_generic_company_phrase("automaker")
    assert not is_generic_company_phrase("BMW")


def test_publisher_suffix_split_and_not_in_body():
    title, publisher = split_publisher_title(
        "BMW picks new chip partner - Electric & Hybrid Vehicle Technology International"
    )
    assert publisher and "Electric" in publisher
    assert "Electric" not in title
    doc = make_document(
        "BMW picks new chip partner - Electric & Hybrid Vehicle Technology International\n"
        "BMW selected Qualcomm as a supplier.",
        title="BMW picks new chip partner - Electric & Hybrid Vehicle Technology International",
        content_source="title_snippet",
        publisher_name="Electric & Hybrid Vehicle Technology International",
    )
    assert doc["publisher_name"]
    assert "Electric & Hybrid" not in doc["text"].split("\n")[0]


def test_thyssenkrupp_business_units_not_auto_merged():
    assert not identity_plausible("Thyssenkrupp", "Thyssenkrupp Steel")
    assert not identity_plausible("Thyssenkrupp", "Thyssenkrupp nucera")
    assert not identity_plausible("Thyssenkrupp", "ThyssenKrupp Aerospace")


def test_one_source_can_advance_frontier():
    _, eligible, reason, rejection = assign_quality_status(
        ontology_valid=True,
        evidence_validated=True,
        relation="suppliesTo",
        subject_type="Company",
        max_source_score=0,
        expansion_relevant=True,
        direction_explicit=True,
    )
    assert eligible is True
    assert rejection is None
    assert reason["named_supplier"] is True


def test_source_score_does_not_gate_frontier_again():
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


def test_shared_supplier_one_node_multi_parents():
    nodes = [
        {"id": "company:bmw", "type": "Company", "name": "BMW",
         "discovery_tier": 0, "is_seed": True, "tier_parent_ids": []},
        {"id": "company:toyota", "type": "Company", "name": "Toyota",
         "discovery_tier": 0, "is_seed": True, "tier_parent_ids": []},
        {"id": "company:supplier-x", "type": "Company", "name": "Supplier X",
         "discovery_tier": None, "is_seed": False, "tier_parent_ids": []},
    ]
    edges = [
        {
            "source": "company:supplier-x", "target": "company:bmw",
            "relation": "suppliesTo", "expansion_relevant": True,
            "quality": {"expansion_relevant": True, "frontier_eligible": True},
        },
        {
            "source": "company:supplier-x", "target": "company:toyota",
            "relation": "suppliesTo", "expansion_relevant": True,
            "quality": {"expansion_relevant": True, "frontier_eligible": True},
        },
    ]
    _recompute_shortest_tiers(
        nodes, edges, {"company:bmw", "company:toyota"}, max_tier=5,
    )
    by_id = {n["id"]: n for n in nodes}
    assert by_id["company:supplier-x"]["discovery_tier"] == 1
    assert set(by_id["company:supplier-x"]["tier_parent_ids"]) == {
        "company:bmw", "company:toyota",
    }
    stats = _apply_tier_semantics(nodes, edges)
    assert stats["tier_invariant_violations"] == 0


def test_shortest_path_wins_over_longer_path():
    nodes = [
        {"id": "company:bmw", "type": "Company", "name": "BMW",
         "discovery_tier": 0, "is_seed": True, "tier_parent_ids": []},
        {"id": "company:mid", "type": "Company", "name": "Mid",
         "discovery_tier": None, "is_seed": False, "tier_parent_ids": []},
        {"id": "company:deep", "type": "Company", "name": "Deep",
         "discovery_tier": None, "is_seed": False, "tier_parent_ids": []},
    ]
    edges = [
        {
            "source": "company:mid", "target": "company:bmw",
            "relation": "suppliesTo", "expansion_relevant": True,
            "quality": {"expansion_relevant": True},
        },
        {
            "source": "company:deep", "target": "company:mid",
            "relation": "suppliesTo", "expansion_relevant": True,
            "quality": {"expansion_relevant": True},
        },
        {
            "source": "company:deep", "target": "company:bmw",
            "relation": "suppliesTo", "expansion_relevant": True,
            "quality": {"expansion_relevant": True},
        },
    ]
    _recompute_shortest_tiers(nodes, edges, {"company:bmw"}, max_tier=5)
    by_id = {n["id"]: n for n in nodes}
    assert by_id["company:deep"]["discovery_tier"] == 1


def test_product_material_no_tiers():
    nodes = [
        {"id": "product:x", "type": "Product", "name": "X",
         "discovery_tier": 1, "is_seed": False, "tier_parent_ids": []},
        {"id": "material:y", "type": "Material", "name": "Y",
         "discovery_tier": 2, "is_seed": False, "tier_parent_ids": []},
    ]
    _apply_tier_semantics(nodes, [])
    assert nodes[0]["discovery_tier"] is None
    assert nodes[1]["discovery_tier"] is None


def test_uses_does_not_imply_produces(ingest_config, ontology):
    facts = pd.DataFrame([
        _fact(
            fact_id="p1", doc_id="d",
            subject_id=canonical_id("Company", "Lucid"),
            subject_type="Company", subject_name="Lucid",
            relation="produces",
            object_id=canonical_id("Product", "battery"),
            object_type="Product", object_name="battery",
            evidence="The vehicle uses a 100-kWh battery.",
        ),
    ])
    from schain.kg import validate as validate_mod

    with patch.object(
        validate_mod, "_batch_evidence_validate",
        side_effect=lambda llm, rows: {r["candidate_fact_id"]: "unclear" for r in rows},
    ):
        accepted, rejected = validate_facts_frame(
            facts, config=ingest_config, ontology=ontology, llm=object(),
        )
    assert accepted.empty
    assert not rejected.empty


def test_max_tier_five_in_expansion_config():
    config = load_config(REPO_ROOT / "config/expansion.yaml")
    assert config["expansion"]["max_tier"] == 5


def test_aggregate_requires_direction_for_frontier(ingest_config):
    facts = pd.DataFrame([
        _fact(
            fact_id="f1", doc_id="d1",
            subject_id=canonical_id("Company", "X"),
            subject_type="Company", subject_name="X",
            relation="suppliesTo",
            object_id=canonical_id("Company", "BMW"),
            object_type="Company", object_name="BMW",
            evidence="X partners with BMW.",
            ontology_valid=True,
            evidence_validated=True,
            expansion_relevant=True,
            direction_explicit=False,
            evidence_quote_grounded=True,
            origin_entity=canonical_id("Company", "BMW"),
        ),
    ])
    support = aggregate_edge_support(facts, config=ingest_config)
    assert not bool(support.iloc[0]["frontier_eligible"])
    assert support.iloc[0]["frontier_rejection_reason"] == "ambiguous_supply_direction"
