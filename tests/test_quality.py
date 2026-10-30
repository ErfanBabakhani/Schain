"""Quality path after removing multi-source / independent-source verification."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from neo4j_graphrag.components.types import Neo4jGraph, Neo4jNode

from schain.config import storage_path
from schain.documents import make_document
from schain.graph_export import build_graph_bundle
from schain.kg.facts import FACT_COLUMNS
from schain.kg.resolve import resolve_entities
from schain.kg.support import (
    aggregate_edge_support,
    assign_quality_status,
    frontier_eligible_suppliers,
)
from schain.kg.validate import (
    is_generic_company_phrase,
    is_ontology_valid,
    validate_facts_frame,
)
from schain.names import canonical_id
from schain.ontology import load_ontology
from schain.sources.base import TITLE_SNIPPET, SearchResult
from schain.sources.ranking import rank_discovery_candidate
from schain.sources.reputation import clamp_score, load_source_reputation, match_domain, score_source
from schain.sources.retrieve import document_kwargs_from_result
from schain.verification import verify_supplies_to_edges


@pytest.fixture
def reputation(ingest_config):
    return load_source_reputation(ingest_config)


@pytest.fixture
def ontology(ingest_config):
    return load_ontology(ingest_config)


def _fact(**kwargs):
    row = {column: None for column in FACT_COLUMNS}
    row.update(kwargs)
    return row


def _accepted_supply(subject: str, obj: str, doc_id: str, **extra):
    return _fact(
        fact_id=f"f-{doc_id}-{subject}",
        doc_id=doc_id,
        subject_id=canonical_id("Company", subject),
        subject_type="Company",
        subject_name=subject,
        subject_mention=subject,
        relation="suppliesTo",
        object_id=canonical_id("Company", obj),
        object_type="Company",
        object_name=obj,
        object_mention=obj,
        evidence=f"{subject} supplies to {obj}.",
        ontology_valid=True,
        evidence_validated=True,
        evidence_quote_grounded=True,
        validation_status="accepted",
        source_effective_score=extra.pop("source_effective_score", 0),
        source_class=extra.pop("source_class", "retrieval_wrapper"),
        content_source=extra.pop("content_source", "title_snippet"),
        provider=extra.pop("provider", "google_news"),
        origin_entity=extra.pop("origin_entity", canonical_id("Company", obj)),
        expansion_relevant=extra.pop("expansion_relevant", True),
        direction_explicit=extra.pop("direction_explicit", True),
        **extra,
    )


# --- source reputation (informational) --------------------------------------------


def test_known_domain_score(reputation):
    scored = score_source(
        url="https://www.reuters.com/article/1",
        content_source="full_text",
        reputation=reputation,
    )
    assert scored.known
    assert scored.effective_score >= 80


def test_unknown_domain_gets_default_score(reputation):
    scored = score_source(url="https://random-blog.example/xyz", reputation=reputation)
    assert not scored.known
    assert scored.base_score == 50


def test_score_clamped_0_100():
    assert clamp_score(-5) == 0
    assert clamp_score(150) == 100


def test_domain_matching_does_not_use_substring(reputation):
    matched = match_domain("notreuters.com", reputation)
    assert matched is None or matched[0] != "reuters.com"


# --- one-source edge + frontier ---------------------------------------------------


def test_one_explicit_valid_source_can_create_edge(ingest_config):
    facts = pd.DataFrame([
        _accepted_supply("Qualcomm", "BMW", "d1", source_effective_score=0),
    ])
    support = aggregate_edge_support(facts, config=ingest_config)
    assert len(support) == 1
    assert support.iloc[0]["quality_status"] in ("accepted", "accepted_high_reputation")
    assert int(support.iloc[0]["document_count"]) == 1


def test_one_explicit_title_snippet_can_advance_frontier(ingest_config):
    facts = pd.DataFrame([
        _accepted_supply(
            "Qualcomm", "BMW", "d1",
            content_source="title_snippet",
            source_class="retrieval_wrapper",
            source_effective_score=0,
        ),
    ])
    support = aggregate_edge_support(facts, config=ingest_config)
    assert bool(support.iloc[0]["frontier_eligible"]) is True
    names = frontier_eligible_suppliers(
        support, {canonical_id("Company", "BMW")}, ["suppliesTo"],
    )
    assert "Qualcomm" in names


def test_google_news_wrapper_does_not_block_valid_edge(ingest_config):
    facts = pd.DataFrame([
        _accepted_supply(
            "LG Energy Solution", "BMW", "gn1",
            source_class="retrieval_wrapper",
            source_domain="news.google.com",
            source_effective_score=0,
            provider="google_news",
        ),
    ])
    support = aggregate_edge_support(facts, config=ingest_config)
    assert bool(support.iloc[0]["frontier_eligible"]) is True


def test_source_score_does_not_gate_frontier(ingest_config):
    status, eligible, reason, rejection = assign_quality_status(
        ontology_valid=True,
        evidence_validated=True,
        relation="suppliesTo",
        subject_type="Company",
        max_source_score=0.0,
        expansion_relevant=True,
        direction_explicit=True,
    )
    assert status == "accepted"
    assert eligible is True
    assert reason["direction_explicit"] is True
    assert rejection is None


def test_qualcomm_single_valid_source_becomes_tier1(ingest_config):
    facts = pd.DataFrame([_accepted_supply("Qualcomm", "BMW", "q1")])
    support = aggregate_edge_support(facts, config=ingest_config)
    assert bool(support.iloc[0]["frontier_eligible"])
    assert "Qualcomm" in frontier_eligible_suppliers(
        support, {canonical_id("Company", "BMW")}, ["suppliesTo"],
    )


def test_support_count_equals_unique_accepted_document_count(ingest_config):
    facts = pd.DataFrame([
        _accepted_supply("Qualcomm", "BMW", "doc-a"),
        _accepted_supply("Qualcomm", "BMW", "doc-b"),
        # Legacy verification pseudo-doc must not inflate support.
        _accepted_supply("Qualcomm", "BMW", "verify:ignored"),
    ])
    # Distinct evidence strings so fact_ids differ; document_count is what matters.
    facts.loc[0, "evidence"] = "Qualcomm supplies chips to BMW (doc a)."
    facts.loc[1, "evidence"] = "Qualcomm supplies chips to BMW (doc b)."
    facts.loc[2, "evidence"] = "verification observation"
    support = aggregate_edge_support(facts, config=ingest_config)
    assert int(support.iloc[0]["document_count"]) == 2
    assert int(support.iloc[0]["fact_count"]) == 2


# --- validation stays strict ------------------------------------------------------


def test_invalid_ontology_still_rejected(ingest_config, ontology):
    facts = pd.DataFrame([
        _fact(
            fact_id="bad1", doc_id="d",
            subject_id=canonical_id("Company", "CATL"),
            subject_type="Company", subject_name="CATL",
            relation="suppliesTo",
            object_id=canonical_id("Product", "Battery"),
            object_type="Product", object_name="Battery",
            evidence="CATL: Battery",
        ),
    ])
    accepted, rejected = validate_facts_frame(
        facts, config=ingest_config, ontology=ontology, llm=None,
    )
    assert accepted.empty
    assert not rejected.empty
    assert rejected.iloc[0]["rejection_stage"] == "ontology"


def test_invalid_evidence_still_rejected(ingest_config, ontology):
    facts = pd.DataFrame([
        _fact(
            fact_id="own1", doc_id="d",
            subject_id=canonical_id("Person", "Thomas Thym"),
            subject_type="Person", subject_name="Thomas Thym",
            relation="owns",
            object_id=canonical_id("Company", "BMW"),
            object_type="Company", object_name="BMW",
            evidence="Thomas Thym is CEO of BMW Group.",
        ),
    ])

    class UnclearLLM:
        def invoke(self, prompt):
            class R:
                content = '{"results":[{"candidate_fact_id":"x","decision":"unclear"}]}'

            return R()

    from schain.kg import validate as validate_mod

    def always_unclear(llm, rows):
        return {row["candidate_fact_id"]: "unclear" for row in rows}

    with patch.object(validate_mod, "_batch_evidence_validate", side_effect=always_unclear):
        accepted, rejected = validate_facts_frame(
            facts, config=ingest_config, ontology=ontology, llm=UnclearLLM(),
        )
    assert accepted.empty
    assert rejected.iloc[0]["rejection_stage"] == "evidence"


def test_generic_chip_suppliers_not_company_entity(ingest_config, ontology):
    assert is_generic_company_phrase("chip suppliers")
    assert is_generic_company_phrase("battery suppliers")
    assert not is_generic_company_phrase("Qualcomm")

    facts = pd.DataFrame([
        _fact(
            fact_id="g1", doc_id="d",
            subject_id=canonical_id("Company", "chip suppliers"),
            subject_type="Company", subject_name="chip suppliers",
            relation="suppliesTo",
            object_id=canonical_id("Company", "BMW"),
            object_type="Company", object_name="BMW",
            evidence="BMW works with chip suppliers.",
        ),
    ])
    accepted, rejected = validate_facts_frame(
        facts, config=ingest_config, ontology=ontology, llm=None,
    )
    assert accepted.empty
    assert rejected.iloc[0]["rejection_stage"] == "generic_entity"


# --- provenance / aliases / seed / export -----------------------------------------


def test_provider_metadata_preserved(ingest_config, tmp_path):
    result = SearchResult(
        provider="google_news",
        query='"BMW" suppliers',
        title="BMW names Qualcomm lead chip supplier",
        url="https://news.google.com/rss/articles/XYZ",
        snippet="Qualcomm will supply chips to BMW.",
        text=None,  # title/snippet fallback
        entity_key=canonical_id("Company", "BMW"),
        tier=0,
    )
    result._source_class = "retrieval_wrapper"  # type: ignore[attr-defined]
    kwargs = document_kwargs_from_result(result)
    assert kwargs["provider"] == "google_news"
    assert kwargs["content_source"] == TITLE_SNIPPET
    doc = make_document(
        "BMW names Qualcomm lead chip supplier for next-gen platforms. "
        "Qualcomm will supply chips to BMW Group.",
        **kwargs,
    )
    assert doc["provider"] == "google_news"
    assert doc["query"] == '"BMW" suppliers'
    assert doc["origin_entity"] == canonical_id("Company", "BMW")


def test_bmw_group_resolves_to_bmw(ontology):
    aliases = ontology.get("aliases") or {}
    company_aliases = aliases.get("Company") or {}
    assert company_aliases.get("BMW Group") == "BMW"
    graph = Neo4jGraph(
        nodes=[
            Neo4jNode(id="0", label="Company", properties={"name": "BMW Group"}),
            Neo4jNode(id="1", label="Company", properties={"name": "BMW"}),
        ],
        relationships=[],
    )
    resolution, _ = resolve_entities(graph, llm=None, aliases=aliases)
    assert resolution["0"]["canonical_id"] == resolution["1"]["canonical_id"]
    assert resolution["0"]["canonical_name"] == "BMW"


def test_seed_always_exported(ingest_config):
    config = dict(ingest_config)
    config["seeds"] = {"ev_manufacturers": ["BMW"]}
    # Empty facts still export the seed.
    path = storage_path(config, "extracted_facts")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=FACT_COLUMNS).to_parquet(path, index=False)
    bundle = build_graph_bundle(config, mode="export")
    seeds = [n for n in bundle["nodes"] if n.get("is_seed")]
    assert any(n["name"] == "BMW" and n["discovery_tier"] == 0 for n in seeds)


def test_duplicate_document_still_deduplicated(tmp_path):
    text = "CATL supplies battery cells to BMW under a long-term agreement."
    a = make_document(text, source_url="https://example.com/a", content_source="full_text")
    b = make_document(text, source_url="https://example.com/b", content_source="full_text")
    assert a["doc_id"] == b["doc_id"]
    assert a["content_hash"] == b["content_hash"]


def test_verification_search_is_noop(ingest_config):
    stats = verify_supplies_to_edges(ingest_config, connection=None, llm=None)
    assert stats["enabled"] is False
    assert stats["queries_executed"] == 0


def test_export_quality_object_simplified(ingest_config):
    config = dict(ingest_config)
    config["seeds"] = {"ev_manufacturers": ["BMW"]}
    facts = pd.DataFrame([
        _accepted_supply(
            "Qualcomm", "BMW", "exp1",
            provider="google_news",
            source_class="retrieval_wrapper",
            source_effective_score=0,
        ),
    ])
    facts_path = storage_path(config, "extracted_facts")
    facts_path.parent.mkdir(parents=True, exist_ok=True)
    facts.to_parquet(facts_path, index=False)

    docs = pd.DataFrame([{
        "doc_id": "exp1",
        "title": "BMW names Qualcomm lead chip supplier",
        "source_url": "https://news.google.com/rss/articles/XYZ",
        "content_source": "title_snippet",
        "provider": "google_news",
        "source_class": "retrieval_wrapper",
        "source_effective_score": 0,
        "char_count": 80,
        "content_hash": "abc",
        "ingested_at": "2026-01-01T00:00:00+00:00",
    }])
    docs_path = storage_path(config, "normalized_documents")
    docs.to_parquet(docs_path, index=False)

    support = aggregate_edge_support(facts, config=config)
    support_path = storage_path(config, "edge_support")
    support.to_parquet(support_path, index=False)

    bundle = build_graph_bundle(config, mode="export")
    edge = next(e for e in bundle["edges"] if e["relation"] == "suppliesTo")
    assert edge["support_count"] == 1
    assert edge["independent_support_count"] is None
    assert edge["quality"]["status"] in ("accepted", "accepted_high_reputation")
    assert edge["quality"]["verification_status"] == "not_required"
    assert edge["quality"]["frontier_eligible"] is True
    assert edge["evidence"][0]["provider"] == "google_news"
    assert "verification_status" not in bundle["statistics"] or "quality_status" in bundle["statistics"]
    assert bundle["statistics"].get("quality_status")


def test_wrapper_rank_penalty_does_not_discard_explicit(reputation):
    gn = SearchResult(
        provider="google_news",
        query='"BMW" suppliers',
        title="BMW names Qualcomm lead chip supplier",
        url="https://news.google.com/x",
        snippet="Qualcomm selected as lead chip supplier for BMW.",
        text=None,
    )
    gn._source_class = "retrieval_wrapper"  # type: ignore[attr-defined]
    gn._source_effective_score = 0  # type: ignore[attr-defined]
    score = rank_discovery_candidate(gn, "BMW")
    assert score > 40  # still strongly preferred vs portal pages


def test_is_ontology_valid_company_supplies(ontology):
    assert is_ontology_valid("Company", "suppliesTo", "Company", ontology)
    assert not is_ontology_valid("Company", "suppliesTo", "Product", ontology)
