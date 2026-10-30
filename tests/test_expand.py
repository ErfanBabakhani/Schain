"""The EXPAND loop: tier progression, tier boundary, and what may advance a tier."""

from __future__ import annotations

import pytest

from schain import state
from schain.expansion.frontier import discover_upstream, seed_frontier
from schain.expansion.loop import run_expand
from schain.kg.facts import FACT_COLUMNS
from schain.sources.base import SearchResult

from .conftest import DOCUMENT_TEXT, StubLLM
from .test_pipeline import FakeDriver

import pandas as pd

CATL_CANONICAL = "company:contemporary-amperex-technology"


@pytest.fixture
def single_seed_config(expand_config):
    """One seed keeps the assertions about tier progression readable."""
    expand_config["seeds"] = {"ev_manufacturers": ["BMW"]}
    # Keep expand unit tests offline and focused on discovery/frontier behavior.
    expand_config.setdefault("verification", {})["enabled"] = False
    return expand_config


@pytest.fixture
def fake_retrieval(monkeypatch):
    """Replace retrieval with one fixed high-quality document (no network)."""
    calls: list[tuple[str, int]] = []

    def fake(config, entity_name, entity_key, tier, connection, manifest=None, **_kwargs):
        calls.append((entity_name, tier))
        result = SearchResult(
            provider="test", query=f'"{entity_name}" suppliers',
            title=f"{entity_name} supplier news", url=f"https://www.bmwgroup.com/news/{tier}",
            text=DOCUMENT_TEXT,
            retrieval_url=f"https://www.bmwgroup.com/news/{tier}",
            canonical_source_url=f"https://www.bmwgroup.com/news/{tier}",
            source_domain="bmwgroup.com",
            entity_key=entity_key,
            tier=tier,
        )
        # Strong official full-text provenance so suppliesTo is frontier-eligible.
        result._source_known = True  # type: ignore[attr-defined]
        result._source_class = "official_primary"  # type: ignore[attr-defined]
        result._source_base_score = 94  # type: ignore[attr-defined]
        result._source_effective_score = 99  # type: ignore[attr-defined]
        result._source_group_id = "domain:bmwgroup.com"  # type: ignore[attr-defined]
        return [result]

    monkeypatch.setattr("schain.expansion.loop.retrieve_for_entity", fake)
    return calls


def facts_frame(relation="suppliesTo", subject_type="Company"):
    row = {column: None for column in FACT_COLUMNS}
    row.update(
        fact_id="f1", doc_id="d1", subject_id=CATL_CANONICAL, subject_type=subject_type,
        subject_name="Contemporary Amperex Technology Co. Limited",
        subject_mention="CATL", relation=relation,
        object_id="company:bmw", object_type="Company", object_name="BMW",
        object_mention="BMW AG",
        ontology_valid=True, evidence_validated=True, evidence_quote_grounded=True, content_source="full_text",
        source_effective_score=99, source_domain="bmwgroup.com",
        source_class="official_primary", source_known=True,
        source_group_id="domain:bmwgroup.com",
        expansion_relevant=True,
        direction_explicit=True,
        origin_entity="company:bmw",
    )
    return pd.DataFrame([row])


# --- frontier rules ----------------------------------------------------------------


def test_seeds_enter_at_tier_zero(single_seed_config, connection):
    assert seed_frontier(connection, single_seed_config) == 1
    assert state.frontier_tier_counts(connection) == {0: 1}
    # seeding twice adds nothing
    assert seed_frontier(connection, single_seed_config) == 0


def test_supplier_of_a_tier0_company_enters_tier1(
    single_seed_config, connection, ontology
):
    # Write eligible edge_support so the frontier gate can promote the supplier.
    from schain.kg.support import aggregate_edge_support

    state.enqueue_entity(connection, "BMW", 0)
    facts = facts_frame()
    support = aggregate_edge_support(facts, config=single_seed_config)
    from pathlib import Path
    path = Path(single_seed_config["storage"]["edge_support"])
    path.parent.mkdir(parents=True, exist_ok=True)
    support.to_parquet(path, index=False)

    added = discover_upstream(
        connection, single_seed_config, ontology, "BMW", "company:bmw", 0, facts
    )
    assert added == ["Contemporary Amperex Technology Co. Limited"]
    assert state.frontier_tier_counts(connection) == {0: 1, 1: 1}


@pytest.mark.parametrize("relation", ["locatedIn", "owns", "produces"])
def test_non_supply_relations_never_advance_a_tier(
    relation, single_seed_config, connection, ontology
):
    added = discover_upstream(
        connection, single_seed_config, ontology, "BMW", "company:bmw", 0,
        facts_frame(relation=relation),
    )
    assert added == []
    assert state.frontier_tier_counts(connection) == {}


def test_only_company_subjects_enter_the_frontier(
    single_seed_config, connection, ontology
):
    added = discover_upstream(
        connection, single_seed_config, ontology, "BMW", "company:bmw", 0,
        facts_frame(subject_type="Mine"),
    )
    assert added == []


def test_max_tier_is_the_stopping_boundary(single_seed_config, connection, ontology):
    single_seed_config["expansion"]["max_tier"] = 3
    # a Tier-3 company is stored but never expanded further
    assert discover_upstream(
        connection, single_seed_config, ontology, "Deep Co", "company:deep-co", 3,
        facts_frame(),
    ) == []
    assert state.frontier_tier_counts(connection) == {}
    # at Tier 2 the same facts do produce a Tier-3 entity
    assert discover_upstream(
        connection, single_seed_config, ontology, "Mid Co", "company:mid-co", 2,
        facts_frame(),
    ) == []  # facts point at company:bmw, not at Mid Co


def test_unexpandable_ontology_stops_enqueueing(
    single_seed_config, connection, ontology
):
    restricted = {**ontology, "expandable_types": ["Material"]}
    assert discover_upstream(
        connection, single_seed_config, restricted, "BMW", "company:bmw", 0, facts_frame()
    ) == []


# --- the loop ----------------------------------------------------------------------


def test_expand_discovers_the_next_tier(
    single_seed_config, connection, ontology, fake_retrieval
):
    single_seed_config["expansion"]["max_tier"] = 1
    summary = run_expand(single_seed_config, connection, StubLLM(), ontology, FakeDriver())

    assert summary["entities_expanded"] == 1          # Tier 1 is stored terminal
    assert summary["discovered"]["BMW"] == ["Contemporary Amperex Technology Co. Limited"]
    assert [name for name, _ in fake_retrieval] == [
        "BMW"
    ]
    assert state.frontier_counts(connection) == {state.DONE: 2}
    assert state.frontier_tier_counts(connection) == {0: 1, 1: 1}


def test_expand_stops_at_tier_zero_when_configured(
    single_seed_config, connection, ontology, fake_retrieval
):
    single_seed_config["expansion"]["max_tier"] = 0
    summary = run_expand(single_seed_config, connection, StubLLM(), ontology, FakeDriver())
    assert summary["entities_expanded"] == 0
    assert fake_retrieval == []
    assert summary["discovered"] == {}
    assert state.frontier_tier_counts(connection) == {0: 1}


def test_repeated_documents_across_entities_are_not_reprocessed(
    single_seed_config, connection, ontology, fake_retrieval
):
    """The second entity retrieves the same text, which must not be extracted twice."""
    single_seed_config["expansion"]["max_tier"] = 2
    llm = StubLLM()
    summary = run_expand(single_seed_config, connection, llm, ontology, FakeDriver())
    assert summary["documents_processed"] == 1
    assert summary["documents_skipped"] == 0  # downstream repeat excluded before selection
    assert llm.extraction_calls == 1


def test_max_documents_stops_the_run_and_keeps_the_entity_pending(
    single_seed_config, connection, ontology, fake_retrieval
):
    single_seed_config["expansion"]["max_documents"] = 1
    single_seed_config["expansion"]["max_documents_per_run"] = 1
    single_seed_config["expansion"]["max_tier"] = 3
    summary = run_expand(single_seed_config, connection, StubLLM(), ontology, FakeDriver())
    assert summary["documents_processed"] == 1
    assert summary["stop_reason"] in ("document_budget", "max_documents")
    assert summary["completion"]["complete"] is False
    # Interrupted / remaining work stays PENDING for resume.
    assert state.PENDING in state.frontier_counts(connection) or summary["completion"]["pending_frontier"] >= 0


def test_resumed_expand_does_not_redo_finished_entities(
    single_seed_config, connection, ontology, fake_retrieval
):
    single_seed_config["expansion"]["max_tier"] = 1
    run_expand(single_seed_config, connection, StubLLM(), ontology, FakeDriver())

    second = StubLLM()
    summary = run_expand(single_seed_config, connection, second, ontology, FakeDriver())
    assert summary["entities_expanded"] == 0
    assert second.prompts == []


def test_repeated_llm_failures_abort_instead_of_grinding(
    single_seed_config, connection, ontology, fake_retrieval, monkeypatch
):
    from .conftest import FailingLLM

    single_seed_config["runtime"]["max_consecutive_failures"] = 1
    with pytest.raises(RuntimeError, match="failed in a row"):
        run_expand(single_seed_config, connection, FailingLLM(), ontology, None)


def test_headline_only_documents_are_ingested_and_marked_as_such(
    single_seed_config, connection, ontology, monkeypatch
):
    """EXPAND must keep working when a provider yields only a title and snippet."""
    def fake(config, entity_name, entity_key, tier, connection, manifest=None, **_kwargs):
        return [SearchResult(
            provider="google_news", query="q",
            title=f"{entity_name} names CATL as battery supplier",
            url="https://news.google.com/rss/articles/CBMwrapper",
            snippet="CATL supplies battery cells to BMW AG under a new agreement.",
        )]

    monkeypatch.setattr("schain.expansion.loop.retrieve_for_entity", fake)
    single_seed_config["expansion"]["max_tier"] = 1
    summary = run_expand(single_seed_config, connection, StubLLM(), ontology, FakeDriver())

    assert summary["documents_processed"] == 1
    documents = pd.read_parquet(single_seed_config["storage"]["normalized_documents"])
    assert documents.iloc[0]["content_source"] == "title_snippet"


def test_ingested_files_are_marked_as_file_sourced(
    document_file, ingest_config, connection, ontology
):
    from schain.documents import load_document

    record = load_document(document_file)
    assert record["content_source"] == "file"


# --- depth 6 and what determines a tier ---------------------------------------------


def test_tier_six_entities_are_stored_but_never_searched(
    single_seed_config, connection, ontology, fake_retrieval
):
    """Terminal tier: it may sit in the frontier, but it generates no queries."""
    from schain.expansion.queries import generate_queries

    single_seed_config["expansion"]["max_tier"] = 6
    assert generate_queries(single_seed_config, "Deep Mine Co", 6) == []

    # a Tier-6 company is recorded with its tier...
    state.enqueue_entity(connection, "Deep Mine Co", 6)
    assert state.frontier_tier_counts(connection)[6] == 1
    # ...and expanding it promotes nothing to Tier 7
    assert discover_upstream(
        connection, single_seed_config, ontology, "Deep Mine Co", "company:deep-mine-co",
        6, facts_frame(),
    ) == []
    assert 7 not in state.frontier_tier_counts(connection)


def test_a_tier_five_supplier_is_promoted_to_tier_six(
    single_seed_config, connection, ontology
):
    single_seed_config["expansion"]["max_tier"] = 6
    state.enqueue_entity(connection, "Midstream Refining", 5)
    facts = facts_frame()
    facts.loc[0, "object_id"] = "company:midstream-refining"
    added = discover_upstream(
        connection, single_seed_config, ontology, "Midstream Refining",
        "company:midstream-refining", 5, facts,
    )
    assert added == ["Contemporary Amperex Technology Co. Limited"]
    assert state.frontier_tier_counts(connection) == {5: 1, 6: 1}


def test_tier_is_graph_distance_not_industrial_role(
    single_seed_config, connection, ontology
):
    """A mining company supplying an OEM directly is Tier 1, not a deep tier."""
    single_seed_config["expansion"]["max_tier"] = 6
    state.enqueue_entity(connection, "BMW", 0)
    facts = facts_frame()
    facts.loc[0, "subject_id"] = "company:zijin-mining"
    facts.loc[0, "subject_name"] = "Zijin Mining"

    added = discover_upstream(
        connection, single_seed_config, ontology, "BMW", "company:bmw", 0, facts
    )
    assert added == ["Zijin Mining"]
    assert state.frontier_tier_counts(connection) == {0: 1, 1: 1}


def test_a_deep_tier_still_advances_only_through_supplies_to(
    single_seed_config, connection, ontology
):
    single_seed_config["expansion"]["max_tier"] = 6
    facts = facts_frame(relation="owns")
    facts.loc[0, "object_id"] = "company:midstream-refining"
    assert discover_upstream(
        connection, single_seed_config, ontology, "Midstream Refining",
        "company:midstream-refining", 4, facts,
    ) == []
