"""Retrieval recognizes aliases that entity disambiguation already persisted."""

from __future__ import annotations

import json

import pytest
from neo4j_graphrag.components.types import Neo4jGraph, Neo4jNode
from neo4j_graphrag.llm import LLMResponse

from schain import state
from schain.kg.relation_evidence import UPSTREAM_EXPLICIT
from schain.kg.resolve import resolve_entities
from schain.names import canonical_id
from schain.sources import retrieve
from schain.sources.base import SearchResult

ALPHA = "Alpha Energy Co., Ltd."
ALPHA_KEY = canonical_id("Company", ALPHA)
BETA = "Beta Energy Ltd"
BETA_KEY = canonical_id("Company", BETA)


@pytest.fixture(autouse=True)
def no_live_canonical_resolution(monkeypatch):
    async def unresolved(*args, **kwargs):
        return None
    monkeypatch.setattr(retrieve, "resolve_canonical_url", unresolved)


@pytest.fixture
def config(expand_config):
    """One provider and one query template, so assertions are about aliases only."""
    expand_config["expansion"]["sources"] = {"wikipedia": {"enabled": True}}
    expand_config["expansion"]["search"] = {
        "defaults": {"max_results": 5, "providers": ["wikipedia"]},
        "tiers": {0: {"queries": [{"query": '"{entity}" suppliers'}]}},
    }
    expand_config["expansion"]["relevance"] = {
        "min_chars": 20, "require_entity_mention": True, "keywords": ["supplier"],
    }
    expand_config["expansion"]["retrieval"]["alias_query_max"] = 2
    expand_config["expansion"]["retrieval"]["max_initial_query_variants"] = 30
    return expand_config


@pytest.fixture
def queries(monkeypatch):
    """Every provider query; each returns one AE article and one BE article."""
    issued: list[str] = []

    async def search(client, query, max_results, options):
        issued.append(query)
        return [
            SearchResult(provider="wikipedia", query=query, title="Gamma Mining to supply AE",
                         url="https://news.test/ae",
                         text="Gamma Mining supplies lithium hydroxide to AE under a "
                              "multi-year supply agreement."),
            SearchResult(provider="wikipedia", query=query, title="Gamma Mining to supply BE",
                         url="https://news.test/be",
                         text="Gamma Mining supplies lithium hydroxide to BE under a "
                              "multi-year supply agreement."),
        ]

    monkeypatch.setitem(retrieve.PROVIDERS, "wikipedia", search)
    return issued


class GroupingLLM:
    def __init__(self, groups):
        self.groups = groups

    def invoke(self, prompt):
        return LLMResponse(content=json.dumps({"groups": self.groups}))


def persist_disambiguation(connection, groups):
    """What the pipeline does after Prompt 2: resolve, then save the new cache entries."""
    names = [name for group in groups for name in group]
    graph = Neo4jGraph(nodes=[
        Neo4jNode(id=f"n{index}", label="Company", properties={"name": name})
        for index, name in enumerate(names)
    ])
    _, new_entries = resolve_entities(graph, llm=GroupingLLM(groups))
    state.save_resolution(connection, new_entries)


def test_previously_resolved_alias_is_recognized_during_retrieval(config, connection, queries):
    persist_disambiguation(connection, [["AE", ALPHA], ["BE", BETA]])

    results = retrieve.retrieve_for_entity(config, ALPHA, ALPHA_KEY, 0, connection)

    assert [result.url for result in results] == ["https://news.test/ae"]
    assert results[0].discovery_direction == UPSTREAM_EXPLICIT
    # the alias is searched for through the existing, bounded alias-query templates
    assert queries == [
        f'"{ALPHA}" suppliers', '"supplier to" "AE"', '"AE" procurement suppliers',
    ]


def test_unrelated_company_alias_is_not_accepted(config, connection, queries):
    persist_disambiguation(connection, [["AE", ALPHA], ["BE", BETA]])

    results = retrieve.retrieve_for_entity(config, BETA, BETA_KEY, 0, connection)

    assert [result.url for result in results] == ["https://news.test/be"]
    assert not any("AE" in query for query in queries)


def test_only_safe_aliases_of_the_same_entity_and_type_are_used(connection):
    state.save_resolution(connection, [
        {"label": "Company", "mention": "AE", "canonical_id": ALPHA_KEY,
         "canonical_name": ALPHA, "method": "llm"},
        {"label": "Company", "mention": "BE", "canonical_id": BETA_KEY,
         "canonical_name": BETA, "method": "llm"},
        # same canonical id, different entity type
        {"label": "Product", "mention": "AEX", "canonical_id": ALPHA_KEY,
         "canonical_name": ALPHA, "method": "llm"},
        # a bare prefix is not an identity the resolver itself would trust from cache
        {"label": "Company", "mention": "Alpha", "canonical_id": ALPHA_KEY,
         "canonical_name": ALPHA, "method": "llm"},
    ])

    assert retrieve._resolved_aliases(connection, ALPHA, ALPHA_KEY, {}) == ["AE"]
    assert retrieve._resolved_aliases(connection, BETA, BETA_KEY, {}) == ["BE"]
    # a configured ontology alias for another company outranks the cache
    assert retrieve._resolved_aliases(connection, ALPHA, ALPHA_KEY, {"AE": "Acme Electric"}) == []


def test_configured_aliases_keep_the_alias_query_slots(config, connection, queries, monkeypatch):
    persist_disambiguation(connection, [["AE", ALPHA]])
    monkeypatch.setattr(retrieve, "_company_aliases", lambda config: {"AEC": ALPHA})

    results = retrieve.retrieve_for_entity(config, ALPHA, ALPHA_KEY, 0, connection)

    # alias_query_max=2 is used up by the configured alias; the cached one adds no query
    assert queries == [
        f'"{ALPHA}" suppliers', '"supplier to" "AEC"', '"AEC" procurement suppliers',
    ]
    # but it still widens relevance matching
    assert [result.url for result in results] == ["https://news.test/ae"]


def test_empty_cache_leaves_retrieval_unchanged(config, connection, queries):
    assert retrieve._resolved_aliases(connection, ALPHA, ALPHA_KEY, {}) == []

    assert retrieve.retrieve_for_entity(config, ALPHA, ALPHA_KEY, 0, connection) == []
    assert queries == [f'"{ALPHA}" suppliers']

    # configured ontology aliases still drive queries exactly as before
    queries.clear()
    retrieve.retrieve_for_entity(config, "BMW", "company:bmw", 0, connection)
    assert queries == [
        '"BMW" suppliers', '"supplier to" "Bayerische Motoren Werke AG"',
        '"Bayerische Motoren Werke AG" procurement suppliers',
    ]
