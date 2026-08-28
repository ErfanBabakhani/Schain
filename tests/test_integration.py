"""Integration tests against the real external services.

Each test skips cleanly when its service is not configured, so the suite passes on a
laptop with no Neo4j and no model server. They are the checks to run on the server once
NEO4J_PASSWORD and the vLLM endpoint are in place.
"""

from __future__ import annotations

import os

import pytest

from schain.config import REPO_ROOT, load_config, neo4j_is_configured
from schain.documents import make_document
from schain.llm.provider import build_llm, llm_is_configured
from schain.ontology import load_ontology

pytestmark = pytest.mark.integration

SAMPLE = (
    "CATL supplies battery cells to BMW AG for its electric vehicle plants in Germany.\n"
    "Ganfeng Lithium Co Ltd supplies lithium to CATL under a long-term agreement.\n"
)


def live_config():
    return load_config(REPO_ROOT / "config/ingest.yaml")


# --- Neo4j -------------------------------------------------------------------------


@pytest.fixture
def driver():
    if not neo4j_is_configured():
        pytest.skip("NEO4J_PASSWORD is not set; skipping Neo4j integration test")
    from schain.pipeline import open_driver

    try:
        driver = open_driver(live_config())
    except Exception as error:  # noqa: BLE001 - unreachable database is a skip, not a failure
        pytest.skip(f"Neo4j is not reachable: {error}")
    yield driver
    driver.close()


def test_merge_is_idempotent_against_a_live_database(driver):
    """Writing the same facts twice must not create a second node or a second edge."""
    from schain.kg.writer import merge_nodes, merge_relationships

    database = live_config().get("neo4j", {}).get("database", "neo4j")
    suffix = "itest"
    nodes = [
        {"canonical_id": f"company:{suffix}-catl", "label": "Company", "name": "CATL itest",
         "aliases": ["CATL"], "doc_id": "itest-doc"},
        {"canonical_id": f"company:{suffix}-bmw", "label": "Company", "name": "BMW itest",
         "aliases": ["BMW AG"], "doc_id": "itest-doc"},
    ]
    relationships = [{
        "type": "suppliesTo", "start_id": f"company:{suffix}-catl",
        "end_id": f"company:{suffix}-bmw", "evidence": "CATL supplies BMW.",
        "doc_id": "itest-doc",
    }]

    try:
        for _ in range(2):
            merge_nodes(driver, database, nodes, load_ontology()["entity_types"])
            merge_relationships(driver, database, relationships,
                                load_ontology()["relation_types"])

        counted, _, _ = driver.execute_query(
            "MATCH (n:Entity) WHERE n.canonical_id STARTS WITH $prefix RETURN count(n) AS n",
            prefix=f"company:{suffix}-", database_=database,
        )
        assert counted[0]["n"] == 2

        edges, _, _ = driver.execute_query(
            "MATCH (:Entity {canonical_id: $start})-[r:suppliesTo]->(:Entity {canonical_id: $end}) "
            "RETURN count(r) AS n, collect(r.source_doc_ids)[0] AS docs",
            start=f"company:{suffix}-catl", end=f"company:{suffix}-bmw", database_=database,
        )
        assert edges[0]["n"] == 1
        assert edges[0]["docs"] == ["itest-doc"]  # provenance recorded once, not twice
    finally:
        driver.execute_query(
            "MATCH (n:Entity) WHERE n.canonical_id STARTS WITH $prefix DETACH DELETE n",
            prefix=f"company:{suffix}-", database_=database,
        )


# --- LLM endpoint ------------------------------------------------------------------


@pytest.fixture
def llm():
    config = live_config()
    if not llm_is_configured(config) or not os.environ.get("SCHAIN_RUN_LLM_TESTS"):
        pytest.skip(
            "set SCHAIN_RUN_LLM_TESTS=1 with a reachable llm.base_url to run LLM tests"
        )
    return build_llm(config)


def test_prompt1_extracts_a_supply_relation_from_a_live_model(llm):
    from schain.kg.extract import extract_graph

    graph, chunk_index = extract_graph(
        llm, make_document(SAMPLE), load_ontology(), on_error="RAISE"
    )
    assert graph.nodes, "the model returned no entities"
    labels = {node.label for node in graph.nodes}
    assert labels <= set(load_ontology()["entity_types"])
    assert any(rel.type == "suppliesTo" for rel in graph.relationships)
    assert chunk_index


def test_prompt2_groups_an_abbreviation_with_its_legal_name(llm):
    from neo4j_graphrag.components.types import Neo4jGraph, Neo4jNode

    from schain.kg.resolve import resolve_entities

    graph = Neo4jGraph(nodes=[
        Neo4jNode(id="c:0", label="Company", properties={"name": "CATL"}),
        Neo4jNode(id="c:1", label="Company",
                  properties={"name": "Contemporary Amperex Technology Co. Limited"}),
        Neo4jNode(id="c:2", label="Company", properties={"name": "BMW"}),
    ])
    resolution, _ = resolve_entities(graph, llm=llm)
    ids = {entry["mention"]: entry["canonical_id"] for entry in resolution.values()}
    assert ids["CATL"] == ids["Contemporary Amperex Technology Co. Limited"]
    assert ids["BMW"] != ids["CATL"]


# --- retrieval providers -----------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("SCHAIN_RUN_NETWORK_TESTS"),
    reason="set SCHAIN_RUN_NETWORK_TESTS=1 to query the public retrieval APIs",
)
def test_live_providers_return_candidates():
    import asyncio

    from schain.sources import gdelt, wikipedia
    from schain.sources.base import build_client

    async def go():
        async with build_client(20) as client:
            wiki = await wikipedia.search(client, '"BMW" suppliers', 3, {"language": "en"})
            news = await gdelt.search(client, '"BMW" suppliers', 3, {})
            return wiki, news

    wiki, news = asyncio.run(go())
    assert wiki, "Wikipedia returned no results"
    assert all(result.url.startswith("https://") for result in wiki + news)
