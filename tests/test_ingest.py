"""Offline tests for the one-document INGEST path.

A stub LLM stands in for Prompt 1 and Prompt 2, so the whole chain -
normalize, extract, resolve, facts table, MERGE statements - is exercised without a
model server or a Neo4j instance.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from neo4j_graphrag.llm import LLMInterface, LLMResponse

from documents import content_hash, normalize_text, normalize_url
from kg.resolve import canonical_id, normalization_key
from kg.writer import merge_nodes, merge_relationships
from main import ingest_document
from ontology import ENTITY_TYPES, RELATION_TYPES

DOCUMENT_TEXT = (
    "CATL supplies battery cells to BMW AG.\n"
    "Contemporary Amperex Technology Co. Limited operates a plant located in Germany.\n"
    "BMW produces electric vehicles.\n"
)

EXTRACTION_RESPONSE = json.dumps(
    {
        "nodes": [
            {"id": "0", "label": "Company", "properties": {"name": "CATL"}},
            {"id": "1", "label": "Company", "properties": {"name": "BMW AG"}},
            {"id": "2", "label": "Company",
             "properties": {"name": "Contemporary Amperex Technology Co. Limited"}},
            {"id": "3", "label": "Location", "properties": {"name": "Germany"}},
            {"id": "4", "label": "Product", "properties": {"name": "electric vehicles"}},
            {"id": "5", "label": "Company", "properties": {"name": "BMW"}},
        ],
        "relationships": [
            {"type": "suppliesTo", "start_node_id": "0", "end_node_id": "1",
             "properties": {"evidence": "CATL supplies battery cells to BMW AG."}},
            {"type": "locatedIn", "start_node_id": "2", "end_node_id": "3",
             "properties": {"evidence": "Contemporary Amperex Technology Co. Limited "
                                        "operates a plant located in Germany."}},
            {"type": "produces", "start_node_id": "5", "end_node_id": "4",
             "properties": {"evidence": "BMW produces electric vehicles."}},
        ],
    }
)

# Prompt 2 merges the abbreviation with the full legal name; BMW stays on its own.
DISAMBIGUATION_RESPONSE = json.dumps(
    {"groups": [["CATL", "Contemporary Amperex Technology Co. Limited"], ["BMW AG"]]}
)


class StubLLM(LLMInterface):
    """Answers Prompt 1 and Prompt 2 with fixed JSON, and counts the calls."""

    def __init__(self) -> None:
        super().__init__(model_name="stub")
        self.prompts: list[str] = []

    def _answer(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "Group the names" in prompt:
            return DISAMBIGUATION_RESPONSE
        return EXTRACTION_RESPONSE

    def invoke(self, input, message_history=None, system_instruction=None):
        return LLMResponse(content=self._answer(input))

    async def ainvoke(self, input, message_history=None, system_instruction=None):
        return LLMResponse(content=self._answer(input))


@pytest.fixture
def config(tmp_path):
    return {
        "llm": {"provider": "local", "model": "stub"},
        "ingest": {"chunk_size": 4000, "chunk_overlap": 100, "on_error": "RAISE"},
        "storage": {
            "raw_documents": str(tmp_path / "raw/documents.jsonl"),
            "normalized_documents": str(tmp_path / "processed/documents.parquet"),
            "extracted_facts": str(tmp_path / "processed/facts.parquet"),
            "entity_resolution": str(tmp_path / "processed/entity_resolution.parquet"),
        },
        "neo4j": {"database": "neo4j"},
    }


@pytest.fixture
def document_file(tmp_path):
    path = tmp_path / "catl_bmw.txt"
    path.write_text(DOCUMENT_TEXT, encoding="utf-8")
    return path


def run_ingest(document_file, config, llm=None):
    return ingest_document(
        document_file, config, write_to_neo4j=False, llm=llm or StubLLM()
    )


# --- normalization and identifiers -------------------------------------------------


def test_text_hash_ignores_formatting_noise():
    assert content_hash(normalize_text("CATL supplies\r\nBMW.  \n\n")) == content_hash(
        normalize_text("﻿CATL supplies\nBMW.")
    )


def test_url_normalization_drops_tracking_and_case():
    assert normalize_url("HTTPS://EN.Wikipedia.org/wiki/CATL/?utm_source=x#a") == (
        "https://en.wikipedia.org/wiki/CATL"
    )


def test_legal_suffixes_do_not_create_separate_entities():
    assert normalization_key("BMW AG") == normalization_key("bmw ag.") == "bmw"
    assert canonical_id("Company", "BYD Co., Ltd.") == "company:byd"


# --- the INGEST path ---------------------------------------------------------------


def test_ingest_produces_expected_facts(document_file, config):
    summary = run_ingest(document_file, config)

    assert summary["relationships_extracted"] == 3
    assert summary["facts"] == 3

    facts = pd.read_parquet(config["storage"]["extracted_facts"])
    triples = set(zip(facts.subject_id, facts.relation, facts.object_id))
    assert triples == {
        ("company:contemporary-amperex-technology", "suppliesTo", "company:bmw"),
        ("company:contemporary-amperex-technology", "locatedIn", "location:germany"),
        ("company:bmw", "produces", "product:electric-vehicles"),
    }
    # every fact keeps the sentence it came from, and the chunk it came from
    assert facts.evidence.str.len().gt(0).all()
    assert (facts.chunk_index == 0).all()


def test_prompt2_merges_abbreviation_with_legal_name(document_file, config):
    run_ingest(document_file, config)
    facts = pd.read_parquet(config["storage"]["extracted_facts"])

    catl = facts[facts.relation == "suppliesTo"].iloc[0]
    plant = facts[facts.relation == "locatedIn"].iloc[0]
    # "CATL" and "Contemporary Amperex ..." were separate mentions of one entity
    assert catl.subject_mention == "CATL"
    assert plant.subject_mention == "Contemporary Amperex Technology Co. Limited"
    assert catl.subject_id == plant.subject_id


def test_deterministic_stage_runs_before_the_llm(document_file, config):
    """BMW AG and BMW collapse without being sent to Prompt 2."""
    llm = StubLLM()
    run_ingest(document_file, config, llm=llm)

    prompt2 = [p for p in llm.prompts if "Group the names" in p]
    company_prompt = next(p for p in prompt2 if "Company" in p)
    assert "- BMW AG" in company_prompt
    assert "\n- BMW\n" not in company_prompt  # the short variant never reaches the LLM


def test_repeated_ingest_is_idempotent(document_file, config):
    first = run_ingest(document_file, config)
    second = run_ingest(document_file, config)

    assert first["doc_id"] == second["doc_id"]
    assert second["facts_total"] == first["facts_total"] == 3

    raw_lines = [
        line
        for line in open(config["storage"]["raw_documents"], encoding="utf-8")
        if line.strip()
    ]
    assert len(raw_lines) == 1  # the same document is not stored twice
    assert len(pd.read_parquet(config["storage"]["normalized_documents"])) == 1


def test_entity_resolution_is_persisted(document_file, config):
    run_ingest(document_file, config)
    resolution = pd.read_parquet(config["storage"]["entity_resolution"])
    assert set(resolution.columns) >= {
        "doc_id", "node_id", "label", "canonical_name", "canonical_id", "aliases", "mention"
    }
    bmw = resolution[resolution.canonical_id == "company:bmw"]
    assert set(bmw.iloc[0]["aliases"]) == {"BMW", "BMW AG"}


# --- graph writes ------------------------------------------------------------------


class RecordingDriver:
    """Captures the Cypher the writer would send."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def execute_query(self, query, **kwargs):
        self.queries.append(query)
        return None


def test_graph_writes_use_merge_not_create():
    driver = RecordingDriver()
    merge_nodes(
        driver,
        "neo4j",
        [{"canonical_id": "company:bmw", "label": "Company", "name": "BMW",
          "aliases": ["BMW"], "doc_id": "d1"}],
        ENTITY_TYPES,
    )
    merge_relationships(
        driver,
        "neo4j",
        [{"type": "suppliesTo", "start_id": "company:catl", "end_id": "company:bmw",
          "evidence": "e", "doc_id": "d1"}],
        RELATION_TYPES,
    )
    assert driver.queries and all("MERGE" in q for q in driver.queries)
    assert not any("CREATE (" in q for q in driver.queries)


def test_writer_rejects_labels_outside_the_ontology():
    driver = RecordingDriver()
    with pytest.raises(ValueError, match="unknown entity label"):
        merge_nodes(
            driver, "neo4j",
            [{"canonical_id": "x:1", "label": "Supplier ", "name": "n",
              "aliases": [], "doc_id": "d"}],
            ENTITY_TYPES,
        )
    with pytest.raises(ValueError, match="unknown relation type"):
        merge_relationships(
            driver, "neo4j",
            [{"type": "SUPPLIES_TO", "start_id": "a", "end_id": "b",
              "evidence": "", "doc_id": "d"}],
            RELATION_TYPES,
        )
