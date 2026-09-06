"""Shared fixtures. Nothing here touches the network, an LLM or a database."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from neo4j_graphrag.llm import LLMInterface, LLMResponse

from schain import state
from schain.config import load_config
from schain.ontology import load_ontology

REPO_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_TEXT = (
    "CATL supplies battery cells to BMW AG for its electric vehicle plants.\n"
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
             "properties": {"evidence": "CATL operates a plant located in Germany."}},
            {"type": "produces", "start_node_id": "5", "end_node_id": "4",
             "properties": {"evidence": "BMW produces electric vehicles."}},
        ],
    }
)

# Prompt 2 merges the abbreviation with the full legal name; BMW stays separate.
DISAMBIGUATION_RESPONSE = json.dumps(
    {"groups": [["CATL", "Contemporary Amperex Technology Co. Limited"], ["BMW AG"]]}
)


class StubLLM(LLMInterface):
    """Answers Prompt 1 and Prompt 2 with fixed JSON and records the prompts it saw."""

    def __init__(self, extraction: str = EXTRACTION_RESPONSE,
                 disambiguation: str = DISAMBIGUATION_RESPONSE) -> None:
        super().__init__(model_name="stub")
        self.extraction = extraction
        self.disambiguation = disambiguation
        self.prompts: list[str] = []

    def _answer(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.disambiguation if "Group the names" in prompt else self.extraction

    @property
    def extraction_calls(self) -> int:
        return sum(1 for p in self.prompts if "Input text:" in p)

    @property
    def disambiguation_calls(self) -> int:
        return sum(1 for p in self.prompts if "Group the names" in p)

    def invoke(self, input, message_history=None, system_instruction=None):
        return LLMResponse(content=self._answer(input))

    async def ainvoke(self, input, message_history=None, system_instruction=None):
        return LLMResponse(content=self._answer(input))


class FailingLLM(StubLLM):
    """Fails Prompt 1, to exercise the FAILED path."""

    async def ainvoke(self, input, message_history=None, system_instruction=None):
        raise RuntimeError("model unavailable")


def _with_tmp_storage(config: dict, tmp_path: Path) -> dict:
    """Point every storage path at a temporary directory."""
    config["storage"] = {
        "raw_documents": str(tmp_path / "raw/documents.jsonl"),
        "normalized_documents": str(tmp_path / "processed/documents.parquet"),
        "extracted_facts": str(tmp_path / "processed/facts.parquet"),
        "extractions": str(tmp_path / "processed/extractions"),
        "search_results": str(tmp_path / "processed/search_results.parquet"),
        "entity_resolution": str(tmp_path / "processed/entity_resolution.parquet"),
        "state_db": str(tmp_path / "state/state.db"),
        "graph_export": str(tmp_path / "outputs/supply_chain_graph.v1.json"),
    }
    return config


@pytest.fixture
def ingest_config(tmp_path):
    """The real config/ingest.yaml, with storage redirected into tmp_path."""
    return _with_tmp_storage(load_config(REPO_ROOT / "config/ingest.yaml"), tmp_path)


@pytest.fixture
def expand_config(tmp_path):
    """The real config/expansion.yaml, with storage redirected into tmp_path."""
    return _with_tmp_storage(load_config(REPO_ROOT / "config/expansion.yaml"), tmp_path)


@pytest.fixture
def ontology():
    return load_ontology()


@pytest.fixture
def connection(ingest_config):
    connection = state.connect(Path(ingest_config["storage"]["state_db"]))
    yield connection
    connection.close()


@pytest.fixture
def document_file(tmp_path):
    path = tmp_path / "catl_bmw.txt"
    path.write_text(DOCUMENT_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def stub_llm():
    return StubLLM()
