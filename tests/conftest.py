"""Shared fixtures. Nothing here touches the network, an LLM or a database."""

from __future__ import annotations

import json
import socket
import ipaddress
from pathlib import Path

import pytest
from neo4j_graphrag.llm import LLMInterface, LLMResponse

from schain import state
from schain.config import load_config
from schain.ontology import load_ontology

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def offline_unit_tests(request, monkeypatch):
    """A missing HTTP mock must never turn a unit test into a live experiment."""
    if request.node.get_closest_marker("integration"):
        return

    def loopback(host):
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def guarded_connect(sock, address):
        if isinstance(address, tuple) and loopback(address[0]):
            return original_connect(sock, address)
        raise RuntimeError("Live network is disabled in offline tests; mock the provider/transport")

    def guarded_connect_ex(sock, address):
        if isinstance(address, tuple) and loopback(address[0]):
            return original_connect_ex(sock, address)
        raise RuntimeError("Live network is disabled in offline tests; mock the provider/transport")

    def guarded_dns(host, *args, **kwargs):
        if loopback(host):
            return original_getaddrinfo(host, *args, **kwargs)
        raise RuntimeError("Live DNS is disabled in offline tests; mock the provider/transport")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_dns)

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
             "properties": {"evidence": "CATL supplies battery cells to BMW AG for its electric vehicle plants."}},
            {"type": "locatedIn", "start_node_id": "2", "end_node_id": "3",
             "properties": {"evidence": "Contemporary Amperex Technology Co. Limited operates a plant located in Germany."}},
            {"type": "produces", "start_node_id": "5", "end_node_id": "4",
             "properties": {"evidence": "BMW produces electric vehicles."}},
        ],
    }
)

# Prompt 2 merges the abbreviation with the full legal name, and BMW with BMW AG.
DISAMBIGUATION_RESPONSE = json.dumps(
    {"groups": [
        ["CATL", "Contemporary Amperex Technology Co. Limited"],
        ["BMW AG", "BMW"],
    ]}
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
        lowered = prompt.lower()
        if "group the names" in lowered:
            return self.disambiguation
        if "for each candidate, decide" in lowered or "evidence entails" in lowered:
            import re

            ids = re.findall(r'"candidate_fact_id":\s*"([^"]+)"', prompt)
            return json.dumps(
                {"results": [{"candidate_fact_id": cid, "decision": "supports"} for cid in ids]}
            )
        if "repairing one ontology-invalid" in lowered:
            # Inspect only the Evidence field, not the allowed-pattern list.
            evidence_block = prompt
            if "Evidence:" in prompt:
                evidence_block = prompt.split("Evidence:", 1)[1]
                if "Allowed patterns" in evidence_block:
                    evidence_block = evidence_block.split("Allowed patterns", 1)[0]
            e_low = evidence_block.lower()
            if "manufactur" in e_low or " produces " in f" {e_low} ":
                return json.dumps(
                    {"decision": "repair", "relation": "produces", "evidence_support": True}
                )
            return json.dumps(
                {"decision": "reject", "relation": None, "evidence_support": False}
            )
        if "does the evidence explicitly support" in lowered:
            return json.dumps({"decision": "supports"})
        return self.extraction

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
        "rejected_facts": str(tmp_path / "processed/rejected_facts.parquet"),
        "edge_support": str(tmp_path / "processed/edge_support.parquet"),
        "verification_evidence": str(tmp_path / "processed/verification_evidence.parquet"),
        "state_db": str(tmp_path / "state/state.db"),
        "graph_export": str(tmp_path / "outputs/supply_chain_graph.v1.json"),
    }
    config.setdefault("source_reputation", {
        "enabled": True,
        "path": str(REPO_ROOT / "config/source_reputation.yaml"),
    })
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
