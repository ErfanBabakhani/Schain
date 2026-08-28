"""Prompt 1: prompt construction and response parsing.

The extraction call itself is neo4j-graphrag's; what matters here is that our schema and
prompt reach it correctly and that the responses a real model actually produces - valid,
malformed, out-of-ontology, empty - are handled the way the config says.
"""

from __future__ import annotations

import json

import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import LLMInterface, LLMResponse

from schain.documents import make_document
from schain.kg.extract import extract_graph
from schain.ontology import SupplyChainERTemplate, build_graph_schema, render_schema

DOCUMENT = make_document("CATL supplies battery cells to BMW AG. " * 20)


class ReplyingLLM(LLMInterface):
    """Returns a fixed body for every extraction call."""

    def __init__(self, body: str, supports_structured_output: bool = False) -> None:
        super().__init__(model_name="stub")
        self.body = body
        self.supports_structured_output = supports_structured_output
        self.response_formats: list[object] = []
        self.prompts: list[str] = []

    def invoke(self, input, message_history=None, system_instruction=None):
        return LLMResponse(content=self.body)

    async def ainvoke(self, input, message_history=None, system_instruction=None, **kwargs):
        self.prompts.append(input if isinstance(input, str) else str(input))
        self.response_formats.append(kwargs.get("response_format"))
        return LLMResponse(content=self.body)


VALID = json.dumps({
    "nodes": [
        {"id": "0", "label": "Company", "properties": {"name": "CATL"}},
        {"id": "1", "label": "Company", "properties": {"name": "BMW AG"}},
    ],
    "relationships": [
        {"type": "suppliesTo", "start_node_id": "0", "end_node_id": "1",
         "properties": {"evidence": "CATL supplies battery cells to BMW AG."}},
    ],
})


def extract(body, ontology, **kwargs):
    return extract_graph(ReplyingLLM(body), DOCUMENT, ontology, **kwargs)


# --- the prompt ---------------------------------------------------------------------


def test_prompt_carries_the_ontology_and_the_document(ontology):
    llm = ReplyingLLM(VALID)
    extract_graph(llm, DOCUMENT, ontology, chunk_size=100000)
    prompt = llm.prompts[0]
    for label in ontology["entity_types"]:
        assert label in prompt
    for relation in ontology["relation_types"]:
        assert relation in prompt
    assert "(Company)-[:suppliesTo]->(Company)" in prompt   # allowed patterns are shown
    assert "CATL supplies battery cells" in prompt          # the document text
    assert "suppliesTo points from the supplier to the customer" in prompt


def test_schema_is_rendered_compactly_not_as_a_dict_dump(ontology):
    rendered = render_schema(build_graph_schema(ontology).model_dump(exclude_none=True))
    assert "Node labels:" in rendered and "'properties':" not in rendered
    full = SupplyChainERTemplate().format(
        text="T", schema=build_graph_schema(ontology).model_dump(exclude_none=True), examples="E"
    )
    assert len(full) < 2500     # the raw dict dump was over 4000 characters


# --- response parsing --------------------------------------------------------------


def test_valid_response_is_parsed(ontology):
    graph, chunk_index = extract(VALID, ontology, chunk_size=100000)
    assert {node.properties["name"] for node in graph.nodes} == {"CATL", "BMW AG"}
    assert graph.relationships[0].type == "suppliesTo"
    assert graph.relationships[0].properties["evidence"].startswith("CATL supplies")
    assert chunk_index  # chunk provenance is returned alongside


def test_node_ids_are_namespaced_per_chunk(ontology):
    """Two chunks returning id "0" must not collide into one node."""
    graph, chunk_index = extract(VALID, ontology, chunk_size=200, chunk_overlap=0)
    assert len(chunk_index) > 1
    assert len({node.id for node in graph.nodes}) == len(graph.nodes)
    assert all(node.id.split(":")[0] in chunk_index for node in graph.nodes)


def test_json_wrapped_in_prose_and_fences_is_recovered(ontology):
    """Instruct models often add a preamble; the library repairs this."""
    graph, _ = extract(f"Here is the JSON:\n```json\n{VALID}\n```", ontology, chunk_size=100000)
    assert len(graph.nodes) == 2


def test_malformed_json_is_skipped_when_configured_to_ignore(ontology, caplog):
    graph, _ = extract("this is not json", ontology, on_error="IGNORE", chunk_size=100000)
    assert graph.nodes == [] and graph.relationships == []


def test_malformed_json_raises_when_configured_to_raise(ontology):
    with pytest.raises(LLMGenerationError):
        extract("this is not json", ontology, on_error="RAISE", chunk_size=100000)


def test_empty_result_is_not_an_error(ontology):
    graph, _ = extract(json.dumps({"nodes": [], "relationships": []}), ontology,
                       on_error="RAISE", chunk_size=100000)
    assert graph.nodes == []


def test_a_document_yielding_nothing_produces_no_facts(ontology, ingest_config, connection):
    """An extraction with no relationships must not write phantom rows."""
    from schain.kg.facts import graph_to_facts
    from schain.kg.resolve import resolve_entities

    graph, chunk_index = extract(json.dumps({"nodes": [], "relationships": []}), ontology,
                                 chunk_size=100000)
    resolution, _ = resolve_entities(graph, llm=None)
    assert graph_to_facts(graph, DOCUMENT, chunk_index, resolution).empty


def test_out_of_ontology_labels_do_not_reach_the_graph_write(ontology):
    """A model inventing a 'Supplier' label is refused by the writer's allow-list."""
    import pandas as pd

    from schain.kg.facts import FACT_COLUMNS
    from schain.kg.writer import merge_nodes

    row = {column: None for column in FACT_COLUMNS}
    row.update(fact_id="f", doc_id="d", subject_id="supplier:x", subject_type="Supplier",
               subject_name="X", subject_mention="X", relation="suppliesTo",
               object_id="company:bmw", object_type="Company", object_name="BMW",
               object_mention="BMW")
    from schain.kg.writer import facts_to_graph_rows

    nodes, _ = facts_to_graph_rows(pd.DataFrame([row]))
    with pytest.raises(ValueError, match="unknown entity label"):
        merge_nodes(object(), "neo4j", nodes, ontology["entity_types"])


# --- structured output -------------------------------------------------------------


def test_structured_output_is_requested_when_enabled(ontology):
    """llm.structured_output=true must reach the extractor, not be silently ignored."""
    llm = ReplyingLLM(VALID, supports_structured_output=True)
    graph, _ = extract_graph(llm, DOCUMENT, ontology, chunk_size=100000,
                             structured_output=True)
    assert llm.response_formats and llm.response_formats[0] is not None
    assert len(graph.nodes) == 2


def test_structured_output_is_refused_on_a_model_that_cannot_do_it(ontology):
    llm = ReplyingLLM(VALID, supports_structured_output=False)
    with pytest.raises(ValueError, match="structured output"):
        extract_graph(llm, DOCUMENT, ontology, chunk_size=100000, structured_output=True)


def test_json_mode_and_structured_output_are_independent():
    """Prompt 2 is a plain call, so it must keep JSON mode even with schema decoding on."""
    from schain.llm.provider import build_llm

    both = build_llm({"llm": {"model": "m", "base_url": "http://x/v1",
                              "json_mode": True, "structured_output": True}})
    assert both.model_params["response_format"] == {"type": "json_object"}

    off = build_llm({"llm": {"model": "m", "base_url": "http://x/v1", "json_mode": False}})
    assert "response_format" not in off.model_params


def test_non_thinking_mode_is_requested_for_the_local_provider():
    from schain.llm.provider import build_llm

    local = build_llm({"llm": {"provider": "local", "model": "m", "base_url": "http://x/v1"}})
    assert local.model_params["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_hosted_apis_do_not_receive_the_vllm_only_field(monkeypatch):
    """chat_template_kwargs is vLLM-specific and would be rejected by a hosted API."""
    from schain.llm.provider import build_llm

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    hosted = build_llm({"llm": {"provider": "api", "model": "m", "base_url": "https://api.x/v1"}})
    assert "extra_body" not in hosted.model_params


def test_api_provider_without_a_key_is_reported(monkeypatch):
    from schain.llm.provider import build_llm

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        build_llm({"llm": {"provider": "api", "model": "m"}})


def test_unknown_provider_is_reported():
    from schain.llm.provider import build_llm

    with pytest.raises(ValueError, match="unknown llm.provider"):
        build_llm({"llm": {"provider": "magic", "model": "m"}})
