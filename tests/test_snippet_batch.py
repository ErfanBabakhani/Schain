import json

import pytest
from neo4j_graphrag.llm import LLMResponse

from schain.documents import make_document
from schain.kg.extract import build_snippet_batch_prompt, extract_snippet_batch
from schain.pipeline import process_document

from .conftest import StubLLM


class BatchReply:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return LLMResponse(content=json.dumps(self.payload))


def make_entry(doc_id, supplier, customer, evidence):
    return {
        "document_id": doc_id,
        "nodes": [
            {"id": "0", "label": "Company", "properties": {"name": supplier}},
            {"id": "1", "label": "Company", "properties": {"name": customer}},
        ],
        "relationships": [{
            "type": "suppliesTo", "start_node_id": "0", "end_node_id": "1",
            "properties": {"evidence": evidence},
        }],
    }


def test_snippet_batch_maps_graphs_and_evidence_to_each_document(ontology):
    documents = [
        {"doc_id": "doc-catl", "text": "CATL supplies battery cells to BMW."},
        {"doc_id": "doc-panasonic", "text": "Panasonic supplies battery cells to Tesla."},
    ]
    llm = BatchReply({"documents": [
        make_entry("doc-panasonic", "Panasonic", "Tesla",
                   "Panasonic supplies battery cells to Tesla."),
        make_entry("doc-catl", "CATL", "BMW", "CATL supplies battery cells to BMW."),
    ]})

    result = extract_snippet_batch(llm, documents, ontology)

    assert len(llm.prompts) == 1
    assert "never combine evidence across document IDs" in llm.prompts[0]
    assert set(result) == {"doc-catl", "doc-panasonic"}
    catl = result["doc-catl"]
    panasonic = result["doc-panasonic"]
    assert catl["graph"].relationships[0].properties["evidence"] == documents[0]["text"]
    assert panasonic["graph"].relationships[0].properties["evidence"] == documents[1]["text"]
    assert set(catl["chunk_index"]).isdisjoint(panasonic["chunk_index"])
    assert catl["metrics"]["extraction_batch_size"] == 2


def test_snippet_batch_prompt_accepts_at_most_ten_documents(ontology):
    documents = [{"doc_id": f"doc-{index}", "text": f"snippet {index}"}
                 for index in range(10)]
    assert "doc-9" in build_snippet_batch_prompt(documents, ontology)
    with pytest.raises(ValueError, match="between one and ten"):
        build_snippet_batch_prompt(documents + [{"doc_id": "doc-10", "text": "extra"}], ontology)


def test_snippet_batch_does_not_turn_missing_or_cross_document_output_into_zero_reward(ontology):
    documents = [
        {"doc_id": "doc-a", "text": "CATL supplies cells to BMW."},
        {"doc_id": "doc-b", "text": "Panasonic supplies cells to Tesla."},
    ]
    missing = BatchReply({"documents": [
        make_entry("doc-a", "CATL", "BMW", "CATL supplies cells to BMW."),
    ]})
    with pytest.raises(ValueError, match="omitted document IDs"):
        extract_snippet_batch(missing, documents, ontology)

    cross_document = BatchReply({"documents": [
        make_entry("doc-a", "CATL", "BMW", "Panasonic supplies cells to Tesla."),
        make_entry("doc-b", "Panasonic", "Tesla", "Panasonic supplies cells to Tesla."),
    ]})
    with pytest.raises(ValueError, match="outside document"):
        extract_snippet_batch(cross_document, documents, ontology)


def test_batched_graph_uses_existing_per_document_validation_pipeline(
    ontology, ingest_config, connection,
):
    text = "CATL supplies battery cells to BMW."
    document = make_document(text, source_url="https://news.test/catl-bmw")
    llm = BatchReply({"documents": [
        make_entry(document["doc_id"], "CATL", "BMW", text),
    ]})
    extraction = extract_snippet_batch(
        llm, [{"doc_id": document["doc_id"], "text": text}], ontology,
    )[document["doc_id"]]
    pipeline_llm = StubLLM()

    summary = process_document(
        document, ingest_config, connection, pipeline_llm, ontology,
        precomputed_extraction=extraction,
    )

    assert summary["facts"] == 1
    assert summary["status"] == "RESOLVED"
    assert summary["llm_calls"] >= 1
    assert summary["llm_calls_by_stage"]["extraction"] == 1
    assert pipeline_llm.extraction_calls == 0
