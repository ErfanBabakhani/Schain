"""Targeted regressions for the v0.3.0 reliability and grounding fixes."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pandas as pd
from neo4j_graphrag.components.types import Neo4jGraph
from neo4j_graphrag.llm import LLMResponse

from schain import state
from schain.checkpoint import atomic_write_json
from schain.documents import make_document
from schain.graph_export import build_graph_bundle
from schain.ingest import run_ingest
from schain.kg.facts import FACT_COLUMNS, upstream_companies
from schain.kg.integrity import enforce_referential_integrity
from schain.kg.validate import validate_facts_frame
from schain.names import canonical_id
from schain.pipeline import process_document

from .conftest import StubLLM


def _fact(subject: str, relation: str, object_name: str, evidence: str, *,
          object_type: str = "Company", origin_entity: str | None = None) -> pd.DataFrame:
    row = {column: None for column in FACT_COLUMNS}
    row.update(
        fact_id="model-fact-id",
        doc_id="rimac-doc",
        chunk_index=0,
        subject_id=canonical_id("Company", subject),
        subject_type="Company",
        subject_name=subject,
        subject_mention=subject,
        relation=relation,
        object_id=canonical_id(object_type, object_name),
        object_type=object_type,
        object_name=object_name,
        object_mention=object_name,
        evidence=evidence,
        model="stub",
        origin_entity=origin_entity,
        expansion_relevant=False,
    )
    return pd.DataFrame([row], columns=FACT_COLUMNS)


def test_missing_endpoint_is_rejected_with_lossless_diagnostic():
    graph = Neo4jGraph.model_validate({
        "nodes": [
            {"id": "chunk-a:0", "label": "Company", "properties": {"name": "Rimac Technology"}},
        ],
        "relationships": [{
            "type": "locatedIn",
            "start_node_id": "chunk-a:0",
            "end_node_id": "chunk-a:9",
            "properties": {"evidence": "Rimac Technology is headquartered in Croatia."},
        }],
    })

    clean, failures = enforce_referential_integrity(
        graph,
        document_id="rimac-doc",
        chunk_index_by_id={"chunk-a": 3},
    )

    assert clean.relationships == []
    assert len(failures) == 1
    failure = failures[0]
    assert failure["chunk_id"] == "chunk-a" and failure["chunk_index"] == 3
    assert failure["missing_node_ids"] == ["chunk-a:9"]
    assert failure["reason"] == "missing_relationship_endpoint"
    assert failure["repair_attempted"] is False
    assert failure["original_relationship"]["properties"]["evidence"].endswith("Croatia.")


def test_legacy_dangling_checkpoint_is_safely_upgraded(
    tmp_path, ingest_config, connection, ontology,
):
    document = make_document("Rimac Technology is headquartered in Croatia.")
    doc_id, _ = state.register_document(connection, document)
    state.set_document_status(connection, doc_id, state.EXTRACTED)
    extraction = Path(ingest_config["storage"]["extractions"]) / f"{doc_id}.json"
    atomic_write_json(extraction, {
        "graph": {
            "nodes": [{
                "id": "legacy:0",
                "label": "Company",
                "properties": {"name": "Rimac Technology"},
            }],
            "relationships": [{
                "type": "locatedIn",
                "start_node_id": "legacy:0",
                "end_node_id": "legacy:1",
                "properties": {"evidence": document["text"]},
            }],
        },
        "chunk_index": {"legacy": 0},
    })

    llm = StubLLM(disambiguation='{"groups": []}')
    result = process_document(document, ingest_config, connection, llm, ontology)

    assert result["status"] == state.RESOLVED and result["facts"] == 0
    assert llm.extraction_calls == 0
    upgraded = json.loads(extraction.read_text(encoding="utf-8"))
    assert upgraded["graph"]["relationships"] == []
    assert upgraded["integrity_failures"][0]["original_relationship"]["end_node_id"] == "legacy:1"
    diagnostic = json.loads((tmp_path / "processed" / "extraction_integrity.json").read_text())
    assert diagnostic[0]["reason"] == "missing_relationship_endpoint"


def test_exact_evidence_remains_unchanged(ingest_config, ontology):
    sentence = "Rimac Technology supplies battery systems to BMW Group."
    facts = _fact("Rimac Technology", "suppliesTo", "BMW Group", sentence)
    # Simulate an older Parquet row after additive columns are projected into it.
    facts.at[0, "original_evidence"] = float("nan")
    accepted, rejected = validate_facts_frame(
        facts,
        config=ingest_config,
        ontology=ontology,
        llm=StubLLM(),
        document_text=sentence,
    )

    assert rejected.empty and len(accepted) == 1
    assert accepted.iloc[0].evidence == sentence
    assert accepted.iloc[0].original_evidence == sentence
    assert not bool(accepted.iloc[0].evidence_repair_attempted)
    assert accepted.iloc[0].evidence_repair_status == "not_needed"


def test_paraphrased_evidence_recovers_exact_quote_then_requires_semantics(
    ingest_config, ontology,
):
    source = (
        "Rimac Technology, a Croatian automotive technology company, supplies "
        "high-voltage battery systems to BMW Group."
    )
    model_evidence = "Rimac Technology supplies high-voltage battery systems to BMW Group."
    accepted, rejected = validate_facts_frame(
        _fact("Rimac Technology", "suppliesTo", "BMW Group", model_evidence),
        config=ingest_config,
        ontology=ontology,
        llm=StubLLM(),
        document_text=source,
    )

    assert rejected.empty and len(accepted) == 1
    repaired = accepted.iloc[0]
    assert repaired.evidence == source
    assert repaired.original_evidence == model_evidence
    assert bool(repaired.evidence_repair_attempted)
    assert repaired.evidence_repair_status == "succeeded"
    assert bool(repaired.evidence_validated)

    unverified, rejected = validate_facts_frame(
        _fact("Rimac Technology", "suppliesTo", "BMW Group", model_evidence),
        config=ingest_config,
        ontology=ontology,
        llm=None,
        document_text=source,
    )
    assert unverified.empty
    assert rejected.iloc[0].rejection_reason == "evidence_repair_semantic_unverified"
    assert rejected.iloc[0].evidence_repair_reason == "semantic_validation_unavailable"


def test_quote_recovery_cannot_rewrite_supply_into_vehicle_production(ingest_config, ontology):
    source = "Rimac Technology supplies battery systems used in the BMW i7 to BMW Group."
    accepted, rejected = validate_facts_frame(
        _fact(
            "Rimac Technology",
            "produces",
            "BMW i7",
            "Rimac Technology produces the BMW i7.",
            object_type="Product",
        ),
        config=ingest_config,
        ontology=ontology,
        llm=StubLLM(),
        document_text=source,
    )
    assert accepted.empty
    assert rejected.iloc[0].rejection_reason == "evidence_not_in_source"
    assert rejected.iloc[0].evidence_repair_status == "failed"


def test_recovery_rejects_negation_reverse_direction_and_collaboration(ingest_config, ontology):
    cases = [
        (
            _fact("Rimac Technology", "suppliesTo", "BMW Group", "Rimac supplies BMW."),
            "Rimac Technology does not supply battery systems to BMW Group.",
        ),
        (
            _fact("BMW Group", "suppliesTo", "Rimac Technology", "BMW supplies Rimac."),
            "BMW Group buys battery systems from Rimac Technology.",
        ),
        (
            _fact("Rimac Technology", "suppliesTo", "BMW Group", "Rimac supplies BMW."),
            "Rimac Technology collaborates with BMW Group on battery research.",
        ),
    ]
    for facts, source in cases:
        accepted, rejected = validate_facts_frame(
            facts,
            config=ingest_config,
            ontology=ontology,
            llm=StubLLM(),
            document_text=source,
        )
        assert accepted.empty and len(rejected) == 1


def test_semantic_positive_and_negative_relation_examples(ingest_config, ontology):
    cases = [
        ("produces", "battery module", "Product", "Rimac Technology manufactures a battery module.", True),
        ("locatedIn", "Croatia", "Location", "Rimac Technology is headquartered in Croatia.", True),
        ("locatedIn", "Croatia", "Location", "Rimac Technology attended a conference in Croatia.", False),
    ]
    for relation, target, target_type, source, expected in cases:
        class SemanticLLM(StubLLM):
            def _answer(self, prompt: str) -> str:
                if "For each candidate, decide" in prompt:
                    ids = re.findall(r'"candidate_fact_id":\s*"([^"]+)"', prompt)
                    decision = "supports" if expected else "unclear"
                    return json.dumps({
                        "results": [
                            {"candidate_fact_id": candidate_id, "decision": decision}
                            for candidate_id in ids
                        ]
                    })
                return super()._answer(prompt)

        accepted, _ = validate_facts_frame(
            _fact("Rimac Technology", relation, target, source, object_type=target_type),
            config=ingest_config,
            ontology=ontology,
            llm=SemanticLLM(),
            document_text=source,
        )
        assert (len(accepted) == 1) is expected


class _SameLoopLLM(StubLLM):
    """A client surrogate that fails if reused from a different event loop."""

    def __init__(self) -> None:
        super().__init__(disambiguation='{"groups": []}')
        self.loop_ids: set[int] = set()

    def _answer(self, prompt: str) -> str:
        if "Input text:" in prompt:
            self.prompts.append(prompt)
            source = prompt.rsplit("Input text:", 1)[-1].strip()
            match = re.search(r"(Supplier\d+) supplies components to BMW Group\.", source)
            assert match
            supplier = match.group(1)
            sentence = f"{supplier} supplies components to BMW Group."
            return json.dumps({
                "nodes": [
                    {"id": "0", "label": "Company", "properties": {"name": supplier}},
                    {"id": "1", "label": "Company", "properties": {"name": "BMW Group"}},
                ],
                "relationships": [{
                    "type": "suppliesTo",
                    "start_node_id": "0",
                    "end_node_id": "1",
                    "properties": {"evidence": sentence},
                }],
            })
        return super()._answer(prompt)

    async def ainvoke(self, input, message_history=None, system_instruction=None, **kwargs):
        loop_id = id(asyncio.get_running_loop())
        if self.loop_ids and loop_id not in self.loop_ids:
            raise RuntimeError("Event loop is closed")
        self.loop_ids.add(loop_id)
        return LLMResponse(content=self._answer(input))


def test_three_documents_reuse_one_live_async_loop_and_resume_without_duplicates(
    tmp_path, ingest_config, connection, ontology,
):
    corpus = tmp_path / "three-documents"
    corpus.mkdir()
    for index in range(3):
        (corpus / f"supplier-{index}.txt").write_text(
            f"Supplier{index} supplies components to BMW Group.",
            encoding="utf-8",
        )

    llm = _SameLoopLLM()
    first = run_ingest(ingest_config, connection, llm, ontology, None, corpus)
    facts = pd.read_parquet(ingest_config["storage"]["extracted_facts"])
    assert first["documents_processed"] == 3 and first["documents_failed"] == 0
    assert len(llm.loop_ids) == 1
    assert len(facts) == 3 and facts.fact_id.is_unique

    resumed = _SameLoopLLM()
    second = run_ingest(ingest_config, connection, resumed, ontology, None, corpus)
    assert second["documents_skipped"] == 3 and resumed.prompts == []
    assert len(pd.read_parquet(ingest_config["storage"]["extracted_facts"])) == 3


def test_graph_export_shape_and_downstream_tier_safety(ingest_config):
    origin = canonical_id("Company", "Rimac Technology")
    downstream = _fact(
        "Rimac Technology",
        "suppliesTo",
        "BMW Group",
        "Rimac Technology supplies batteries to BMW Group.",
        origin_entity=origin,
    )
    downstream.loc[0, ["ontology_valid", "evidence_validated", "direction_explicit",
                       "evidence_quote_grounded", "validation_status"]] = [True, True, True, True, "accepted"]
    facts_path = Path(ingest_config["storage"]["extracted_facts"])
    facts_path.parent.mkdir(parents=True, exist_ok=True)
    downstream.to_parquet(facts_path, index=False)

    bundle = build_graph_bundle(ingest_config, mode="export")
    assert {"schema_version", "nodes", "edges", "statistics"}.issubset(bundle)
    assert bundle["edges"][0]["source"] == canonical_id("Company", "Rimac Technology")
    assert bundle["edges"][0]["target"] == canonical_id("Company", "BMW Group")
    assert upstream_companies(downstream, {origin}, ["suppliesTo"]) == []


def test_audit_records_structure_repair_acceptance_and_cache_hit(
    tmp_path, ingest_config, connection, ontology,
):
    source = (
        "Rimac Technology, a Croatian automotive technology company, supplies "
        "high-voltage battery systems to BMW Group. "
        "Rimac Technology is headquartered in Croatia."
    )
    response = json.dumps({
        "nodes": [
            {"id": "0", "label": "Company", "properties": {"name": "Rimac Technology"}},
            {"id": "1", "label": "Company", "properties": {"name": "BMW Group"}},
        ],
        "relationships": [
            {
                "type": "suppliesTo", "start_node_id": "0", "end_node_id": "1",
                "properties": {"evidence": "Rimac Technology supplies high-voltage battery systems to BMW Group."},
            },
            {
                "type": "locatedIn", "start_node_id": "0", "end_node_id": "2",
                "properties": {"evidence": "Rimac Technology is headquartered in Croatia."},
            },
        ],
    })
    ingest_config["audit"] = {
        "enabled": True,
        "capture_content": True,
        "path": str(tmp_path / "audit"),
        "run_id": "audit-run",
    }
    document = make_document(source)
    llm = StubLLM(extraction=response, disambiguation='{"groups": []}')
    first = process_document(document, ingest_config, connection, llm, ontology)
    assert first["accepted_facts"] == 1
    process_document(document, ingest_config, connection, llm, ontology)

    audit_file = tmp_path / "audit" / "audit-run" / f"{document['doc_id']}.jsonl"
    records = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").splitlines()]
    events = {record["event"] for record in records}
    assert {
        "llm_call_started", "llm_call_finished", "parsed_extraction_graph",
        "structural_integrity_checked", "entity_resolution_completed",
        "candidate_facts_created", "grounding_checked", "evidence_repair_succeeded",
        "candidate_accepted", "validation_completed", "document_cache_hit",
    }.issubset(events)
    structural = next(record for record in records if record["event"] == "structural_integrity_checked")
    assert structural["valid"] is False
    assert structural["failures"][0]["original_relationship"]["end_node_id"].endswith(":2")
    assert any("rendered_prompt" in record for record in records if record["event"] == "llm_call_started")
    assert any("raw_model_response" in record for record in records if record["event"] == "llm_call_finished")
    diagnostic = json.loads((tmp_path / "processed" / "extraction_integrity.json").read_text())
    assert diagnostic[0]["original_relationship"]["properties"]["evidence"].endswith("Croatia.")


def test_audit_disabled_creates_no_audit_output(
    tmp_path, ingest_config, connection, ontology,
):
    audit_path = tmp_path / "disabled-audit"
    ingest_config["audit"] = {"enabled": False, "path": str(audit_path)}
    document = make_document("Rimac Technology supplies batteries to BMW Group.")
    response = json.dumps({
        "nodes": [
            {"id": "0", "label": "Company", "properties": {"name": "Rimac Technology"}},
            {"id": "1", "label": "Company", "properties": {"name": "BMW Group"}},
        ],
        "relationships": [{
            "type": "suppliesTo", "start_node_id": "0", "end_node_id": "1",
            "properties": {"evidence": "Rimac Technology supplies batteries to BMW Group."},
        }],
    })
    process_document(
        document,
        ingest_config,
        connection,
        StubLLM(extraction=response, disambiguation='{"groups": []}'),
        ontology,
    )
    assert not audit_path.exists()
