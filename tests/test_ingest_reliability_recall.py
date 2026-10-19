"""Regressions for the 2026-10 ingest audit failures."""

from __future__ import annotations

import json
import re

import pandas as pd
from neo4j_graphrag.components.types import Neo4jGraph
from neo4j_graphrag.llm import LLMResponse

from schain.kg.integrity import enforce_referential_integrity
from schain.kg.resolve import resolve_entities
from schain.kg.validate import _batch_evidence_validate, validate_facts_frame
from schain.ontology import EXTRACTION_EXAMPLES, SupplyChainERTemplate
from schain.graph_export import build_graph_bundle
from schain.ingest import run_ingest


class EventAudit:
    capture_content = True

    def __init__(self):
        self.events: list[dict] = []

    def emit(self, event, *, stage, **payload):
        self.events.append({"event_type": event, "pipeline_stage": stage, **payload})

    def contextual_llm_call_ids(self, *args):
        return []


def _candidate_rows(count=10):
    return [
        {
            "candidate_fact_id": f"candidate-{index}",
            "subject_name": f"Supplier {index}",
            "subject_type": "Company",
            "relation": "suppliesTo",
            "object_name": f"Customer {index}",
            "object_type": "Company",
            "evidence": f"Supplier {index} supplies cells to Customer {index}.",
            "_relation_policy": {},
        }
        for index in range(count)
    ]


def _prompt_ids(prompt: str) -> list[str]:
    return [
        candidate_id
        for candidate_id in re.findall(r'"candidate_fact_id":\s*"([^"]+)"', prompt)
        if candidate_id != "..."
    ]


class IncompleteThenValidLLM:
    def __init__(self, *, always_incomplete=False, defects=False):
        self.calls: list[list[str]] = []
        self.always_incomplete = always_incomplete
        self.defects = defects

    def invoke(self, prompt):
        ids = _prompt_ids(prompt)
        self.calls.append(ids)
        if len(self.calls) == 1 and self.defects:
            return LLMResponse(content=json.dumps({"results": [
                {"candidate_fact_id": ids[0], "decision": "supports"},
                {"candidate_fact_id": ids[0], "decision": "supports"},
                {"candidate_fact_id": "unknown-id", "decision": "supports"},
                {"candidate_fact_id": ids[1], "decision": "maybe"},
                {"decision": "supports"},
                *[
                    {"candidate_fact_id": cid, "decision": "supports"}
                    for cid in ids[3:]
                ],
            ]}))
        returned = [] if self.always_incomplete else ids
        if len(self.calls) == 1 and not self.always_incomplete:
            returned = ids[:-1]
        return LLMResponse(content=json.dumps({"results": [
            {"candidate_fact_id": cid, "decision": "supports"}
            for cid in returned
        ]}))


def test_incomplete_ten_item_batch_retries_only_missing_id_and_preserves_nine():
    rows = _candidate_rows()
    audit = EventAudit()
    rows[0]["_semantic_validation_options"] = {"max_attempts": 2, "retry_batch_size": 1}
    rows[0]["_audit_trace"] = audit
    llm = IncompleteThenValidLLM()

    decisions = _batch_evidence_validate(llm, rows)

    assert decisions == {row["candidate_fact_id"]: "supports" for row in rows}
    assert len(llm.calls) == 2
    assert llm.calls[0] == [row["candidate_fact_id"] for row in rows]
    assert llm.calls[1] == [rows[-1]["candidate_fact_id"]]
    assert any(event["event_type"] == "semantic_validation_targeted_retry"
               for event in audit.events)


def test_audit_capture_does_not_change_semantic_validation_decisions():
    plain_rows = _candidate_rows(4)
    audited_rows = _candidate_rows(4)
    options = {"max_attempts": 2, "retry_batch_size": 1}
    plain_rows[0]["_semantic_validation_options"] = options
    audited_rows[0]["_semantic_validation_options"] = options
    audited_rows[0]["_audit_trace"] = EventAudit()

    without_audit = _batch_evidence_validate(IncompleteThenValidLLM(), plain_rows)
    with_audit = _batch_evidence_validate(IncompleteThenValidLLM(), audited_rows)

    assert with_audit == without_audit


def test_duplicate_unknown_malformed_and_invalid_ids_are_targeted_for_retry():
    rows = _candidate_rows(5)
    audit = EventAudit()
    rows[0]["_semantic_validation_options"] = {"max_attempts": 2, "retry_batch_size": 1}
    rows[0]["_audit_trace"] = audit
    llm = IncompleteThenValidLLM(defects=True)

    decisions = _batch_evidence_validate(llm, rows)

    assert set(decisions) == {row["candidate_fact_id"] for row in rows}
    assert all(value == "supports" for value in decisions.values())
    assert llm.calls[1:] == [["candidate-0"], ["candidate-1"], ["candidate-2"]]
    first = next(event for event in audit.events
                 if event["event_type"] == "semantic_validation_batch_response")
    assert first["duplicate_candidate_fact_ids"] == ["candidate-0"]
    assert first["invalid_candidate_fact_ids"] == ["candidate-1"]
    assert first["missing_candidate_fact_ids"] == ["candidate-2"]
    assert first["unknown_candidate_fact_ids"] == ["unknown-id"]
    assert first["malformed_item_count"] == 1


def test_exhausted_incomplete_decisions_fail_closed_without_global_exception():
    rows = _candidate_rows(2)
    audit = EventAudit()
    rows[0]["_semantic_validation_options"] = {"max_attempts": 2, "retry_batch_size": 1}
    rows[0]["_audit_trace"] = audit

    decisions = _batch_evidence_validate(
        IncompleteThenValidLLM(always_incomplete=True), rows,
    )

    assert decisions == {"candidate-0": "incomplete", "candidate-1": "incomplete"}
    exhausted = [event for event in audit.events
                 if event["event_type"] == "semantic_validation_unresolved"]
    assert exhausted[0]["final_policy"] == "reject_fail_closed"


def test_exhausted_candidate_is_rejected_while_valid_peer_is_persistable(
    ingest_config, ontology,
):
    class OneDecisionLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            ids = _prompt_ids(prompt)
            returned = ids[:1] if self.calls == 1 else []
            return LLMResponse(content=json.dumps({"results": [
                {"candidate_fact_id": cid, "decision": "supports"}
                for cid in returned
            ]}))

    facts = pd.DataFrame([
        {
            "doc_id": "doc", "chunk_index": 0,
            "subject_id": "company:northstar", "subject_name": "Northstar Cells",
            "subject_type": "Company", "relation": "produces",
            "object_id": "material:modules", "object_name": "battery modules",
            "object_type": "Material",
            "evidence": "Northstar Cells manufactures battery modules.",
        },
        {
            "doc_id": "doc", "chunk_index": 0,
            "subject_id": "company:arbor", "subject_name": "Arbor Mobility",
            "subject_type": "Company", "relation": "produces",
            "object_id": "product:a4", "object_name": "A4 electric vehicle",
            "object_type": "Product",
            "evidence": "Arbor Mobility builds the A4 electric vehicle.",
        },
    ])
    source = " ".join(facts.evidence.tolist())

    accepted, rejected = validate_facts_frame(
        facts, config=ingest_config, ontology=ontology,
        llm=OneDecisionLLM(), document_text=source,
    )

    assert accepted.subject_name.tolist() == ["Northstar Cells"]
    assert rejected.subject_name.tolist() == ["Arbor Mobility"]
    assert rejected.rejection_reason.tolist() == ["semantic_validation_incomplete"]


def _integrity_graph(nodes, relationships):
    return Neo4jGraph.model_validate({"nodes": nodes, "relationships": relationships})


def test_missing_location_node_is_reconstructed_from_subject_bound_verbatim_evidence(ontology):
    graph = _integrity_graph(
        [{"id": "c:0", "label": "Company", "properties": {"name": "Portwell"}}],
        [{"type": "locatedIn", "start_node_id": "c:0", "end_node_id": "c:9",
          "properties": {"evidence": "Portwell is headquartered in Setúbal, Portugal."}}],
    )
    outcomes = []

    repaired, failures = enforce_referential_integrity(
        graph, document_id="doc", chunk_index_by_id={"c": 0},
        ontology=ontology, recovery_outcomes=outcomes,
    )

    assert failures == [] and len(repaired.relationships) == 1
    location = next(node for node in repaired.nodes if node.label == "Location")
    assert location.properties["name"] == "Setúbal, Portugal"
    assert repaired.relationships[0].end_node_id == location.id
    event = next(item for item in outcomes
                 if item["event_type"] == "endpoint_node_reconstructed")
    assert event["original_endpoint_id"] == "c:9"


def test_compound_location_evidence_is_bounded_to_the_named_subject(ontology):
    graph = _integrity_graph(
        [
            {"id": "c:0", "label": "Company", "properties": {"name": "Portwell"}},
            {"id": "c:1", "label": "Company", "properties": {"name": "AstraForge"}},
        ],
        [
            {"type": "locatedIn", "start_node_id": "c:0", "end_node_id": "c:9",
             "properties": {"evidence": (
                 "Portwell is headquartered in Setubal, Portugal, whereas AstraForge "
                 "has its registered headquarters in Ghent, Belgium."
             )}},
            {"type": "locatedIn", "start_node_id": "c:1", "end_node_id": "c:10",
             "properties": {"evidence": (
                 "Portwell is headquartered in Setubal, Portugal, whereas AstraForge "
                 "has its registered headquarters in Ghent, Belgium."
             )}},
        ],
    )

    repaired, failures = enforce_referential_integrity(
        graph, document_id="doc", chunk_index_by_id={"c": 0}, ontology=ontology,
    )

    assert failures == []
    locations = {
        node.properties["name"] for node in repaired.nodes if node.label == "Location"
    }
    assert locations == {"Setubal, Portugal", "Ghent, Belgium"}


def test_name_instead_of_id_recovers_unique_existing_node_across_chunks(ontology):
    graph = _integrity_graph(
        [
            {"id": "a:0", "label": "Company", "properties": {"name": "Meridian Energy Storage"}},
            {"id": "b:1", "label": "Company", "properties": {"name": "Solstice Mobility"}},
        ],
        [{"type": "suppliesTo", "start_node_id": "b:Meridian Energy Storage",
          "end_node_id": "b:1",
          "properties": {"evidence": "Solstice Mobility buys modules from Meridian Energy Storage."}}],
    )
    outcomes = []

    repaired, failures = enforce_referential_integrity(
        graph, document_id="doc", chunk_index_by_id={"a": 0, "b": 1},
        ontology=ontology, recovery_outcomes=outcomes,
    )

    assert failures == []
    assert repaired.relationships[0].start_node_id == "a:0"
    assert any(item["event_type"] == "endpoint_id_recovered" for item in outcomes)


def test_ambiguous_unknown_and_type_incompatible_endpoints_remain_dropped(ontology):
    cases = [
        (
            [
                {"id": "a:0", "label": "Company", "properties": {"name": "Atlas"}},
                {"id": "a:1", "label": "Mine", "properties": {"name": "Atlas"}},
                {"id": "a:2", "label": "Company", "properties": {"name": "Buyer"}},
            ],
            "a:Atlas", "endpoint_recovery_ambiguous",
        ),
        ([{"id": "a:2", "label": "Company", "properties": {"name": "Buyer"}}],
         "a:Unknown", "endpoint_unrecoverable"),
        ([
            {"id": "a:0", "label": "Product", "properties": {"name": "Wrong Type"}},
            {"id": "a:2", "label": "Company", "properties": {"name": "Buyer"}},
         ], "a:Wrong Type", "endpoint_unrecoverable"),
    ]
    for nodes, missing_start, expected_event in cases:
        graph = _integrity_graph(nodes, [{
            "type": "suppliesTo", "start_node_id": missing_start, "end_node_id": "a:2",
            "properties": {"evidence": "A supplier sells parts to Buyer."},
        }])
        outcomes = []
        repaired, failures = enforce_referential_integrity(
            graph, document_id="doc", chunk_index_by_id={"a": 0},
            ontology=ontology, recovery_outcomes=outcomes,
        )
        assert repaired.relationships == [] and len(failures) == 1
        assert any(item["event_type"] == expected_event for item in outcomes)


class GroupingLLM:
    def __init__(self, groups):
        self.groups = groups

    def invoke(self, prompt):
        return LLMResponse(content=json.dumps({"groups": self.groups}))


def _company_graph(*names):
    return _integrity_graph([
        {"id": f"c:{index}", "label": "Company", "properties": {"name": name}}
        for index, name in enumerate(names)
    ], [])


def test_llm_short_name_merge_requires_collision_free_document_support():
    audit = EventAudit()
    resolution, _ = resolve_entities(
        _company_graph("Meridian Energy Storage", "Meridian"),
        llm=GroupingLLM([["Meridian Energy Storage", "Meridian"]]),
        document_text=(
            "Meridian Energy Storage manufactures modules. Meridian supplies them to Solstice."
        ),
        audit=audit,
    )
    assert len({item["canonical_id"] for item in resolution.values()}) == 1
    assert any(event["event_type"] == "alias_merge_evaluated" and event["accepted"]
               for event in audit.events)


def test_ambiguous_short_name_and_distinct_subsidiary_are_not_merged():
    graph = _company_graph("Atlas Motors", "Atlas Mining", "Atlas")
    resolution, _ = resolve_entities(
        graph,
        llm=GroupingLLM([["Atlas Motors", "Atlas"], ["Atlas Mining"]]),
        document_text="Atlas Motors and Atlas Mining are different companies. Atlas won an award.",
    )
    assert len({item["canonical_id"] for item in resolution.values()}) == 3


def test_explicit_subsidiary_is_not_treated_as_parent_shorthand():
    graph = _company_graph("Acme", "Acme Battery")
    resolution, _ = resolve_entities(
        graph,
        llm=GroupingLLM([["Acme", "Acme Battery"]]),
        document_text=(
            "Acme Battery is a subsidiary of Acme. Acme Battery manufactures cells "
            "while Acme provides group financing."
        ),
    )
    assert len({item["canonical_id"] for item in resolution.values()}) == 2


def test_produces_prompt_balances_positive_and_hard_negative_examples():
    prompt = SupplyChainERTemplate.DEFAULT_TEMPLATE + EXTRACTION_EXAMPLES
    assert "manufactures battery modules" in prompt
    assert "builds the A4 electric vehicle" in prompt
    assert "does not manufacture the A4" in prompt
    assert "could manufacture cells someday" in prompt
    assert "component maker produces the component" in prompt
    assert "scan the text once for every allowed relation type" in prompt


def test_failed_document_metrics_and_completion_are_truthful(
    tmp_path, ingest_config, connection, ontology, monkeypatch,
):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "failed.txt").write_text("Supplier supplies cells to Buyer.", encoding="utf-8")
    error = ValueError("controlled validation failure")
    error._schain_document_summary = {
        "elapsed_seconds": 3.5,
        "llm_calls": 26,
        "extraction_chunks": 7,
        "extraction_calls": 7,
        "extraction_cache_hits": 0,
        "extraction_retries": 0,
        "candidate_facts": 38,
        "accepted_facts": 0,
        "rejected_facts": 0,
    }

    def fail_document(*args, **kwargs):
        raise error

    monkeypatch.setattr("schain.ingest.process_document", fail_document)
    summary = run_ingest(
        ingest_config, connection, llm=None, ontology=ontology,
        input_path=input_dir,
    )

    assert summary["documents_failed"] == 1
    assert summary["llm_calls"] == 26
    assert summary["extraction_chunks"] == 7
    assert summary["candidate_facts"] == 38
    assert summary["completion"] == {
        "complete": False,
        "export_success": False,
        "pipeline_processing_complete": False,
        "documents_failed": 1,
        "documents_skipped": 0,
        "graph_has_accepted_facts": False,
        "stop_reason": "document_failures",
    }


def test_graph_bundle_uses_ingest_processing_completion_not_empty_frontier(
    ingest_config, ontology,
):
    completion = {
        "complete": False,
        "export_success": True,
        "pipeline_processing_complete": False,
        "documents_failed": 1,
        "documents_skipped": 0,
        "graph_has_accepted_facts": False,
        "stop_reason": "document_failures",
    }
    bundle = build_graph_bundle(
        ingest_config,
        mode="ingest",
        pipeline_summary={"completion": completion, "documents_failed": 1},
    )

    assert bundle["completion"]["complete"] is False
    assert bundle["completion"]["export_success"] is True
    assert bundle["metadata"]["completion"] == bundle["completion"]
    assert bundle["metadata"]["experiment_complete"] is False
