"""End-to-end regressions for the optional run-scoped audit artifact."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import yaml
import pandas as pd
from neo4j_graphrag.llm import LLMResponse

from schain.__main__ import main
from schain.audit_trace import RunAuditTrace, audit_path_for_graph
from schain.llm.provider import MeasuredLLM
from schain.kg.extract import extract_graph
from schain.kg.facts import FACT_COLUMNS
from schain.kg.validate import validate_facts_frame

from .conftest import DOCUMENT_TEXT, StubLLM


def _config_file(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "audit-ingest.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _corpus(tmp_path: Path, count: int = 1) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    for index in range(count):
        text = DOCUMENT_TEXT if count == 1 else DOCUMENT_TEXT + f"Document marker {index}.\n"
        (root / f"document-{index}.txt").write_text(text, encoding="utf-8")
    return root


def _run_cli_with_stub(monkeypatch, config_path: Path, corpus: Path, llm: StubLLM) -> int:
    monkeypatch.setattr("schain.__main__.build_llm", lambda _config: llm)
    return main([
        "ingest", "--config", str(config_path), "--input", str(corpus), "--no-neo4j",
    ])


def test_full_cli_audit_is_companion_to_unchanged_graph(
    tmp_path, ingest_config, monkeypatch, capsys,
):
    monkeypatch.setenv("SCHAIN_AUDIT_TRACE", "1")
    monkeypatch.setenv("SCHAIN_AUDIT_FULL_CONTENT", "1")
    # Secret-looking configuration must be redacted recursively.
    ingest_config["llm"]["api_key"] = "do-not-leak"
    config_path = _config_file(tmp_path, ingest_config)
    corpus = _corpus(tmp_path)
    llm = StubLLM()

    assert _run_cli_with_stub(monkeypatch, config_path, corpus, llm) == 0
    summary = json.loads(capsys.readouterr().out)
    graph_path = Path(summary["graph_export"])
    audit_path = graph_path.with_name(graph_path.stem + ".audit.json")
    assert audit_path == audit_path_for_graph(graph_path)
    assert graph_path.is_file() and audit_path.is_file()

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert set(graph) == {
        "schema_version", "metadata", "ontology", "statistics",
        "nodes", "edges", "documents", "completion",
    }
    assert audit["audit_schema_version"] == "1.0"
    assert audit["run"]["complete"] is True
    assert audit["run"]["full_content"] is True
    assert len(audit["documents"]) == 1 and len(audit["chunks"]) == 1
    assert audit["documents"][0]["normalized_source_text"] == DOCUMENT_TEXT.rstrip()
    assert audit["chunks"][0]["source_text"] == DOCUMENT_TEXT.rstrip()

    extraction_call = next(
        call for call in audit["llm_calls"] if call["pipeline_stage"] == "extraction"
    )
    assert extraction_call["rendered_prompt"] in llm.prompts
    assert extraction_call["assistant_message_content"] == llm.extraction
    assert extraction_call["provider_raw_response"] is None
    assert extraction_call["provider_raw_response_unavailable_reason"]
    assert {item["event_type"] for item in audit["integrity_checks"]} >= {
        "structural_integrity_checked", "relationship_integrity_result",
    }
    assert len(audit["candidate_facts"]) == 3
    assert len(audit["accepted_facts"]) == 3
    assert all(item["fact_id"] for item in audit["accepted_facts"])
    assert audit["graph_export"]["status"] == "success"
    exported_fact_ids = {
        fact_id for item in audit["graph_export"]["edge_fact_map"]
        for fact_id in item["fact_ids"]
    }
    assert exported_fact_ids == set(audit["graph_export"]["accepted_fact_ids_in_export"])
    assert audit["summary"]["reference_integrity"]["valid"] is True
    methods = {item.get("method") for item in audit["entity_resolution"]}
    assert any(method and "llm" in method for method in methods)
    checks = {item.get("check") for item in audit["validation_decisions"]}
    assert {
        "grounding", "entity_type", "generic_entity", "ontology",
        "lexical_evidence", "direction", "semantic_entailment",
    }.issubset(checks)
    raw = audit_path.read_text(encoding="utf-8")
    assert "do-not-leak" not in raw
    assert audit["configuration"]["llm"]["api_key"] == "<redacted>"


def test_sync_async_and_concurrent_calls_have_exact_content_and_unique_ids(tmp_path):
    graph_path = tmp_path / "result.v1.json"
    config = {"llm": {"provider": "local", "model": "stub", "temperature": 0}}
    trace = RunAuditTrace(
        output_path=audit_path_for_graph(graph_path), graph_path=graph_path,
        mode="ingest", config=config, input_path=tmp_path / "input.txt",
        neo4j_enabled=False, capture_content=True,
    )
    audit = trace.document("doc-1")
    wrapped = MeasuredLLM(StubLLM(), {}, audit=audit)

    assert wrapped.invoke("sync exact prompt").content

    async def concurrent() -> None:
        await asyncio.gather(
            wrapped.ainvoke("async exact prompt A"),
            wrapped.ainvoke("async exact prompt B"),
        )

    asyncio.run(concurrent())
    trace.finalize(complete=True, status="completed")
    payload = json.loads(trace.output_path.read_text(encoding="utf-8"))
    calls = payload["llm_calls"]
    assert len(calls) == 3
    assert len({call["llm_call_id"] for call in calls}) == 3
    assert {call["rendered_prompt"] for call in calls} == {
        "sync exact prompt", "async exact prompt A", "async exact prompt B",
    }
    assert all(call["status"] == "success" for call in calls)
    assert [event["sequence"] for event in payload["events"]] == list(
        range(1, len(payload["events"]) + 1)
    )


def test_parser_failure_produces_valid_partial_audit(
    tmp_path, ingest_config, monkeypatch, capsys,
):
    monkeypatch.setenv("SCHAIN_AUDIT_TRACE", "1")
    monkeypatch.setenv("SCHAIN_AUDIT_FULL_CONTENT", "1")
    config_path = _config_file(tmp_path, ingest_config)
    corpus = _corpus(tmp_path)
    assert _run_cli_with_stub(
        monkeypatch, config_path, corpus,
        StubLLM(extraction="this is not extraction JSON"),
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    payload = json.loads(audit_path_for_graph(Path(summary["graph_export"])).read_text())
    assert payload["run"]["complete"] is False
    assert payload["run"]["completion_status"] == "completed_with_document_failures"
    failure = next(
        event for event in payload["events"]
        if event["event_type"] == "chunk_extraction_attempt_failed"
    )
    assert failure["failure_kind"] == "parser_or_schema_error"
    assert any(event["event_type"] == "document_failed" for event in payload["events"])
    assert payload["graph_export"]["status"] == "success"


def test_dangling_relationship_is_an_explicit_loss_event(
    tmp_path, ingest_config, monkeypatch, capsys,
):
    monkeypatch.setenv("SCHAIN_AUDIT_TRACE", "1")
    monkeypatch.setenv("SCHAIN_AUDIT_FULL_CONTENT", "1")
    response = json.dumps({
        "nodes": [{"id": "0", "label": "Company", "properties": {"name": "CATL"}}],
        "relationships": [{
            "type": "suppliesTo", "start_node_id": "0", "end_node_id": "missing",
            "properties": {"evidence": DOCUMENT_TEXT.splitlines()[0]},
        }],
    })
    config_path = _config_file(tmp_path, ingest_config)
    assert _run_cli_with_stub(
        monkeypatch, config_path, _corpus(tmp_path), StubLLM(extraction=response),
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    payload = json.loads(audit_path_for_graph(Path(summary["graph_export"])).read_text())
    outcome = next(
        item for item in payload["relationship_outcomes"]
        if item["event_type"] == "relationship_integrity_result"
    )
    assert outcome["target_node_exists"] is False
    assert outcome["action"] == "dropped_missing_endpoint"
    assert outcome["missing_node_ids"]
    assert payload["candidate_facts"] == []


def test_resume_and_cache_use_are_truthful(
    tmp_path, ingest_config, monkeypatch, capsys,
):
    monkeypatch.setenv("SCHAIN_AUDIT_TRACE", "1")
    config_path = _config_file(tmp_path, ingest_config)
    corpus = _corpus(tmp_path)
    first_llm = StubLLM()
    assert _run_cli_with_stub(monkeypatch, config_path, corpus, first_llm) == 0
    capsys.readouterr()

    # A normal resume skips the resolved document and makes no model call.
    second_llm = StubLLM()
    assert _run_cli_with_stub(monkeypatch, config_path, corpus, second_llm) == 0
    summary = json.loads(capsys.readouterr().out)
    payload = json.loads(audit_path_for_graph(Path(summary["graph_export"])).read_text())
    assert second_llm.prompts == []
    assert any(event["event_type"] == "document_cache_hit" for event in payload["events"])
    assert payload["llm_calls"] == []


def test_chunk_checkpoint_reuse_records_artifact_without_fake_call(
    tmp_path, ingest_config, ontology,
):
    graph_path = tmp_path / "graph.json"
    trace = RunAuditTrace(
        output_path=audit_path_for_graph(graph_path), graph_path=graph_path,
        mode="ingest", config=ingest_config, input_path=tmp_path / "doc.txt",
        neo4j_enabled=False, capture_content=True,
    )
    document = {"doc_id": "cache-doc", "text": DOCUMENT_TEXT.rstrip()}
    checkpoint = tmp_path / "chunks"
    first_metrics, second_metrics = {}, {}
    extract_graph(
        MeasuredLLM(StubLLM(), first_metrics, audit=trace.document("cache-doc")),
        document, ontology, checkpoint_dir=checkpoint, metrics=first_metrics,
        max_retries=0, audit=trace.document("cache-doc"),
    )
    call_count = len(trace.call_ids("cache-doc", "extraction"))
    extract_graph(
        MeasuredLLM(StubLLM(), second_metrics, audit=trace.document("cache-doc")),
        document, ontology, checkpoint_dir=checkpoint, metrics=second_metrics,
        max_retries=0, audit=trace.document("cache-doc"),
    )
    assert second_metrics["extraction_cache_hits"] == 1
    assert second_metrics["extraction_calls"] == 0
    assert len(trace.call_ids("cache-doc", "extraction")) == call_count
    trace.finalize(complete=True, status="completed")
    payload = json.loads(trace.output_path.read_text(encoding="utf-8"))
    cache_event = next(
        event for event in payload["events"]
        if event["event_type"] == "extraction_cache_hit"
    )
    assert cache_event["cache_key"]
    assert cache_event["parsed_artifact"]["nodes"]
    assert cache_event["historical_prompt"] == "unavailable_from_chunk_checkpoint"


def test_validation_audit_records_repair_success_failure_and_exact_rejection(
    tmp_path, ingest_config, ontology,
):
    trace = RunAuditTrace(
        output_path=tmp_path / "validation.audit.json",
        graph_path=tmp_path / "validation.json", mode="ingest",
        config=ingest_config, input_path=None, neo4j_enabled=False,
        capture_content=True,
    )

    def fact(evidence: str) -> pd.DataFrame:
        row = {column: None for column in FACT_COLUMNS}
        row.update(
            fact_id="initial", doc_id="validation-doc", chunk_index=0,
            subject_id="company:rimac", subject_type="Company",
            subject_name="Rimac Technology", subject_mention="Rimac Technology",
            relation="suppliesTo", object_id="company:bmw", object_type="Company",
            object_name="BMW Group", object_mention="BMW Group", evidence=evidence,
        )
        return pd.DataFrame([row], columns=FACT_COLUMNS)

    rich_source = (
        "Rimac Technology, a Croatian automotive technology company, supplies "
        "high-voltage battery systems to BMW Group."
    )
    model_evidence = "Rimac Technology supplies high-voltage battery systems to BMW Group."
    audit = trace.document("validation-doc")
    accepted, rejected = validate_facts_frame(
        fact(model_evidence), config=ingest_config, ontology=ontology,
        llm=MeasuredLLM(StubLLM(), {}, audit=audit),
        document_text=rich_source, audit=audit,
    )
    assert len(accepted) == 1 and rejected.empty

    failed, rejected = validate_facts_frame(
        fact("Rimac Technology supplies BMW Group."),
        config=ingest_config, ontology=ontology, llm=None,
        document_text="Rimac Technology collaborates with BMW Group on research.",
        audit=audit,
    )
    assert failed.empty
    assert rejected.iloc[0].rejection_reason == "evidence_not_in_source"
    trace.finalize(complete=True, status="completed")
    payload = json.loads(trace.output_path.read_text(encoding="utf-8"))
    event_types = {event["event_type"] for event in payload["events"]}
    assert "evidence_repair_succeeded" in event_types
    assert "evidence_repair_failed" in event_types
    rejected_check = next(
        item for item in payload["validation_decisions"]
        if item.get("check") == "grounding" and item.get("decision") == "fail"
    )
    assert rejected_check["reason"] == "evidence_not_in_source"
    skipped = [
        item for item in payload["validation_decisions"]
        if item.get("decision") == "skipped_due_to_earlier_rejection"
    ]
    assert {item["check"] for item in skipped} >= {"ontology", "semantic_entailment"}


def test_multiple_documents_keep_document_and_call_references_separate(
    tmp_path, ingest_config, monkeypatch, capsys,
):
    monkeypatch.setenv("SCHAIN_AUDIT_TRACE", "1")
    config_path = _config_file(tmp_path, ingest_config)
    assert _run_cli_with_stub(
        monkeypatch, config_path, _corpus(tmp_path, count=2), StubLLM(),
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    payload = json.loads(audit_path_for_graph(Path(summary["graph_export"])).read_text())
    document_ids = {document["document_id"] for document in payload["documents"]}
    assert len(document_ids) == 2
    assert {call["document_id"] for call in payload["llm_calls"]} == document_ids
    assert len({call["llm_call_id"] for call in payload["llm_calls"]}) == len(
        payload["llm_calls"]
    )


def test_controlled_startup_exception_writes_partial_audit(
    tmp_path, ingest_config, monkeypatch,
):
    monkeypatch.setenv("SCHAIN_AUDIT_TRACE", "1")
    config_path = _config_file(tmp_path, ingest_config)
    corpus = _corpus(tmp_path)
    monkeypatch.setattr(
        "schain.__main__.build_llm",
        lambda _config: (_ for _ in ()).throw(RuntimeError("startup failed")),
    )
    assert main([
        "ingest", "--config", str(config_path), "--input", str(corpus), "--no-neo4j",
    ]) == 1
    graph_path = Path(ingest_config["storage"]["graph_export"])
    payload = json.loads(audit_path_for_graph(graph_path).read_text(encoding="utf-8"))
    assert payload["run"]["complete"] is False
    assert payload["run"]["error"]["message"] == "startup failed"
    assert payload["graph_export"]["status"] == "not_started"


def test_disabled_audit_creates_no_companion(
    tmp_path, ingest_config, monkeypatch, capsys,
):
    monkeypatch.delenv("SCHAIN_AUDIT_TRACE", raising=False)
    monkeypatch.delenv("SCHAIN_AUDIT_FULL_CONTENT", raising=False)
    config_path = _config_file(tmp_path, ingest_config)
    assert _run_cli_with_stub(
        monkeypatch, config_path, _corpus(tmp_path), StubLLM(),
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    graph_path = Path(summary["graph_export"])
    assert graph_path.is_file()
    assert not audit_path_for_graph(graph_path).exists()
