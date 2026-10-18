"""Regression tests for partial extraction and durable resume, using local stubs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from neo4j_graphrag.llm import LLMInterface, LLMResponse

from schain import state
from schain.checkpoint import atomic_write_json
from schain.documents import load_document, make_document
from schain.kg.extract import extract_graph
from schain.pipeline import clear_generated_data, process_document

from .conftest import StubLLM
from .test_extract import VALID
from .test_pipeline import Prompt2FailsLLM


class ChunkLLM(LLMInterface):
    def __init__(self, fail_bravo: bool = False, transient_once: bool = False):
        super().__init__(model_name="stub")
        self.fail_bravo = fail_bravo
        self.transient_once = transient_once
        self.inputs: list[str] = []

    def invoke(self, input, message_history=None, system_instruction=None):
        return LLMResponse(content=VALID)

    async def ainvoke(self, input, message_history=None, system_instruction=None):
        text = input.rsplit("Input text:", 1)[-1].strip()
        self.inputs.append(text)
        if self.fail_bravo and "bravo" in text:
            raise RuntimeError("deliberate chunk failure")
        if self.transient_once and "bravo" in text:
            self.transient_once = False
            raise RuntimeError("connection temporarily unavailable")
        return LLMResponse(content=VALID)


def chunk_document():
    return make_document("alpha " * 20 + "bravo " * 20 + "charlie " * 20)


def test_resume_reuses_successful_chunks_and_preserves_their_ids(tmp_path, ontology):
    document = chunk_document()
    options = dict(chunk_size=120, chunk_overlap=0, on_error="RAISE", checkpoint_dir=tmp_path)
    failing = ChunkLLM(fail_bravo=True)
    with pytest.raises(RuntimeError, match="deliberate chunk failure"):
        extract_graph(failing, document, ontology, **options)
    completed = list(tmp_path.glob("*.json"))
    assert len(completed) == len(failing.inputs) - 1

    resumed = ChunkLLM()
    graph, index = extract_graph(resumed, document, ontology, **options)
    assert len(resumed.inputs) == 1 and "bravo" in resumed.inputs[0]
    assert len(graph.nodes) == len(index) * 2
    assert {node.id.split(":", 1)[0] for node in graph.nodes} == set(index)

    cached = ChunkLLM()
    again, again_index = extract_graph(cached, document, ontology, **options)
    assert cached.inputs == []
    assert again.model_dump() == graph.model_dump() and again_index == index


def test_transient_error_retries_only_its_own_chunk(tmp_path, ontology):
    llm = ChunkLLM(transient_once=True)
    _, index = extract_graph(
        llm, chunk_document(), ontology, chunk_size=120, chunk_overlap=0,
        on_error="RAISE", checkpoint_dir=tmp_path,
    )
    assert len(llm.inputs) == len(index) + 1
    assert sum("bravo" in text for text in llm.inputs) == 2
    assert sum("alpha" in text for text in llm.inputs) == 1


def test_chunk_cache_invalidates_when_model_parameters_change(tmp_path, ontology):
    document = chunk_document()
    llm = ChunkLLM()
    options = dict(chunk_size=120, chunk_overlap=0, checkpoint_dir=tmp_path)
    _, index = extract_graph(llm, document, ontology, **options)
    llm.inputs.clear()
    llm.model_params["temperature"] = 0.5
    extract_graph(llm, document, ontology, **options)
    assert len(llm.inputs) == len(index)


def test_damaged_chunk_cache_is_recomputed(tmp_path, ontology):
    llm = ChunkLLM()
    options = dict(chunk_size=10000, chunk_overlap=0, checkpoint_dir=tmp_path)
    extract_graph(llm, chunk_document(), ontology, **options)
    next(tmp_path.glob("*.json")).write_text('{"nodes":', encoding="utf-8")
    llm.inputs.clear()
    graph, _ = extract_graph(llm, chunk_document(), ontology, **options)
    assert len(llm.inputs) == 1 and graph.nodes


def test_interrupted_json_write_keeps_previous_checkpoint(tmp_path, monkeypatch):
    target = tmp_path / "checkpoint.json"
    atomic_write_json(target, {"complete": True})

    def interrupted(payload, handle, **kwargs):
        handle.write('{"incomplete":')
        raise OSError("disk write failed")

    monkeypatch.setattr("schain.checkpoint.json.dump", interrupted)
    with pytest.raises(OSError, match="disk write failed"):
        atomic_write_json(target, {"complete": False})
    assert json.loads(target.read_text()) == {"complete": True}
    assert sorted(path.name for path in tmp_path.iterdir()) == [target.name]


def test_resume_repairs_source_persistence_after_registration(
    document_file, ingest_config, connection, ontology, monkeypatch,
):
    document = load_document(document_file, source_url="https://original.test/article")

    def broken_write(*args, **kwargs):
        raise OSError("normalized document write failed")

    with monkeypatch.context() as patch:
        patch.setattr("schain.pipeline.upsert_normalized_parquet", broken_write)
        with pytest.raises(OSError, match="normalized document write failed"):
            process_document(document, ingest_config, connection, StubLLM(), ontology)
    assert state.document_status(connection, document["doc_id"]) == state.FAILED

    rediscovered = load_document(document_file, source_url="https://mirror.test/article")
    process_document(rediscovered, ingest_config, connection, StubLLM(), ontology)
    normalized = pd.read_parquet(ingest_config["storage"]["normalized_documents"])
    facts = pd.read_parquet(ingest_config["storage"]["extracted_facts"])
    assert normalized.source_url.tolist() == ["https://original.test/article"]
    assert facts.source_url.eq("https://original.test/article").all()
    raw = Path(ingest_config["storage"]["raw_documents"]).read_text().splitlines()
    assert len(raw) == 1


def test_prompt2_resume_keeps_the_original_source_attribution(
    document_file, ingest_config, connection, ontology,
):
    original = load_document(document_file, source_url="https://original.test/article")
    with pytest.raises(RuntimeError, match="disambiguation endpoint down"):
        process_document(original, ingest_config, connection, Prompt2FailsLLM(), ontology)
    rediscovered = load_document(document_file, source_url="https://mirror.test/article")
    llm = StubLLM()
    process_document(rediscovered, ingest_config, connection, llm, ontology)
    facts = pd.read_parquet(ingest_config["storage"]["extracted_facts"])
    assert llm.extraction_calls == 0
    assert facts.source_url.eq("https://original.test/article").all()


def test_resolved_resume_restores_saved_alias_groups_before_commit(
    document_file, ingest_config, connection, ontology, monkeypatch,
):
    document = load_document(document_file)
    process_document(document, ingest_config, connection, StubLLM(), ontology)
    captured = {}

    def capture(driver, config, facts, ontology, resolution):
        captured.update(resolution)
        return 0, 0

    monkeypatch.setattr("schain.pipeline.commit_facts", capture)
    llm = StubLLM()
    summary = process_document(document, ingest_config, connection, llm, ontology, object())
    assert summary["status"] == state.COMMITTED and llm.prompts == []
    assert any(set(row["aliases"]) == {"BMW", "BMW AG"} for row in captured.values())


def test_restart_removes_chunk_checkpoints(
    document_file, ingest_config, connection, ontology,
):
    process_document(load_document(document_file), ingest_config, connection, StubLLM(), ontology)
    extraction_dir = Path(ingest_config["storage"]["extractions"])
    assert list(extraction_dir.glob("*.chunks/*.json"))
    connection.close()
    clear_generated_data(ingest_config)
    assert not list(extraction_dir.glob("*.chunks"))


def test_resolved_no_driver_resume_skips_but_can_commit_later(
    document_file, ingest_config, connection, ontology,
):
    document = load_document(document_file)
    first = process_document(document, ingest_config, connection, StubLLM(), ontology)
    assert first['status'] == state.RESOLVED
    resumed = StubLLM()
    skipped = process_document(document, ingest_config, connection, resumed, ontology)
    assert skipped['skipped'] == 'already resolved'
    assert skipped['facts'] == 0 and resumed.prompts == []
    from .test_pipeline import FakeDriver
    committed = process_document(document, ingest_config, connection, resumed, ontology, FakeDriver())
    assert committed['status'] == state.COMMITTED and resumed.prompts == []


def test_changed_url_content_resumes_unfinished_original(
    document_file, ingest_config, connection, ontology,
):
    original = load_document(document_file, source_url='https://source.test/article')
    with pytest.raises(RuntimeError, match='disambiguation endpoint down'):
        process_document(original, ingest_config, connection, Prompt2FailsLLM(), ontology)
    changed = make_document('Changed body contains no supply evidence.', source_url=original['source_url'])
    assert changed['doc_id'] != original['doc_id']
    resumed = StubLLM()
    summary = process_document(changed, ingest_config, connection, resumed, ontology)
    assert summary['doc_id'] == original['doc_id']
    assert summary['status'] == state.RESOLVED and summary['skipped'] is None
    assert resumed.extraction_calls == 0
    facts = pd.read_parquet(ingest_config['storage']['extracted_facts'])
    assert facts.doc_id.eq(original['doc_id']).all()
    assert facts.evidence.map(lambda quote: quote in original['text']).all()


def test_document_metrics_count_calls_and_persist_partial_coverage(
    document_file, ingest_config, connection, ontology,
):
    ingest_config['ingest']['chunk_size'] = 150
    ingest_config['ingest']['chunk_overlap'] = 0
    ingest_config['ingest']['max_chunks_per_document'] = 1
    document = load_document(document_file)
    llm = StubLLM()
    result = process_document(document, ingest_config, connection, llm, ontology)
    assert result['llm_calls'] == len(llm.prompts)
    assert result['extraction_calls'] == llm.extraction_calls == 1
    assert result['extraction_chunks_available'] > 1
    assert result['extraction_coverage'] == 'partial'
    assert result['extraction_chunk_indices'] == [0]
    assert result['candidate_facts'] >= result['accepted_facts']
    assert result['elapsed_seconds'] >= 0
    saved = json.loads((Path(ingest_config['storage']['extractions']) / 'metrics' /
                        f"{document['doc_id']}.json").read_text())
    assert saved['total_llm_calls'] == result['llm_calls']
    normalized = pd.read_parquet(ingest_config['storage']['normalized_documents'])
    assert normalized.iloc[0].text == document['text']


def test_atomic_parquet_failure_keeps_previous_artifact(tmp_path):
    from schain.checkpoint import atomic_write_parquet
    path = tmp_path / 'facts.parquet'
    atomic_write_parquet(pd.DataFrame({'fact_id': ['preserved']}), path)

    class InterruptedFrame:
        def to_parquet(self, target, **kwargs):
            Path(target).write_bytes(b'PARTIAL')
            raise OSError('interrupted')

    with pytest.raises(OSError, match='interrupted'):
        atomic_write_parquet(InterruptedFrame(), path)
    assert pd.read_parquet(path).fact_id.tolist() == ['preserved']
    assert [p.name for p in tmp_path.iterdir()] == ['facts.parquet']


def test_explicit_chunk_cap_prioritizes_upstream_and_preserves_indices(tmp_path, ontology):
    text = 'BorgWarner supplies Stellantis.\n' + 'padding ' * 20 + '\nAcme supplies BorgWarner.'
    metrics = {}
    llm = ChunkLLM()
    graph, index = extract_graph(
        llm, make_document(text), ontology, chunk_size=80, chunk_overlap=0,
        max_chunks_per_document=1, origin_name='BorgWarner', metrics=metrics,
        checkpoint_dir=tmp_path,
    )
    assert len(llm.inputs) == 1 and 'Acme supplies BorgWarner' in llm.inputs[0]
    assert metrics['extraction_coverage'] == 'partial'
    assert metrics['extraction_chunk_selection'] == 'upstream_priority_then_original_order'
    assert metrics['extraction_chunk_indices'][0] > 0
    assert set(index.values()) == set(metrics['extraction_chunk_indices'])


def test_extraction_retry_count_is_explicit_and_bounded(tmp_path, ontology):
    from .conftest import FailingLLM
    metrics = {}
    with pytest.raises(RuntimeError, match='model unavailable'):
        extract_graph(FailingLLM(), make_document('One short article.'), ontology,
                      max_retries=1, metrics=metrics, checkpoint_dir=tmp_path)
    assert metrics['extraction_calls'] == 2
    assert metrics['extraction_retries'] == 1


def test_restart_removes_runtime_snapshot(document_file, ingest_config, connection, ontology):
    from schain.run_config import snapshot_path
    target = snapshot_path(ingest_config)
    atomic_write_json(target, {'stale': True})
    connection.close()
    clear_generated_data(ingest_config)
    assert not target.exists()


def test_validator_service_failure_retries_without_reextracting(
    document_file, ingest_config, connection, ontology,
):
    class ValidatorDown(StubLLM):
        def invoke(self, input, *args, **kwargs):
            if 'For each candidate, decide' in input:
                raise RuntimeError('validator endpoint timeout')
            return super().invoke(input, *args, **kwargs)

    document = load_document(document_file)
    with pytest.raises(RuntimeError, match='document remains retryable'):
        process_document(document, ingest_config, connection, ValidatorDown(), ontology)
    assert state.document_status(connection, document['doc_id']) == state.EXTRACTED
    assert not Path(ingest_config['storage']['extracted_facts']).exists()
    recovered = StubLLM()
    result = process_document(document, ingest_config, connection, recovered, ontology)
    assert result['status'] == state.RESOLVED and result['accepted_facts'] > 0
    assert recovered.extraction_calls == 0


def test_runtime_deadline_prevents_new_model_calls_and_keeps_work_retryable(
    document_file, ingest_config, connection, ontology,
):
    import time
    from schain.llm.provider import RuntimeDeadlineExceeded
    document = load_document(document_file)
    ingest_config['_runtime_deadline_monotonic'] = time.monotonic() - 1
    llm = StubLLM()
    with pytest.raises(RuntimeDeadlineExceeded):
        process_document(document, ingest_config, connection, llm, ontology)
    assert llm.prompts == []
    assert state.document_status(connection, document['doc_id']) == state.FAILED
    saved = json.loads((Path(ingest_config['storage']['extractions']) / 'metrics' /
                        f"{document['doc_id']}.json").read_text())
    assert saved['last_attempt']['stop_reason'] == 'runtime_deadline'
    assert saved['last_attempt']['extraction_retries'] == 0
    ingest_config.pop('_runtime_deadline_monotonic')
    resumed = process_document(document, ingest_config, connection, llm, ontology)
    assert resumed['status'] == state.RESOLVED
