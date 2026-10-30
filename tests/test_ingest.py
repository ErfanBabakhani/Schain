"""Multi-document INGEST and its resume behaviour."""

from __future__ import annotations

import json

import pandas as pd

from schain import state
from schain.ingest import run_ingest

from .conftest import DOCUMENT_TEXT, FailingLLM, StubLLM
from .test_pipeline import FakeDriver

SECOND_DOCUMENT = (
    "Ganfeng Lithium Co Ltd supplies lithium to CATL under a long-term agreement.\n"
    "Ganfeng Lithium operates mining assets located in Argentina.\n"
)


def corpus(tmp_path, *texts):
    directory = tmp_path / "corpus"
    directory.mkdir(exist_ok=True)
    for index, text in enumerate(texts):
        (directory / f"doc{index}.txt").write_text(text, encoding="utf-8")
    return directory


def test_directory_of_documents_is_ingested(tmp_path, ingest_config, connection, ontology):
    directory = corpus(tmp_path, DOCUMENT_TEXT, SECOND_DOCUMENT)
    summary = run_ingest(ingest_config, connection, StubLLM(), ontology, None, directory)

    assert summary["files"] == 2
    assert summary["documents_processed"] == 2
    assert summary["documents_failed"] == 0
    assert summary["documents"] == {state.RESOLVED: 2}


def test_a_single_file_can_be_ingested(document_file, ingest_config, connection, ontology):
    summary = run_ingest(ingest_config, connection, StubLLM(), ontology, None, document_file)
    assert summary["files"] == 1 and summary["documents_processed"] == 1


def test_resumed_run_makes_no_llm_calls(tmp_path, ingest_config, connection, ontology):
    directory = corpus(tmp_path, DOCUMENT_TEXT, SECOND_DOCUMENT)
    run_ingest(ingest_config, connection, StubLLM(), ontology, FakeDriver(), directory)

    second = StubLLM()
    summary = run_ingest(ingest_config, connection, second, ontology, FakeDriver(), directory)
    assert summary["documents_skipped"] == 2
    assert second.prompts == []
    assert summary["documents"] == {state.COMMITTED: 2}


def test_one_bad_document_does_not_stop_the_run(tmp_path, ingest_config, connection, ontology):
    directory = corpus(tmp_path, DOCUMENT_TEXT)
    (directory / "empty.txt").write_text("   \n", encoding="utf-8")

    summary = run_ingest(ingest_config, connection, StubLLM(), ontology, None, directory)
    assert summary["files"] == 2
    assert summary["documents_processed"] == 1
    assert summary["documents_failed"] == 1


def test_llm_failure_is_counted_and_recorded(tmp_path, ingest_config, connection, ontology):
    directory = corpus(tmp_path, DOCUMENT_TEXT)
    summary = run_ingest(ingest_config, connection, FailingLLM(), ontology, None, directory)
    assert summary["documents_failed"] == 1
    assert summary["documents"] == {state.FAILED: 1}


def test_missing_input_directory_is_reported(ingest_config, connection, ontology, tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        run_ingest(ingest_config, connection, StubLLM(), ontology, None, tmp_path / "nope")


def test_facts_accumulate_across_documents(tmp_path, ingest_config, connection, ontology):
    class SourceAwareLLM(StubLLM):
        def _answer(self, prompt):
            # An extraction stub must return evidence from the document it receives.
            if "Input text:" in prompt and SECOND_DOCUMENT.strip() in prompt:
                self.prompts.append(prompt)
                return json.dumps({
                    "nodes": [
                        {"id": "0", "label": "Company", "properties": {"name": "Ganfeng Lithium Co Ltd"}},
                        {"id": "1", "label": "Company", "properties": {"name": "CATL"}},
                    ],
                    "relationships": [{"type": "suppliesTo", "start_node_id": "0", "end_node_id": "1",
                        "properties": {"evidence": SECOND_DOCUMENT.splitlines()[0]}}],
                })
            return super()._answer(prompt)

    directory = corpus(tmp_path, DOCUMENT_TEXT, SECOND_DOCUMENT)
    run_ingest(ingest_config, connection, SourceAwareLLM(), ontology, None, directory)
    facts = pd.read_parquet(ingest_config["storage"]["extracted_facts"])
    assert facts.doc_id.nunique() == 2
    # fact ids are deterministic, so nothing is duplicated
    assert facts.fact_id.is_unique
