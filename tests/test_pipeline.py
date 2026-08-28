"""The per-document pipeline and its resume guarantees."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from schain import state
from schain.documents import load_document
from schain.pipeline import clear_generated_data, process_document

from .conftest import StubLLM


class Prompt2FailsLLM(StubLLM):
    """Prompt 1 succeeds, Prompt 2 raises."""

    def invoke(self, input, message_history=None, system_instruction=None):
        raise RuntimeError("disambiguation endpoint down")


class FakeDriver:
    """Accepts writes and records them."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def execute_query(self, query, **kwargs):
        if self.fail:
            raise RuntimeError("neo4j unavailable")
        self.calls.append(query)
        return None

    def close(self):
        pass


def run(document_file, config, connection, ontology, llm, driver=None):
    return process_document(
        load_document(document_file), config, connection, llm, ontology, driver
    )


def test_document_runs_end_to_end_without_neo4j(
    document_file, ingest_config, connection, ontology, stub_llm
):
    summary = run(document_file, ingest_config, connection, ontology, stub_llm)

    assert summary["facts"] == 3
    assert summary["status"] == state.RESOLVED   # no driver, so it stops before Neo4j
    assert stub_llm.extraction_calls == 1 and stub_llm.disambiguation_calls >= 1

    facts = pd.read_parquet(ingest_config["storage"]["extracted_facts"])
    triples = set(zip(facts.subject_id, facts.relation, facts.object_id))
    assert triples == {
        ("company:contemporary-amperex-technology", "suppliesTo", "company:bmw"),
        ("company:contemporary-amperex-technology", "locatedIn", "location:germany"),
        ("company:bmw", "produces", "product:electric-vehicles"),
    }


def test_facts_carry_full_provenance(
    document_file, ingest_config, connection, ontology, stub_llm
):
    process_document(
        load_document(document_file, source_url="https://news.test/catl-bmw?utm_source=x"),
        ingest_config, connection, stub_llm, ontology,
    )
    facts = pd.read_parquet(ingest_config["storage"]["extracted_facts"])
    row = facts.iloc[0]
    assert row.source_url == "https://news.test/catl-bmw"
    assert row.model == "Qwen/Qwen3.8-27B"
    assert row.chunk_index == 0
    assert row.evidence and row.doc_id and row.extracted_at
    assert facts.evidence.str.len().gt(0).all()


def test_prompt1_output_is_persisted_before_anything_else(
    document_file, ingest_config, connection, ontology, stub_llm
):
    run(document_file, ingest_config, connection, ontology, stub_llm)
    saved = list(Path(ingest_config["storage"]["extractions"]).glob("*.json"))
    assert len(saved) == 1
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["graph"]["nodes"] and payload["chunk_index"]


def test_prompt1_is_not_repeated_when_prompt2_fails(
    document_file, ingest_config, connection, ontology
):
    """The hard rule: a later failure must never force Prompt 1 to run again."""
    with pytest.raises(RuntimeError, match="disambiguation endpoint down"):
        run(document_file, ingest_config, connection, ontology, Prompt2FailsLLM())

    document = load_document(document_file)
    assert state.document_status(connection, document["doc_id"]) == state.EXTRACTED

    recovered = StubLLM()
    summary = run(document_file, ingest_config, connection, ontology, recovered)
    assert recovered.extraction_calls == 0          # reused from disk
    assert recovered.disambiguation_calls >= 1      # only Prompt 2 re-ran
    assert summary["status"] == state.RESOLVED


def test_neither_prompt_repeats_when_the_graph_write_fails(
    document_file, ingest_config, connection, ontology, stub_llm
):
    run(document_file, ingest_config, connection, ontology, stub_llm)
    document = load_document(document_file)
    assert state.document_status(connection, document["doc_id"]) == state.RESOLVED

    with pytest.raises(RuntimeError, match="neo4j unavailable"):
        run(document_file, ingest_config, connection, ontology, StubLLM(), FakeDriver(fail=True))
    # still RESOLVED: the facts are safe, only the write is outstanding
    assert state.document_status(connection, document["doc_id"]) == state.RESOLVED

    fresh = StubLLM()
    summary = run(document_file, ingest_config, connection, ontology, fresh, FakeDriver())
    assert fresh.prompts == []                      # no LLM call at all
    assert summary["status"] == state.COMMITTED


def test_committed_document_is_skipped(
    document_file, ingest_config, connection, ontology, stub_llm
):
    run(document_file, ingest_config, connection, ontology, stub_llm, FakeDriver())
    again = StubLLM()
    summary = run(document_file, ingest_config, connection, ontology, again, FakeDriver())
    assert summary["skipped"] == "already committed"
    assert again.prompts == []


def test_the_same_content_under_another_filename_is_not_re_extracted(
    document_file, ingest_config, connection, ontology, stub_llm, tmp_path
):
    """Identical text hashes to the same doc_id, so it is recognized, not reprocessed."""
    run(document_file, ingest_config, connection, ontology, stub_llm)
    facts_before = len(pd.read_parquet(ingest_config["storage"]["extracted_facts"]))

    copy = tmp_path / "copy_of_the_same_article.txt"
    copy.write_text(document_file.read_text(encoding="utf-8"), encoding="utf-8")
    again = StubLLM()
    run(copy, ingest_config, connection, ontology, again)

    assert again.prompts == []   # neither prompt ran again
    assert len(pd.read_parquet(ingest_config["storage"]["extracted_facts"])) == facts_before


def test_the_same_url_with_changed_text_is_skipped_as_duplicate(
    document_file, ingest_config, connection, ontology, stub_llm, tmp_path
):
    """URL-level deduplication: one article revisited later is not ingested twice."""
    url = "https://news.test/catl-and-bmw"
    process_document(
        load_document(document_file, source_url=url),
        ingest_config, connection, stub_llm, ontology,
    )

    revised = tmp_path / "revised.txt"
    revised.write_text(document_file.read_text(encoding="utf-8") + "\nUpdated.\n", encoding="utf-8")
    again = StubLLM()
    summary = process_document(
        load_document(revised, source_url=url),
        ingest_config, connection, again, ontology,
    )
    assert summary["skipped"] == "duplicate"
    assert again.prompts == []


def test_extraction_failure_is_recorded_as_failed(
    document_file, ingest_config, connection, ontology
):
    from .conftest import FailingLLM

    with pytest.raises(RuntimeError):
        run(document_file, ingest_config, connection, ontology, FailingLLM())
    document = load_document(document_file)
    assert state.document_status(connection, document["doc_id"]) == state.FAILED
    row = state.documents_with_status(connection, [state.FAILED])[0]
    assert row["attempts"] == 1 and row["error"]


def test_graph_write_merges_nodes_and_relationships(
    document_file, ingest_config, connection, ontology, stub_llm
):
    driver = FakeDriver()
    summary = run(document_file, ingest_config, connection, ontology, stub_llm, driver)
    assert summary["status"] == state.COMMITTED
    assert summary["nodes_merged"] == 4          # CATL, BMW, Germany, electric vehicles
    assert summary["relationships_merged"] == 3
    assert all("MERGE" in query for query in driver.calls)


def test_entity_resolution_is_persisted_to_parquet_and_state(
    document_file, ingest_config, connection, ontology, stub_llm
):
    run(document_file, ingest_config, connection, ontology, stub_llm)
    frame = pd.read_parquet(ingest_config["storage"]["entity_resolution"])
    assert {"doc_id", "node_id", "label", "canonical_id", "canonical_name",
            "aliases", "mention", "method"} <= set(frame.columns)
    bmw = frame[frame.canonical_id == "company:bmw"]
    assert set(bmw.iloc[0]["aliases"]) == {"BMW", "BMW AG"}
    assert state.load_resolution_cache(connection)


def test_restart_clears_state_and_generated_data(
    document_file, ingest_config, connection, ontology, stub_llm
):
    run(document_file, ingest_config, connection, ontology, stub_llm)
    connection.close()
    removed = clear_generated_data(ingest_config)
    assert removed
    assert not Path(ingest_config["storage"]["extracted_facts"]).exists()
    assert not list(Path(ingest_config["storage"]["extractions"]).glob("*.json"))

    fresh_connection = state.connect(Path(ingest_config["storage"]["state_db"]))
    assert state.document_counts(fresh_connection) == {}
    fresh_connection.close()
