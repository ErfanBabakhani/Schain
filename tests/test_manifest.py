"""The retrieval audit manifest: one row per candidate, and a summary that agrees with it."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from schain import state
from schain.expansion.loop import run_expand
from schain.sources import manifest as mf
from schain.sources import retrieve
from schain.sources.base import SearchResult

from .conftest import DOCUMENT_TEXT, StubLLM
from .test_pipeline import FakeDriver

ARTICLE = (
    "BMW has named CATL as a battery supplier for its European electric vehicle plants "
    "under a multi-year sourcing agreement signed this year. "
) * 3


def result(url="https://news.test/a", provider="gdelt", query='"BMW" suppliers',
           title="BMW names supplier", snippet="A supplier was named.", text=None):
    return SearchResult(provider=provider, query=query, title=title, url=url,
                        snippet=snippet, text=text)


def manifest_at(tmp_path) -> mf.RetrievalManifest:
    return mf.RetrievalManifest(tmp_path / "retrieval_manifest_test.csv", "test-run")


def rows_from(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- the row itself -------------------------------------------------------------------


def test_every_documented_column_is_written(tmp_path):
    manifest = manifest_at(tmp_path)
    manifest.record_candidate("BMW", 0, "gdelt", "NEWS", '"BMW" suppliers', 1,
                              result(text=ARTICLE))
    path = manifest.flush()

    rows = rows_from(path)
    assert len(rows) == 1
    assert set(rows[0]) == set(mf.FIELDS)
    row = rows[0]
    assert row["run_id"] == "test-run"
    assert row["entity"] == "BMW" and row["tier"] == "0"
    assert row["provider"] == "gdelt" and row["source_type"] == "NEWS"
    assert row["query"] == '"BMW" suppliers' and row["result_rank"] == "1"
    assert row["url"] == "https://news.test/a" and row["domain"] == "news.test"
    assert row["retrieved_at"]


def test_content_lengths_and_word_count_are_recorded(tmp_path):
    manifest = manifest_at(tmp_path)
    manifest.record_candidate("BMW", 0, "gdelt", "NEWS", "q", 1, result(text=ARTICLE))
    row = rows_from(manifest.flush())[0]
    assert int(row["content_length_chars"]) == len(ARTICLE)
    assert int(row["content_length_words"]) == len(ARTICLE.split())
    assert int(row["snippet_length_chars"]) == len("A supplier was named.")
    assert row["content_mode"] == mf.FULL_TEXT


def test_a_candidate_with_no_body_is_recorded_as_snippet_only(tmp_path):
    manifest = manifest_at(tmp_path)
    candidate = result(provider="google_news", url="https://news.google.com/rss/x")
    key = manifest.record_candidate("BMW", 0, "google_news", "NEWS", "q", 1, candidate)
    manifest.update(key, fetch_status=mf.FAILED, http_status=403,
                    content_mode=mf.SNIPPET_ONLY, error="HTTP_403")
    row = rows_from(manifest.flush())[0]
    assert row["fetch_status"] == "FAILED"
    assert row["http_status"] == "403"
    assert row["content_mode"] == "SNIPPET_ONLY"
    assert row["error"] == "HTTP_403"
    assert int(row["content_length_chars"]) == 0


def test_defaults_are_the_not_yet_known_states(tmp_path):
    manifest = manifest_at(tmp_path)
    manifest.record_candidate("BMW", 0, "gdelt", "NEWS", "q", 1, result())
    row = rows_from(manifest.flush())[0]
    assert row["fetch_status"] == mf.NOT_ATTEMPTED
    assert row["relevance_status"] == mf.NOT_CHECKED
    assert row["llm_status"] == mf.NOT_PROCESSED
    assert row["dedup_status"] == mf.UNIQUE


def test_recording_the_same_candidate_again_updates_rather_than_duplicates(tmp_path):
    """Resume re-records reused results; the audit trail must not grow duplicates."""
    manifest = manifest_at(tmp_path)
    for _ in range(3):
        manifest.record_candidate("BMW", 0, "gdelt", "NEWS", "q", 1, result())
    manifest.update_by_url("https://news.test/a", llm_status=mf.PROCESSED)
    for _ in range(2):
        manifest.record_candidate("BMW", 0, "gdelt", "NEWS", "q", 1, result())

    rows = rows_from(manifest.flush())
    assert len(rows) == 1
    assert rows[0]["llm_status"] == mf.PROCESSED   # earlier progress is preserved


def test_the_same_url_from_two_providers_is_two_rows(tmp_path):
    """Each provider's candidate is audited separately, even for one URL."""
    manifest = manifest_at(tmp_path)
    manifest.record_candidate("BMW", 0, "gdelt", "NEWS", "q", 1, result(provider="gdelt"))
    manifest.record_candidate("BMW", 0, "general_web", "GENERAL_WEB", "q", 1,
                              result(provider="general_web"))
    assert len(rows_from(manifest.flush())) == 2


def test_full_article_text_is_not_written_into_the_manifest(tmp_path):
    manifest = manifest_at(tmp_path)
    manifest.record_candidate("BMW", 0, "gdelt", "NEWS", "q", 1, result(text=ARTICLE))
    assert "multi-year sourcing agreement" not in manifest.flush().read_text(encoding="utf-8")


# --- the summary ----------------------------------------------------------------------


def test_summary_counts_agree_with_the_rows(tmp_path):
    manifest = manifest_at(tmp_path)
    a = manifest.record_candidate("BMW", 0, "gdelt", "NEWS", "q", 1,
                                  result(url="https://a.test/1", text=ARTICLE))
    b = manifest.record_candidate("BMW", 0, "google_news", "NEWS", "q", 2,
                                  result(url="https://b.test/2", provider="google_news"))
    c = manifest.record_candidate("BMW", 1, "general_web", "GENERAL_WEB", "q", 3,
                                  result(url="https://a.test/1", provider="general_web"))
    manifest.update(a, fetch_status=mf.SUCCESS, content_mode=mf.FULL_TEXT,
                    relevance_status=mf.RELEVANT, llm_status=mf.PROCESSED)
    manifest.update(b, fetch_status=mf.FAILED, content_mode=mf.SNIPPET_ONLY,
                    relevance_status=mf.IRRELEVANT)
    manifest.update(c, dedup_status=mf.DUPLICATE_URL, relevance_status=mf.RELEVANT,
                    llm_status=mf.LLM_FAILED)

    summary = manifest.summary()
    rows = rows_from(manifest.flush())
    assert summary["raw_candidates"] == len(rows) == 3
    assert summary["unique_urls"] == 2
    assert summary["duplicate_urls"] == 1
    assert summary["successful_fetches"] == 1
    assert summary["failed_fetches"] == 1
    assert summary["full_text_documents"] == 1
    assert summary["snippet_only_documents"] == 1
    assert summary["relevant_documents"] == 2
    assert summary["irrelevant_documents"] == 1
    assert summary["sent_to_prompt1"] == 1      # one unique URL, two audit rows
    assert summary["prompt1_failures"] == 1
    assert summary["by_provider"] == {"gdelt": 1, "google_news": 1, "general_web": 1}
    assert summary["by_tier"] == {"0": 2, "1": 1}


def test_summary_of_an_empty_run_is_all_zeroes(tmp_path):
    summary = manifest_at(tmp_path).summary()
    assert summary["raw_candidates"] == 0 and summary["sent_to_prompt1"] == 0


# --- through the retrieval layer ------------------------------------------------------


@pytest.fixture
def one_provider_config(expand_config):
    expand_config["expansion"]["sources"] = {"gdelt": {"enabled": True, "fetch_full_text": True}}
    expand_config["expansion"]["search"] = {
        "defaults": {"max_results": 5, "providers": ["gdelt"]},
        "tiers": {0: {"queries": [{"query": '"{entity}" suppliers'},
                                  {"query": '"{entity}" sourcing'}]}},
    }
    expand_config["expansion"]["relevance"] = {
        "min_chars": 50, "require_entity_mention": True, "keywords": ["supplier"],
    }
    return expand_config


def test_retrieval_records_fetch_dedup_and_relevance(one_provider_config, connection,
                                                     tmp_path, monkeypatch):
    async def provider(client, query, max_results, options):
        # Both queries surface the same article; the second is a duplicate URL.
        candidates = [result(url="https://a.test/1", query=query, text=None)]
        if "suppliers" in query:
            candidates.append(
                result(url="https://b.test/2", query=query, title="Museum reopens",
                       snippet="Nothing to do with sourcing here at all, just a museum.")
            )
        return candidates

    async def body(client, url):
        if url == "https://a.test/1":
            return ARTICLE, 200, None
        return None, 404, "HTTP_404"

    monkeypatch.setitem(retrieve.PROVIDERS, "gdelt", provider)
    monkeypatch.setattr("schain.sources.retrieve.fetch_article_text", body)

    manifest = manifest_at(tmp_path)
    kept = retrieve.retrieve_for_entity(
        one_provider_config, "BMW", "company:bmw", 0, connection, manifest
    )
    rows = {(row["url"], row["dedup_status"]): row for row in rows_from(manifest.flush())}

    assert [r.url for r in kept] == ["https://a.test/1"]
    fetched = rows[("https://a.test/1", mf.UNIQUE)]
    assert fetched["fetch_status"] == mf.SUCCESS and fetched["http_status"] == "200"
    assert fetched["content_mode"] == mf.FULL_TEXT
    assert int(fetched["content_length_words"]) > 0
    assert fetched["relevance_status"] == mf.RELEVANT

    assert ("https://a.test/1", mf.DUPLICATE_URL) in rows
    irrelevant = rows[("https://b.test/2", mf.UNIQUE)]
    assert irrelevant["relevance_status"] == mf.IRRELEVANT
    assert irrelevant["error"] == "http_404"


# --- through a whole EXPAND run -------------------------------------------------------


def test_expand_writes_a_manifest_and_a_summary(expand_config, connection, ontology,
                                                tmp_path, monkeypatch, caplog):
    import logging

    expand_config["seeds"] = {"oem": ["BMW"]}
    expand_config["expansion"]["max_tier"] = 1

    def fake_retrieval(config, entity_name, entity_key, tier, connection, manifest=None, **_kwargs):
        if tier > 0:
            return []
        candidates = [
            result(url="https://news.test/used", text=DOCUMENT_TEXT),
            result(url="https://news.test/dup", text=DOCUMENT_TEXT),   # same content
        ]
        for rank, candidate in enumerate(candidates, start=1):
            manifest.record_candidate(entity_name, tier, candidate.provider, "NEWS",
                                      candidate.query, rank, candidate)
            manifest.update_by_url(candidate.url, relevance_status=mf.RELEVANT,
                                   fetch_status=mf.SUCCESS)
        return candidates

    monkeypatch.setattr("schain.expansion.loop.retrieve_for_entity", fake_retrieval)
    monkeypatch.setattr("schain.expansion.loop.REPO_ROOT", tmp_path)
    caplog.set_level(logging.INFO, logger="schain.sources.manifest")

    summary = run_expand(expand_config, connection, StubLLM(), ontology, FakeDriver())

    manifest_path = Path(summary["retrieval_manifest"])
    assert manifest_path.exists()
    assert manifest_path.parent == tmp_path / "logs"
    assert manifest_path.name.startswith("retrieval_manifest_")

    rows = {row["url"]: row for row in rows_from(manifest_path)}
    assert rows["https://news.test/used"]["llm_status"] == mf.PROCESSED
    # the second document has identical content, so it is skipped, not re-extracted
    assert rows["https://news.test/dup"]["llm_status"] == mf.SKIPPED
    assert rows["https://news.test/dup"]["dedup_status"] == mf.DUPLICATE_CONTENT

    assert summary["retrieval"]["raw_candidates"] == 2
    assert summary["retrieval"]["sent_to_prompt1"] == 1
    assert "retrieval summary" in caplog.text


def test_prompt1_failure_is_distinguished_from_a_later_failure(
    expand_config, connection, ontology, tmp_path, monkeypatch
):
    """A Neo4j failure after a good extraction must not be reported as a Prompt 1 failure."""
    from .conftest import FailingLLM

    expand_config["seeds"] = {"oem": ["BMW"]}
    expand_config["expansion"]["max_tier"] = 1
    expand_config["runtime"]["max_consecutive_failures"] = 10

    def fake_retrieval(config, entity_name, entity_key, tier, connection, manifest=None, **_kwargs):
        if tier > 0:
            return []
        candidate = result(url="https://news.test/x", text=DOCUMENT_TEXT, query="q")
        manifest.record_candidate(entity_name, tier, "gdelt", "NEWS", "q", 1, candidate)
        return [candidate]

    monkeypatch.setattr("schain.expansion.loop.retrieve_for_entity", fake_retrieval)
    monkeypatch.setattr("schain.expansion.loop.REPO_ROOT", tmp_path)

    summary = run_expand(expand_config, connection, FailingLLM(), ontology, None)
    rows = rows_from(Path(summary["retrieval_manifest"]))
    assert rows[0]["llm_status"] == mf.LLM_FAILED      # Prompt 1 really did fail
    assert summary["retrieval"]["prompt1_failures"] == 1


def test_resume_does_not_duplicate_manifest_rows(expand_config, connection, ontology,
                                                 tmp_path, monkeypatch):
    expand_config["seeds"] = {"oem": ["BMW"]}
    expand_config["expansion"]["max_tier"] = 1

    def fake_retrieval(config, entity_name, entity_key, tier, connection, manifest=None, **_kwargs):
        if tier > 0:
            return []
        candidate = result(url="https://news.test/x", text=DOCUMENT_TEXT, query="q")
        manifest.record_candidate(entity_name, tier, "gdelt", "NEWS", "q", 1, candidate)
        return [candidate]

    monkeypatch.setattr("schain.expansion.loop.retrieve_for_entity", fake_retrieval)
    monkeypatch.setattr("schain.expansion.loop.REPO_ROOT", tmp_path)

    first = run_expand(expand_config, connection, StubLLM(), ontology, FakeDriver())
    # a second run writes its own manifest; neither file gains duplicate rows
    state.enqueue_entity(connection, "BMW", 0)
    connection.execute("UPDATE frontier SET status = ?", (state.PENDING,))
    connection.commit()
    second = run_expand(expand_config, connection, StubLLM(), ontology, FakeDriver())

    for path in (first["retrieval_manifest"], second["retrieval_manifest"]):
        rows = rows_from(Path(path))
        assert len(rows) == len({(r["provider"], r["query"], r["url"]) for r in rows})
