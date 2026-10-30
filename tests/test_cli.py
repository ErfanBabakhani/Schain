"""The CLI: argument parsing, resume versus restart, and the validate subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from schain import state
from schain.__main__ import build_parser, main

from .conftest import DOCUMENT_TEXT, StubLLM


@pytest.fixture
def cli_config(tmp_path, ingest_config):
    """Write the (tmp-storage) ingest config to disk so --config can point at it."""
    path = tmp_path / "ingest.yaml"
    path.write_text(yaml.safe_dump(ingest_config), encoding="utf-8")
    return path


@pytest.fixture
def corpus(tmp_path):
    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "a.txt").write_text(DOCUMENT_TEXT, encoding="utf-8")
    return directory


@pytest.fixture
def llms(monkeypatch):
    """Hand out a fresh StubLLM per run and keep them for inspection."""
    created: list[StubLLM] = []

    def build(config):
        created.append(StubLLM())
        return created[-1]

    monkeypatch.setattr("schain.__main__.build_llm", build)
    return created


def run_cli(*args):
    return main(list(args))


# --- argument parsing --------------------------------------------------------------


def test_the_three_modes_exist():
    parser = build_parser()
    for mode in ("ingest", "expand", "validate", "export"):
        assert parser.parse_args([mode]).mode == mode


def test_a_mode_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_unknown_mode_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["monitor"])


def test_expansion_limits_are_exposed():
    args = build_parser().parse_args(["expand", "--max-tier", "2", "--max-documents", "5"])
    assert args.max_tier == 2 and args.max_documents == 5


# --- ingest ------------------------------------------------------------------------


def test_ingest_runs_and_reports(cli_config, corpus, llms, capsys):
    assert run_cli("ingest", "--config", str(cli_config), "--input", str(corpus),
                   "--no-neo4j") == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["documents_processed"] == 1
    assert summary["facts"] == 3
    assert llms[0].extraction_calls == 1
    assert "graph_export" in summary
    assert Path(summary["graph_export"]).is_file()


def test_resume_is_the_default(cli_config, corpus, llms, ingest_config):
    """A second run with no flags must not call the model again."""
    run_cli("ingest", "--config", str(cli_config), "--input", str(corpus), "--no-neo4j")
    facts_after_first = len(pd.read_parquet(ingest_config["storage"]["extracted_facts"]))

    run_cli("ingest", "--config", str(cli_config), "--input", str(corpus), "--no-neo4j")
    assert llms[1].prompts == []
    assert len(pd.read_parquet(ingest_config["storage"]["extracted_facts"])) == facts_after_first


def test_restart_recomputes_from_scratch(cli_config, corpus, llms, ingest_config):
    run_cli("ingest", "--config", str(cli_config), "--input", str(corpus), "--no-neo4j")
    export_path = Path(ingest_config["storage"]["graph_export"])
    assert export_path.exists()
    export_path.write_text("stale", encoding="utf-8")

    run_cli("ingest", "--config", str(cli_config), "--input", str(corpus),
            "--no-neo4j", "--restart")
    # the model was called again, because the checkpoint state was cleared
    assert llms[1].extraction_calls == 1
    # and the facts were rebuilt rather than duplicated
    facts = pd.read_parquet(ingest_config["storage"]["extracted_facts"])
    assert len(facts) == 3 and facts.fact_id.is_unique
    # restart removed the stale export, then the successful run rewrote it
    assert export_path.exists()
    assert export_path.read_text(encoding="utf-8") != "stale"


def test_restart_clears_the_state_database(cli_config, corpus, llms, ingest_config):
    run_cli("ingest", "--config", str(cli_config), "--input", str(corpus), "--no-neo4j")
    connection = state.connect(Path(ingest_config["storage"]["state_db"]))
    assert state.document_counts(connection) == {state.RESOLVED: 1}
    connection.close()

    run_cli("ingest", "--config", str(cli_config), "--input", str(corpus),
            "--no-neo4j", "--restart")
    connection = state.connect(Path(ingest_config["storage"]["state_db"]))
    # one document again, not two: the previous run's rows were removed, not appended to
    assert state.document_counts(connection) == {state.RESOLVED: 1}
    connection.close()


def test_a_failing_run_reports_a_nonzero_exit_code(cli_config, corpus, monkeypatch):
    def build(config):
        from .conftest import FailingLLM

        return FailingLLM()

    monkeypatch.setattr("schain.__main__.build_llm", build)
    # the document fails, but the run itself completes and reports the failure
    assert run_cli("ingest", "--config", str(cli_config), "--input", str(corpus),
                   "--no-neo4j") == 0


def test_an_unreachable_endpoint_is_reported_clearly(cli_config, corpus, monkeypatch, caplog):
    def build(config):
        from neo4j_graphrag.exceptions import LLMGenerationError

        raise LLMGenerationError("Connection error.")

    monkeypatch.setattr("schain.__main__.build_llm", build)
    assert run_cli("ingest", "--config", str(cli_config), "--input", str(corpus),
                   "--no-neo4j") == 1
    assert "Check that the server at" in caplog.text


# --- validate ----------------------------------------------------------------------


def test_validate_reports_coverage(cli_config, corpus, llms, tmp_path, capsys):
    run_cli("ingest", "--config", str(cli_config), "--input", str(corpus), "--no-neo4j")
    capsys.readouterr()   # discard the ingest summary so only the report is read back
    reference = tmp_path / "relations.csv"
    reference.write_text(
        "supplier,customer\nContemporary Amperex Technology Co. Limited,BMW\nBosch,Toyota\n",
        encoding="utf-8",
    )
    assert run_cli("validate", "--config", str(cli_config),
                   "--reference", str(reference)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["reference_edges"] == 2
    assert report["matched_edges"] == 1


def test_validate_without_a_reference_explains_what_to_do(cli_config, caplog):
    assert run_cli("validate", "--config", str(cli_config)) == 1
    assert "EXTERNAL_DATASETS.md" in caplog.text


# --- export ------------------------------------------------------------------------


def test_standalone_export_does_not_invoke_the_llm(cli_config, corpus, llms, monkeypatch, capsys):
    run_cli("ingest", "--config", str(cli_config), "--input", str(corpus), "--no-neo4j")
    capsys.readouterr()
    calls_before = len(llms)

    def refuse(config):
        raise AssertionError("export must not build an LLM")

    monkeypatch.setattr("schain.__main__.build_llm", refuse)
    assert run_cli("export", "--config", str(cli_config)) == 0
    report = json.loads(capsys.readouterr().out)
    assert Path(report["graph_export"]).is_file()
    assert len(llms) == calls_before


def test_standalone_export_does_not_open_neo4j(cli_config, corpus, llms, monkeypatch, capsys):
    run_cli("ingest", "--config", str(cli_config), "--input", str(corpus), "--no-neo4j")
    capsys.readouterr()

    def refuse(config):
        raise AssertionError("export must not open Neo4j")

    monkeypatch.setattr("schain.__main__.open_driver", refuse)
    monkeypatch.setattr(
        "schain.__main__.build_llm",
        lambda config: (_ for _ in ()).throw(AssertionError("export must not build an LLM")),
    )
    assert run_cli("export", "--config", str(cli_config)) == 0


def test_expand_also_writes_graph_export(tmp_path, expand_config, llms, monkeypatch, capsys):
    """EXPAND ends with the same derived JSON artifact as INGEST."""
    import yaml

    from schain.sources.base import SearchResult

    expand_config["seeds"] = {"ev_manufacturers": ["BMW"]}
    expand_config["expansion"]["max_tier"] = 0
    expand_config["expansion"]["max_documents"] = 1
    expand_config.setdefault("verification", {})["enabled"] = False
    config_path = tmp_path / "expand.yaml"
    config_path.write_text(yaml.safe_dump(expand_config), encoding="utf-8")

    def fake(config, entity_name, entity_key, tier, connection, manifest=None, **_kwargs):
        result = SearchResult(
            provider="test", query='"BMW" suppliers',
            title="BMW supplier news", url="https://www.bmwgroup.com/news/0",
            text=DOCUMENT_TEXT,
            retrieval_url="https://www.bmwgroup.com/news/0",
            canonical_source_url="https://www.bmwgroup.com/news/0",
            source_domain="bmwgroup.com",
        )
        result._source_known = True
        result._source_class = "official_primary"
        result._source_base_score = 94
        result._source_effective_score = 99
        result._source_group_id = "domain:bmwgroup.com"
        return [result]

    monkeypatch.setattr("schain.expansion.loop.retrieve_for_entity", fake)
    assert run_cli(
        "expand", "--config", str(config_path), "--no-neo4j", "--max-tier", "0",
        "--max-documents", "1",
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert "graph_export" in summary
    assert Path(summary["graph_export"]).is_file()
