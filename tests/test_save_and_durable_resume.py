"""Incremental Neo4j save and durable Colab-resume regressions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import neo4j
import pandas as pd
import pytest
import yaml

from schain import state
from schain.__main__ import build_parser, main
from schain.documents import load_document
from schain.durable import DurableCheckpointManager
from schain.pipeline import process_document

from .conftest import StubLLM


class RecordingDriver:
    def __init__(self, fail_once: bool = False):
        self.calls: list[tuple[str, dict]] = []
        self.fail_once = fail_once

    def execute_query(self, query, **kwargs):
        if self.fail_once:
            self.fail_once = False
            raise neo4j.exceptions.ServiceUnavailable("temporary outage")
        self.calls.append((query, kwargs))

    def close(self):
        return None


def _durable_config(config, tmp_path: Path, experiment_id: str = "expand-test-001"):
    config = dict(config)
    config["checkpoint_store"] = {
        "enabled": True,
        "experiment_id": experiment_id,
        "directory": str(tmp_path / "drive" / "expand_runs"),
        "snapshot_every_documents": 1,
        "keep_snapshots": 2,
        "lock_stale_seconds": 60,
    }
    return config


def test_save_flags_are_explicit_and_conflicting_legacy_flag_is_rejected():
    parser = build_parser()
    assert parser.parse_args(["expand", "--save"]).save_override is True
    assert parser.parse_args(["expand", "--no-save"]).save_override is False
    assert parser.parse_args(["expand", "--no-neo4j"]).no_neo4j is True
    with pytest.raises(SystemExit):
        parser.parse_args(["expand", "--save", "--no-neo4j"])
    with pytest.raises(SystemExit):
        parser.parse_args(["expand", "--new-run", "--resume"])


def test_save_disabled_never_opens_neo4j(
    tmp_path, ingest_config, monkeypatch, capsys,
):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "one.txt").write_text(
        "CATL supplies battery cells to BMW. BMW builds electric vehicles.",
        encoding="utf-8",
    )
    config_path = tmp_path / "ingest.yaml"
    ingest_config.setdefault("save", {})["enabled"] = True
    config_path.write_text(yaml.safe_dump(ingest_config), encoding="utf-8")
    monkeypatch.setattr("schain.__main__.build_llm", lambda _config: StubLLM())
    monkeypatch.setattr(
        "schain.__main__.open_driver",
        lambda _config: (_ for _ in ()).throw(AssertionError("Neo4j was touched")),
    )

    assert main([
        "ingest", "--config", str(config_path), "--input", str(corpus), "--no-save",
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["save"]["enabled"] is False
    connection = state.connect(Path(ingest_config["storage"]["state_db"]))
    assert state.neo4j_commit_summary(connection) == {}
    connection.close()


def test_save_enabled_commits_each_document_incrementally(
    tmp_path, ingest_config, monkeypatch, capsys,
):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = "CATL supplies battery cells to BMW. BMW builds electric vehicles."
    (corpus / "one.txt").write_text(text, encoding="utf-8")
    (corpus / "two.txt").write_text(text + " Second source.", encoding="utf-8")
    config_path = tmp_path / "ingest.yaml"
    ingest_config.setdefault("save", {}).update({
        "enabled": False, "backend": "neo4j", "mode": "incremental",
        "max_attempts": 2, "retry_backoff_seconds": 0,
    })
    config_path.write_text(yaml.safe_dump(ingest_config), encoding="utf-8")
    driver = RecordingDriver()
    monkeypatch.setattr("schain.__main__.build_llm", lambda _config: StubLLM())
    monkeypatch.setattr("schain.__main__.open_driver", lambda _config: driver)

    assert main([
        "ingest", "--config", str(config_path), "--input", str(corpus), "--save",
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["documents_processed"] == 2
    assert summary["save"]["commits"] == {"COMMITTED": 2}
    assert summary["documents"] == {"COMMITTED": 2}
    assert len(driver.calls) >= 4


def test_transient_save_failure_retries_without_reextracting(
    document_file, ingest_config, connection, ontology,
):
    ingest_config.setdefault("save", {}).update({
        "max_attempts": 2, "retry_backoff_seconds": 0,
    })
    llm = StubLLM()
    result = process_document(
        load_document(document_file), ingest_config, connection, llm, ontology,
        RecordingDriver(fail_once=True),
    )
    assert result["status"] == state.COMMITTED
    assert llm.extraction_calls == 1
    assert state.neo4j_commit_summary(connection) == {"COMMITTED": 1}
    row = connection.execute("SELECT attempts FROM neo4j_commits").fetchone()
    assert row["attempts"] == 2


def test_failed_save_is_not_marked_committed(
    document_file, ingest_config, connection, ontology,
):
    class BrokenDriver(RecordingDriver):
        def execute_query(self, query, **kwargs):
            raise RuntimeError("transaction rejected")

    with pytest.raises(RuntimeError, match="transaction rejected"):
        process_document(
            load_document(document_file), ingest_config, connection, StubLLM(),
            ontology, BrokenDriver(),
        )
    document = load_document(document_file)
    assert state.document_status(connection, document["doc_id"]) == state.RESOLVED
    assert state.neo4j_commit_summary(connection) == {"FAILED": 1}


def test_crash_after_neo4j_commit_replays_idempotently_without_llm(
    document_file, ingest_config, connection, ontology, monkeypatch,
):
    class IdempotentDriver(RecordingDriver):
        def __init__(self):
            super().__init__()
            self.fact_ids: set[str] = set()
            self.evidence_records: set[str] = set()

        def execute_query(self, query, **kwargs):
            super().execute_query(query, **kwargs)
            for row in kwargs.get("rows", []):
                self.fact_ids.update(row.get("fact_ids") or [])
                if row.get("evidence_record"):
                    self.evidence_records.add(row["evidence_record"])

    real_set_status = state.set_document_status
    crash_once = {"armed": True}

    def crash_after_commit(conn, doc_id, status, *args, **kwargs):
        if status == state.COMMITTED and crash_once["armed"]:
            crash_once["armed"] = False
            raise RuntimeError("runtime lost after Neo4j commit")
        return real_set_status(conn, doc_id, status, *args, **kwargs)

    driver = IdempotentDriver()
    monkeypatch.setattr("schain.pipeline.state.set_document_status", crash_after_commit)
    with pytest.raises(RuntimeError, match="runtime lost"):
        process_document(
            load_document(document_file), ingest_config, connection, StubLLM(),
            ontology, driver,
        )
    before = (set(driver.fact_ids), set(driver.evidence_records))
    assert state.document_status(connection, load_document(document_file)["doc_id"]) == state.RESOLVED

    resumed_llm = StubLLM()
    result = process_document(
        load_document(document_file), ingest_config, connection, resumed_llm,
        ontology, driver,
    )
    assert result["status"] == state.COMMITTED
    assert resumed_llm.prompts == []
    assert (driver.fact_ids, driver.evidence_records) == before


def test_durable_snapshot_restores_frontier_tier_facts_and_cache(
    tmp_path, expand_config,
):
    config = _durable_config(expand_config, tmp_path)
    manager = DurableCheckpointManager.from_config(config, mode="expand")
    assert manager is not None
    assert manager.prepare(new_run=True)["mode"] == "new"
    connection = state.connect(Path(config["storage"]["state_db"]))
    state.enqueue_entity(connection, "BMW", 0)
    state.enqueue_entity(connection, "CATL", 1, discovered_from="company:bmw")
    state.set_run_value(connection, "elapsed_seconds", 42.5)
    state.save_resolution(connection, [{
        "label": "Company", "mention": "CATL", "canonical_id": "company:catl",
        "canonical_name": "CATL", "aliases": ["CATL"], "method": "test",
    }])
    facts_path = Path(config["storage"]["extracted_facts"])
    facts_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"fact_id": "fact-1"}]).to_parquet(facts_path, index=False)
    generation = manager.snapshot(connection, reason="document_completed:doc-1")
    connection.close()
    manager.release()

    for value in config["storage"].values():
        local = Path(value)
        if local.exists():
            shutil.rmtree(local) if local.is_dir() else local.unlink()
    resumed = DurableCheckpointManager.from_config(config, mode="expand")
    status = resumed.prepare(new_run=False)
    assert status["restored_snapshot"] == generation
    restored = state.connect(Path(config["storage"]["state_db"]))
    assert state.frontier_tier_counts(restored) == {0: 1, 1: 1}
    assert state.run_value(restored, "elapsed_seconds") == 42.5
    assert state.load_resolution_cache(restored)
    assert pd.read_parquet(facts_path).fact_id.tolist() == ["fact-1"]
    restored.close()
    resumed.release()


def test_resume_rejects_changed_critical_configuration(tmp_path, expand_config):
    config = _durable_config(expand_config, tmp_path, "compatibility-test")
    manager = DurableCheckpointManager.from_config(config, mode="expand")
    manager.prepare(new_run=True)
    connection = state.connect(Path(config["storage"]["state_db"]))
    manager.snapshot(connection, reason="initial")
    connection.close()
    manager.release()

    changed = _durable_config(expand_config, tmp_path, "compatibility-test")
    changed["expansion"] = dict(changed["expansion"])
    changed["expansion"]["max_tier"] = int(changed["expansion"]["max_tier"]) + 1
    incompatible = DurableCheckpointManager.from_config(changed, mode="expand")
    with pytest.raises(RuntimeError, match="compatibility check failed"):
        incompatible.prepare(new_run=False)


def test_second_live_session_is_refused(tmp_path, expand_config):
    config = _durable_config(expand_config, tmp_path, "locking-test")
    first = DurableCheckpointManager.from_config(config, mode="expand")
    first.prepare(new_run=True)
    second = DurableCheckpointManager.from_config(config, mode="expand")
    with pytest.raises(RuntimeError, match="locked by another session"):
        second.prepare(new_run=False)
    first.release()


def test_expand_cli_resumes_same_experiment_after_local_runtime_loss(
    tmp_path, expand_config, monkeypatch, capsys,
):
    config = _durable_config(expand_config, tmp_path, "cli-resume-test")
    config["seeds"] = {"test": ["BMW"]}
    config["expansion"] = dict(config["expansion"])
    config["expansion"]["max_tier"] = 0
    config_path = tmp_path / "expand-durable.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr("schain.__main__.build_llm", lambda _config: StubLLM())

    assert main([
        "expand", "--config", str(config_path), "--new-run", "--no-save",
    ]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["checkpoint_store"]["experiment_id"] == "cli-resume-test"
    first_generation = first["checkpoint_store"]["last_snapshot"]

    for value in config["storage"].values():
        local = Path(value)
        if local.exists():
            shutil.rmtree(local) if local.is_dir() else local.unlink()

    assert main([
        "expand", "--config", str(config_path), "--resume", "--no-save",
    ]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["checkpoint_store"]["restored_snapshot"] == first_generation
    assert resumed["frontier_by_tier"] == {"0": 1}
