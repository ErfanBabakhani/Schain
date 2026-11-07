"""Offline server integration tests using the real core and temporary project databases."""
import pytest
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.api import create_app
from backend.models import ProjectSettings
from backend.projects import Conflict, Projects
from backend.storage import connect, initialize, read_graph, save_graph
from backend.worker import Worker, worker_lock
from schain.viewer import validate_graph
from tests.conftest import DOCUMENT_TEXT

TOKEN = "test-token-" + "a" * 32


@pytest.fixture
def projects(tmp_path_factory):
    return Projects(tmp_path_factory.mktemp("srv"))


def settings(**kwargs):
    return ProjectSettings(name="Research", seeds=["BMW"], **kwargs)


def test_projects_seeds_and_isolation(projects):
    first, second = projects.create(settings()), projects.create(settings())
    assert first["id"] != second["id"]
    assert first["status"] == "ready"
    with connect(projects.database(first["id"])) as connection:
        graph = validate_graph(read_graph(connection))
        assert any(n["is_seed"] and n["discovery_tier"] == 0 for n in graph["nodes"])
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    projects.submit(first["id"])
    assert projects.detail(second["id"])["runs"] == []
    with pytest.raises(KeyError):
        projects.database("../outside")


@pytest.mark.parametrize("override", [
    {"max_tier": -1}, {"max_documents_per_run": 0}, {"llm_max_concurrency": 50},
    {"max_minutes": float("nan")}, {"max_total_documents": -1}, {"unknown": 1},
])
def test_limit_validation(override):
    with pytest.raises(ValidationError):
        settings(limits=override)


def test_graph_transaction_and_idempotence(projects):
    project = projects.create(settings())
    path = projects.database(project["id"])
    initialize(path)
    graph = {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [
        {"id": "e", "source": "a", "target": "b", "relation": "suppliesTo",
         "evidence": [{"text": "A supplies B"}], "source_doc_ids": ["d"]}
    ], "documents": [{"doc_id": "d"}]}
    with connect(path) as connection:
        save_graph(connection, graph)
        save_graph(connection, graph)
        assert connection.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] == 1
        assert read_graph(connection)["edges"] == graph["edges"]
        with pytest.raises(ValueError):
            save_graph(connection, {"nodes": [], "edges": graph["edges"]})
        assert read_graph(connection)["edges"] == graph["edges"]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_queue_pause_recovery_and_lock(projects):
    project_id = projects.create(settings())["id"]
    with worker_lock(projects.root):
        with pytest.raises(RuntimeError, match="Another worker"):
            with worker_lock(projects.root):
                pass
        projects.submit(project_id)
        with pytest.raises(Conflict):
            projects.submit(project_id)
        assert projects.pause(project_id)["status"] == "paused"
        projects.submit(project_id)
        with connect(projects.database(project_id)) as connection:
            with connection:
                connection.execute("UPDATE runs SET status='running' WHERE status='queued'")
        worker = Worker(projects)
        worker.recover()
        assert projects.detail(project_id)["status"] == "queued"
        with connect(projects.database(project_id)) as connection:
            with connection:
                connection.execute("UPDATE runs SET status='pausing' WHERE status='queued'")
        worker.recover()
        assert worker.tick()
        assert projects.detail(project_id)["status"] == "paused"
    with worker_lock(projects.root):
        pass


def test_manual_ingest_and_resume(projects, stub_llm):
    project_id = projects.create(settings())["id"]
    source = projects.directory(project_id) / "document.txt"
    source.write_text(DOCUMENT_TEXT, encoding="utf-8")
    worker = Worker(projects, lambda config: stub_llm)
    projects.submit(project_id, "ingest", source)
    with worker_lock(projects.root):
        assert worker.tick()
        detail = projects.detail(project_id)
        assert detail["status"] == "completed", detail["runs"]
        calls = len(stub_llm.prompts)
        # Lose compatibility files, as after a partial restore. SQLite must repair them.
        for filename in ("facts.parquet", "entity_resolution.parquet", "documents.parquet"):
            (projects.directory(project_id) / "artifacts" / filename).unlink(missing_ok=True)
        projects.submit(project_id, "ingest", source)
        worker.tick()
        assert len(stub_llm.prompts) == calls
    with connect(projects.database(project_id)) as connection:
        graph = validate_graph(read_graph(connection))
        assert graph["edges"] and graph["edges"][0]["evidence"]
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM accepted_facts").fetchone()[0] > 0
        assert connection.execute("SELECT COUNT(*) FROM frontier WHERE tier=1").fetchone()[0] >= 1
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_api_auth_validation_upload_and_conflicts(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime", TOKEN))
    assert client.get("/health").status_code == 200
    assert client.get("/projects").status_code == 401
    client.headers["Authorization"] = "Bearer " + TOKEN
    assert client.post("/projects", json={"name": "", "seeds": []}).status_code == 422
    project_id = client.post("/projects", json=settings().model_dump()).json()["id"]
    assert client.get("/projects/invalid").status_code == 404
    assert client.get(f"/projects/{project_id}/graph?limit=0").status_code == 422
    assert client.post(f"/projects/{project_id}/documents", content=b"bad", headers={"X-Filename": "../x.txt"}).status_code == 422
    assert client.post(f"/projects/{project_id}/documents", content=b"bad", headers={"X-Filename": "x.pdf"}).status_code == 422
    assert client.post(f"/projects/{project_id}/documents", content=b"text", headers={"X-Filename": "x.exe"}).status_code == 415
    result = client.post(f"/projects/{project_id}/documents", content=DOCUMENT_TEXT.encode(), headers={"X-Filename": "x.txt"})
    assert result.status_code == 202
    assert client.post(f"/projects/{project_id}/start").status_code == 409
    assert client.post(f"/projects/{project_id}/pause").json()["status"] == "paused"
    assert client.post(f"/projects/{project_id}/resume").status_code == 202
    response = client.options("/projects", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in response.headers


def test_expand_pause_and_resume_without_reextracting(projects, stub_llm, monkeypatch):
    from schain.expansion import loop
    from schain.sources.base import SearchResult

    project_id = projects.create(settings(limits={"max_tier": 2}))["id"]
    retrievals = []

    def retrieve(config, entity_name, entity_key, tier, connection, manifest=None, **kwargs):
        retrievals.append(entity_key)
        return [SearchResult(provider="test", query="BMW suppliers", title="BMW suppliers",
                             url="https://example.com/suppliers", text=DOCUMENT_TEXT,
                             entity_key=entity_key, tier=tier)]

    original = loop.process_document
    def document_then_pause(*args, **kwargs):
        summary = original(*args, **kwargs)
        projects.pause(project_id)
        return summary

    monkeypatch.setattr(loop, "retrieve_for_entity", retrieve)
    monkeypatch.setattr(loop, "process_document", document_then_pause)
    worker = Worker(projects, lambda config: stub_llm)
    projects.submit(project_id)
    with worker_lock(projects.root):
        worker.tick()
        project = projects.detail(project_id)
        assert project["status"] == "paused", project["runs"]
        assert project["runs"][0]["summary"]["stop_reason"] == "pause_requested"
        calls = stub_llm.extraction_calls
        projects.resume(project_id)
        monkeypatch.setattr(loop, "process_document", original)
        worker.tick()
        assert stub_llm.extraction_calls == calls
        assert projects.detail(project_id)["status"] == "completed"
        before = len(retrievals)
        assert not worker.tick()
        assert len(retrievals) == before


def test_settings_lock_and_seed_reconfiguration(projects):
    project_id = projects.create(settings())["id"]
    changed = ProjectSettings(name="New", seeds=["Toyota"], limits={"max_tier": 0})
    projects.update(project_id, changed)
    with connect(projects.database(project_id)) as connection:
        graph = read_graph(connection)
        assert [n["name"] for n in graph["nodes"] if n["is_seed"]] == ["Toyota"]
        assert connection.execute("SELECT entity_name FROM frontier").fetchone()[0] == "Toyota"
    projects.submit(project_id)
    with pytest.raises(Conflict):
        projects.update(project_id, settings())
    projects.pause(project_id)
    with pytest.raises(Conflict, match="immutable"):
        projects.update(project_id, settings())


def test_llamacpp_keeps_json_validation_without_vllm_fields(monkeypatch):
    from schain.config import load_config
    from schain.llm import provider

    captured = {}
    def client(**kwargs):
        captured.update(kwargs)
    monkeypatch.setattr(provider, "OpenAILLM", client)
    config = load_config("config/server.yaml")
    provider.build_llm(config)
    assert captured["model_params"]["response_format"] == {"type": "json_object"}
    assert "extra_body" not in captured["model_params"]
    assert captured["timeout"] == 600


def test_completion_and_total_budget_with_real_loop(projects, stub_llm, monkeypatch):
    from schain.expansion import loop
    from schain.sources.base import SearchResult

    project_id = projects.create(settings(limits={"max_tier": 2, "max_total_documents": 1}))["id"]
    def retrieve(config, entity_name, entity_key, tier, connection, manifest=None, **kwargs):
        return [SearchResult(provider="test", query="supplier", title="Supplier agreement",
                             url=f"https://example.com/{tier}",
                             text=DOCUMENT_TEXT if tier == 0 else "LithiumCo supplies battery materials to CATL under a supply agreement.",
                             entity_key=entity_key, tier=tier)]
    monkeypatch.setattr(loop, "retrieve_for_entity", retrieve)
    projects.submit(project_id)
    worker = Worker(projects, lambda config: stub_llm)
    with worker_lock(projects.root):
        worker.tick()
    with connect(projects.database(project_id)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert projects.detail(project_id)["status"] == "paused"

    projects.submit(project_id)
    worker.llm_factory = lambda config: pytest.fail("Budget must be checked before model initialization")
    with worker_lock(projects.root):
        worker.tick()
    detail = projects.detail(project_id)
    assert detail["status"] == "paused"
    assert detail["runs"][0]["summary"]["stop_reason"] == "total_document_budget"


def test_upload_bounds_and_allowed_origin(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.api.MAX_UPLOAD_BYTES", 3)
    client = TestClient(create_app(tmp_path / "runtime", TOKEN))
    client.headers["Authorization"] = "Bearer " + TOKEN
    project_id = client.post("/projects", json=settings().model_dump()).json()["id"]
    assert client.post(f"/projects/{project_id}/documents", content=b"1234", headers={"X-Filename": "x.txt"}).status_code == 413
    assert client.post(f"/projects/{project_id}/documents", content=b"\xff", headers={"X-Filename": "x.txt"}).status_code == 422
    assert not list((tmp_path / "runtime" / project_id / "uploads").iterdir())
    response = client.options("/projects", headers={"Origin": "http://127.0.0.1:8080", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization"})
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:8080"


def test_worker_process_lock_recovers_after_kill(projects):
    import subprocess
    import sys

    script = ("from pathlib import Path; import sys,time; from backend.worker import worker_lock; "
              "lock=worker_lock(Path(sys.argv[1])); lock.__enter__(); "
              "print('locked',flush=True); time.sleep(30)")
    process = subprocess.Popen([sys.executable, "-c", script, str(projects.root)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError):
            with worker_lock(projects.root):
                pass
    finally:
        process.kill()
        process.communicate(timeout=10)
    with worker_lock(projects.root):
        pass


def test_standalone_api_and_local_frontend(tmp_path):
    pytest.importorskip("uvicorn")
    import os
    import socket
    import subprocess
    import sys
    import time
    import httpx
    from backend.frontend import build

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "SCHAIN_DATA_DIR": str(tmp_path / "runtime"), "SCHAIN_API_TOKEN": TOKEN}
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.api:create_app",
                                "--factory", "--host", "127.0.0.1", "--port", str(port)],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=1) as client:
            for attempt in range(100):
                if process.poll() is not None:
                    pytest.fail("API exited: " + process.stderr.read().decode(errors="replace"))
                try:
                    response = client.get("/health")
                    if response.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(.1)
            else:
                pytest.fail("API did not start")
            client.headers["Authorization"] = "Bearer " + TOKEN
            response = client.post("/projects", json=settings().model_dump())
            assert response.status_code == 201, response.text
            project_id = response.json()["id"]
            graph = client.get(f"/projects/{project_id}/graph", headers={"Origin": "http://127.0.0.1:8080"})
            assert graph.headers["access-control-allow-origin"] == "http://127.0.0.1:8080"
            validate_graph(graph.json())
    finally:
        process.terminate()
        process.communicate(timeout=15)
    worker = subprocess.run([sys.executable, "-m", "backend.worker", "--once"],
                            env=env, capture_output=True, text=True, timeout=30)
    assert worker.returncode == 0, worker.stderr
    frontend = tmp_path / "frontend"
    build(frontend)
    assert (frontend / "index.html").is_file()
    page = (frontend / "index.html").read_text(encoding="utf-8")
    assert 'data-mode="server"' in page
    assert 'id="projectSelect"' in page
    assert '<iframe' not in page
    assert (frontend / "assets/js/explorer.js").is_file()
    assert not (frontend / "explorer.html").exists()


def test_additive_server_migration(tmp_path):
    from schain import state
    from backend.storage import MIGRATION_1

    path = tmp_path / "state.db"
    connection = state.connect(path, journal_mode="DELETE")
    connection.executescript(MIGRATION_1)
    with connection:
        connection.execute("INSERT INTO graph_nodes VALUES('legacy',?)", ('{"id":"legacy"}',))
    connection.close()
    initialize(path)
    initialize(path)
    with connect(path) as connection:
        assert connection.execute("SELECT MAX(version) FROM server_schema").fetchone()[0] == 3
        assert connection.execute("SELECT id FROM graph_nodes").fetchone()[0] == "legacy"


def test_live_candidates_pagination_events_and_cleanup(projects):
    from backend.activity import finish, read_activity, record
    from schain import state

    project_id = projects.create(settings())["id"]
    path = projects.database(project_id)
    projects.submit(project_id)
    run_id = projects.detail(project_id)["runs"][0]["id"]
    document = {"doc_id": "live-doc", "content_hash": "hash", "title": "Supplier report",
                "source_url": "https://example.org/report", "text": "private document text"}
    with connect(path) as connection:
        state.register_document(connection, document)
        with connection:
            connection.execute("UPDATE runs SET status='running' WHERE id=?", (run_id,))
        before = read_graph(connection)["revision"]
    graph = {"nodes": [{"id": "a", "label": "Company", "properties": {"name": "CATL"}},
                       {"id": "b", "label": "Company", "properties": {"name": "BMW"}}],
             "relationships": [{"start_node_id": "a", "end_node_id": "b", "type": "suppliesTo"},
                               {"start_node_id": "missing", "end_node_id": "b", "type": "owns"}]}
    record(path, run_id, "retrieved", document)
    record(path, run_id, "chunk_started", document, {"chunk_index": 0, "chunk_total": 1})
    record(path, run_id, "chunk_extracted", document, {"graph": graph, "chunk_index": 0, "chunk_total": 1})
    record(path, run_id, "validation", document)
    with connect(path) as connection:
        bundle = validate_graph(read_graph(connection, limit=1, node_limit=1))
        assert bundle["revision"] != before
        assert len(bundle["nodes"]) == 2  # Endpoints are included despite node pagination.
        assert len(bundle["edges"]) == 1
        assert bundle["edges"][0]["validation_status"] == "pending"
        assert bundle["documents"][0]["title"] == "Supplier report"
        assert connection.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM accepted_facts").fetchone()[0] == 0
        first = read_activity(connection, after=0, limit=2)
        second = read_activity(connection, after=first["cursor"], limit=2)
        assert first["has_more"] and not second["has_more"]
        assert [e["stage"] for e in first["events"] + second["events"]] == [
            "retrieved", "chunk_started", "chunk_extracted", "validation"]
        assert second["active_documents"][0]["stage"] == "validation"
        assert "private document text" not in str(second)
    # Resolving removes the preview in the same transaction as the accepted graph.
    with connect(path) as connection:
        save_graph(connection, {"nodes": [{"id": "accepted"}], "edges": []}, resolved_document="live-doc")
        assert not read_graph(connection)["edges"]
    finish(path, run_id, "completed")
    with connect(path) as connection:
        assert not read_activity(connection)["active_documents"]
        assert connection.execute("SELECT COUNT(*) FROM live_nodes").fetchone()[0] == 0


def test_worker_live_stages_failure_and_resume(projects, stub_llm, monkeypatch):
    from backend.activity import read_activity
    from schain import pipeline

    project_id = projects.create(settings())["id"]
    path = projects.database(project_id)
    source = projects.directory(project_id) / "supplier-report.txt"
    source.write_text(DOCUMENT_TEXT, encoding="utf-8")
    original = pipeline.validate_facts_frame
    observed = []

    def fail_validation(*args, **kwargs):
        with connect(path) as connection:
            graph = read_graph(connection)
            activity = read_activity(connection)
            assert any(e.get("validation_status") == "pending" for e in graph["edges"])
            assert activity["active_documents"][0]["stage"] == "validation"
            assert connection.execute("SELECT COUNT(*) FROM accepted_facts").fetchone()[0] == 0
            observed.extend(e["stage"] for e in activity["events"])
        raise RuntimeError("Validation temporarily unavailable")

    monkeypatch.setattr(pipeline, "validate_facts_frame", fail_validation)
    projects.submit(project_id, "ingest", source)
    worker = Worker(projects, lambda config: stub_llm)
    with worker_lock(projects.root):
        assert worker.tick()
        assert projects.detail(project_id)["status"] == "failed"
        with connect(path) as connection:
            assert not read_graph(connection)["edges"]
            assert not read_activity(connection)["active_documents"]
        assert "chunk_extracted" in observed and "resolution" in observed
        calls = stub_llm.extraction_calls
        monkeypatch.setattr(pipeline, "validate_facts_frame", original)
        projects.resume(project_id)
        assert worker.tick()
        assert projects.detail(project_id)["status"] == "completed"
        assert stub_llm.extraction_calls == calls
    with connect(path) as connection:
        graph = read_graph(connection)
        assert graph["edges"] and not any(e.get("validation_status") == "pending" for e in graph["edges"])
        activity = read_activity(connection)
        assert activity["pending_relationships"] == 0
        assert any(e["stage"] == "resolved" and e["payload"]["accepted"] > 0 for e in activity["events"])


def test_activity_auth_and_query_validation(tmp_path):
    client = TestClient(create_app(tmp_path / "runtime", TOKEN))
    client.headers["Authorization"] = "Bearer " + TOKEN
    project_id = client.post("/projects", json=settings().model_dump()).json()["id"]
    url = f"/projects/{project_id}/activity"
    assert client.get(url + "?after=0&limit=1").json()["events"] == []
    assert client.get(url + "?after=-1").status_code == 422
    assert client.get(url + "?after=9223372036854775808").status_code == 422
    assert client.get(url + "?limit=201").status_code == 422
    client.headers.pop("Authorization")
    assert client.get(url).status_code == 401


@pytest.mark.parametrize("status", ["running", "completed"])
def test_recovery_removes_interrupted_previews(projects, status):
    from backend.activity import read_activity, record

    project_id = projects.create(settings())["id"]
    path = projects.database(project_id)
    projects.submit(project_id)
    run_id = projects.detail(project_id)["runs"][0]["id"]
    graph = {"nodes": [{"id": "a", "label": "Company"}, {"id": "b", "label": "Company"}],
             "relationships": [{"start_node_id": "a", "end_node_id": "b", "type": "suppliesTo"}]}
    record(path, run_id, "chunk_extracted", {"doc_id": "unfinished"}, {"graph": graph})
    with connect(path) as connection, connection:
        connection.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))
    with worker_lock(projects.root):
        Worker(projects).recover()
    with connect(path) as connection:
        assert not read_graph(connection)["edges"]
        activity = read_activity(connection)
        assert not activity["active_documents"]
        assert activity["pending_relationships"] == 0
    assert projects.detail(project_id)["status"] == ("queued" if status == "running" else status)


def test_graph_pagination_includes_isolated_nodes_and_evidence(projects):
    project_id = projects.create(settings())["id"]
    bundle = {"nodes": [{"id": str(i)} for i in range(7)], "edges": [
        {"id": "one", "source": "0", "target": "1", "relation": "suppliesTo",
         "source_doc_ids": ["d"], "evidence": [{"doc_id": "d", "quote": "First witness"}]},
        {"id": "two", "source": "1", "target": "2", "relation": "owns",
         "source_doc_ids": ["d"], "evidence": [{"doc_id": "d", "quote": "Second witness"}]},
    ], "documents": [{"doc_id": "d"}]}
    with connect(projects.database(project_id)) as connection:
        save_graph(connection, bundle)
        offset = node_offset = 0
        nodes, edges, documents, revisions = {}, {}, {}, set()
        while True:
            page = read_graph(connection, limit=1, offset=offset, node_limit=1, node_offset=node_offset)
            validate_graph(page)
            nodes.update((node["id"], node) for node in page["nodes"])
            edges.update((edge["id"], edge) for edge in page["edges"])
            documents.update((doc["doc_id"], doc) for doc in page["documents"])
            revisions.add(page["revision"])
            offset = page["pagination"]["next_edge_offset"]
            node_offset = page["pagination"]["next_node_offset"]
            if offset == 2 and node_offset == 7:
                break
        assert set(nodes) == {str(i) for i in range(7)}
        assert list(edges.values()) == bundle["edges"]
        assert list(documents.values()) == bundle["documents"]
        assert len(revisions) == 1
