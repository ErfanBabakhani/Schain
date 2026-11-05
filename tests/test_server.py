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
    viewer = (frontend / "explorer.html").read_text(encoding="utf-8")
    assert "__GRAPH_DATA__" not in viewer
    assert "schain-graph" in viewer


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
        assert connection.execute("SELECT MAX(version) FROM server_schema").fetchone()[0] == 2
        assert connection.execute("SELECT id FROM graph_nodes").fetchone()[0] == "legacy"
