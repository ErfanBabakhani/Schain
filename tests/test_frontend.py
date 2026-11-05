"""Graph Explorer build and real-browser integration against temporary projects."""
import functools
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

from backend.frontend import build
from schain.viewer import write_viewer


def test_build_uses_shared_dark_renderer_and_replaces_only_owned_artifact(tmp_path):
    old = tmp_path / "explorer.html"
    old.write_text('<title>SChain · Evidence explorer</title>', encoding="utf-8")
    owned = tmp_path / "notes.txt"
    owned.write_text("keep this file")
    build(tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'data-mode="server"' in html and 'id="projectSelect"' in html
    assert '<iframe' not in html and 'demo-data.js' not in html
    assert 'schain-frontend' in html and '?v=' in html
    assert not old.exists() and owned.read_text() == "keep this file"
    assert (tmp_path / "assets/js/explorer.js").read_bytes() == Path("visualization/assets/js/explorer.js").read_bytes()
    assert not (tmp_path / "assets/js/demo-data.js").exists()


def test_unchanged_build_preserves_files_and_repairs_only_changed_output(tmp_path):
    build(tmp_path)
    paths = [tmp_path / "index.html", tmp_path / "assets/js/explorer.js", tmp_path / "assets/js/projects.js"]
    for path in paths:
        os.utime(path, ns=(1000000000, 1000000000))
    before = {path: path.stat().st_mtime_ns for path in paths}
    build(tmp_path)
    assert {path: path.stat().st_mtime_ns for path in paths} == before
    paths[0].write_text('<body data-mode="demo">wrong frontend</body>')
    build(tmp_path)
    assert 'data-mode="server"' in paths[0].read_text(encoding="utf-8")
    assert paths[1].stat().st_mtime_ns == before[paths[1]]


@pytest.fixture(scope="module")
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as automation:
        options = {"headless": True}
        if os.name == "nt":
            options["channel"] = "msedge"
        executable = os.environ.get("SCHAIN_BROWSER_EXECUTABLE")
        if executable:
            options["executable_path"] = executable
            options.pop("channel", None)
        try:
            browser = automation.chromium.launch(**options)
        except playwright.Error as error:
            pytest.skip(f"Browser unavailable: {error}")
        yield browser
        browser.close()


@pytest.fixture
def workspace(tmp_path_factory, monkeypatch, browser):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from backend.api import create_app
    from backend.projects import Projects

    directory = tmp_path_factory.mktemp("web")
    frontend = build(directory / "frontend")
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(SimpleHTTPRequestHandler, directory=str(frontend)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("SCHAIN_CORS_ORIGINS", origin)
    token = "browser-fixture-" + "a" * 32
    client = TestClient(create_app(directory / "runtime", token))
    client.headers["Authorization"] = "Bearer " + token
    projects = Projects(directory / "runtime")
    context = browser.new_context(viewport={"width": 1366, "height": 900})
    page = context.new_page()
    errors, requests = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    options = {"small_pages": False}

    def route(request_route):
        request = request_route.request
        url = urlsplit(request.url)
        if request.url.startswith(origin + "/"):
            if url.path.startswith(("/data/", "/resolve-source", "/evidence-content", "/evidence-preview")):
                errors.append("Unexpected demo dependency: " + url.path)
                request_route.abort()
                return
            request_route.continue_()
        elif url.netloc == "127.0.0.1:8001":
            path = url.path + ("?" + url.query if url.query else "")
            if options["small_pages"] and url.path.endswith("/graph"):
                query = parse_qs(url.query)
                query.update(limit=["1"], node_limit=["1"])
                path = url.path + "?" + urlencode(query, doseq=True)
            headers = dict(request.headers)
            # TestClient's default token must not hide a missing browser Authorization header.
            headers.setdefault("authorization", "")
            body = request.post_data_buffer
            requests.append((request.method, path, body, headers))
            response = client.request(request.method, path, content=body, headers=headers)
            if url.path.endswith("/graph") and options.get("after_graph"):
                options.pop("after_graph")()
            request_route.fulfill(status=response.status_code, headers=dict(response.headers), body=response.content)
        else:
            errors.append("Unexpected external request: " + request.url)
            request_route.abort()

    context.route("**/*", route)
    page.goto(origin)
    yield {"page": page, "client": client, "projects": projects, "requests": requests,
           "errors": errors, "token": token, "origin": origin, "directory": directory,
           "frontend": frontend, "options": options}
    context.close()
    client.close()
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    assert errors == []


def connect(workspace):
    page = workspace["page"]
    page.locator("#connectButton").click()
    page.locator("#apiToken").fill(workspace["token"])
    page.locator("#connectionForm button.primary-btn").click()
    page.wait_for_function("!document.getElementById('projectSelect').disabled")


def create_project(workspace, name="Project A", seeds=None):
    return workspace["client"].post("/projects", json={"name": name, "seeds": seeds or ["BMW"]}).json()


def wait_status(page, status):
    page.wait_for_function("status => document.getElementById('projectState').textContent === status", arg=status)


def test_empty_workspace_creation_and_lifecycle(workspace):
    page = workspace["page"]
    assert page.locator("#visibleNodeCount").inner_text() == "0"
    assert workspace["client"].get("/projects").json() == []
    assert page.locator("#graphCanvas").is_visible()
    assert page.locator("#projectSelect").is_disabled()
    connect(workspace)
    assert page.locator("#graphEmptyState").inner_text().startswith("No projects yet")
    page.locator("#newProject").click()
    assert page.locator("#projectSeeds").input_value() == ""
    page.locator("#projectName").fill("Battery research")
    page.locator("#projectSeeds").fill("BMW\nToyota")
    page.locator("#saveProject").click()
    wait_status(page, "ready")
    page.wait_for_function("document.getElementById('visibleNodeCount').textContent === '2'")
    project = workspace["client"].get("/projects").json()[0]
    assert project["seeds"] == ["BMW", "Toyota"] and project["runs"] == []
    top = page.locator(".topbar").bounding_box()
    for selector in ("#projectSelect", "#newProject", "#manageProject", "#connectButton", "#datasetMenu"):
        bounds = page.locator(selector).bounding_box()
        assert bounds["y"] >= top["y"] and bounds["y"] + bounds["height"] <= top["y"] + top["height"]
    payload = json.loads(next(body for method, path, body, _ in workspace["requests"] if method == "POST" and path == "/projects"))
    from backend.models import Limits
    assert payload["limits"] == Limits().model_dump()
    page.locator("#startRun").click()
    wait_status(page, "queued")
    assert page.locator("#startRun").is_disabled()
    page.locator("#manageProject").click()
    assert page.locator("#projectSeeds").is_disabled()
    assert page.locator("#max_tier").is_disabled()
    assert page.locator("#documentInput").is_disabled()
    page.locator("#projectDialog [data-close-dialog]").click()
    page.locator("#pauseRun").click()
    wait_status(page, "paused")
    page.locator("#resumeRun").click()
    wait_status(page, "queued")
    from backend.storage import connect as database
    with database(workspace["projects"].database(project["id"])) as connection:
        with connection:
            connection.execute("UPDATE runs SET status='running' WHERE status='queued'")
    wait_status(page, "running")
    page.locator("#pauseRun").click()
    wait_status(page, "pausing")
    assert page.locator("#pauseRun").is_disabled()
    assert "finishing" in page.locator("#projectSummary").inner_text()
    from backend.worker import Worker, worker_lock
    with worker_lock(workspace["projects"].root):
        worker = Worker(workspace["projects"], lambda config: pytest.fail("Paused work must not invoke a model"))
        worker.recover()
        worker.tick()
    wait_status(page, "paused")
    page.set_viewport_size({"width": 1920, "height": 1080})
    assert page.locator("#graphCanvas").bounding_box()["width"] > 1000
    page.screenshot(path=str(workspace["directory"] / "projects.png"))


def test_browser_upload_uses_pipeline_and_publishes_graph(workspace, stub_llm):
    from backend.worker import Worker, worker_lock
    from tests.conftest import DOCUMENT_TEXT

    project = create_project(workspace)
    page = workspace["page"]
    connect(workspace)
    page.locator("#projectSelect").select_option(project["id"])
    wait_status(page, "ready")
    page.locator("#manageProject").click()
    page.locator("#documentInput").set_input_files({"name": "سند.txt", "mimeType": "text/plain", "buffer": DOCUMENT_TEXT.encode()})
    page.locator("#uploadDocument").click()
    wait_status(page, "queued")
    upload = next(r for r in workspace["requests"] if r[0] == "POST" and r[1].endswith("/documents"))
    assert upload[2] == DOCUMENT_TEXT.encode()
    assert upload[3]["x-filename"].endswith(".txt")
    with worker_lock(workspace["projects"].root):
        Worker(workspace["projects"], lambda config: stub_llm).tick()
    wait_status(page, "completed")
    page.wait_for_function("Number(document.getElementById('visibleEdgeCount').textContent) > 0")
    assert "RESOLVED" in page.locator("#documentList").inner_text()
    assert page.locator("#resumeRun").is_enabled()
    page.locator("#projectDialog [data-close-dialog]").click()
    page.locator('[data-side-tab="evidence"]').click()
    assert "CATL" in page.locator("#sideContent").inner_text()
    assert "Source quote verified" in page.locator("#sideContent").inner_text()


def test_pagination_revisions_and_project_switching(workspace):
    from backend.storage import connect as database, read_graph, save_graph

    first = create_project(workspace, "First", ["First Seed"])
    second = create_project(workspace, "Second", ["Second Seed"])
    def graph(project, extra):
        with database(workspace["projects"].database(project["id"])) as connection:
            bundle = read_graph(connection)
            bundle["nodes"].append({"id": extra, "name": extra, "type": "Company"})
            save_graph(connection, bundle)
    graph(first, "First isolated")
    graph(second, "Second isolated")
    workspace["options"]["small_pages"] = True
    workspace["options"]["after_graph"] = lambda: graph(first, "Changed during pagination")
    page = workspace["page"]
    connect(workspace)
    page.locator("#projectSelect").select_option(first["id"])
    page.wait_for_function("document.getElementById('datasetTitle').textContent === 'First' && document.getElementById('datasetMeta').textContent.startsWith('3 entities')")
    page.locator('[data-view="global"]').click()
    assert page.locator("#visibleNodeCount").inner_text() == "3"
    page.locator('[data-dimension="2d"]').click()
    page.locator("#graphCanvas").hover()
    page.mouse.wheel(0, -200)
    before = page.evaluate("SChainExplorer.getViewState()")
    requests = len([r for r in workspace["requests"] if "/graph?" in r[1]])
    page.wait_for_timeout(4500)
    assert len([r for r in workspace["requests"] if "/graph?" in r[1]]) == requests
    assert page.evaluate("SChainExplorer.getViewState()") == before
    graph(first, "Another isolated")
    page.wait_for_function("document.getElementById('visibleNodeCount').textContent === '4'")
    assert page.evaluate("SChainExplorer.getViewState()") == before
    page.evaluate("""id => { const original = window.fetch; window.fetch = (url, options) => String(url).includes(id + '/graph') ? new Promise((resolve,reject) => setTimeout(() => original(url,options).then(resolve,reject), 700)) : original(url,options); }""", first["id"])
    page.locator("#projectSelect").select_option(second["id"])
    page.locator("#projectSelect").select_option(first["id"])
    page.locator("#projectSelect").select_option(second["id"])
    page.wait_for_function("document.getElementById('datasetTitle').textContent === 'Second' && document.getElementById('datasetMeta').textContent.startsWith('2 entities')")
    page.wait_for_timeout(1000)
    assert "First" not in page.locator("#sideContent").inner_text()
    assert page.locator("#visibleNodeCount").inner_text() == "2"
    workspace["token"] = "invalid-token"
    page.locator("#connectButton").click()
    page.locator("#apiToken").fill(workspace["token"])
    page.locator("#connectionForm button.primary-btn").click()
    page.wait_for_function("document.getElementById('connectionError').textContent.includes('Invalid bearer')")
    assert page.locator("#projectSelect").is_disabled()
    assert page.locator("#visibleNodeCount").inner_text() == "0"


def test_offline_renderer_preserves_selection_camera_and_grounding(workspace):
    data = {"nodes": [{"id": "a", "is_seed": True}, {"id": "b", "name": "B", "type": "Company", "discovery_tier": 1}],
            "edges": [{"id": "e", "source": "b", "target": "a", "relation": "suppliesTo", "evidence": [{"quote": "q", "evidence_quote_grounded": False}]}]}
    write_viewer(data, workspace["frontend"] / "offline.html")
    page = workspace["page"]
    page.goto(workspace["origin"] + "/offline.html")
    assert page.locator("#visibleNodeCount").inner_text() == "2"
    assert not page.locator("#projectSelect").is_visible()
    page.locator('[data-dimension="2d"]').click()
    page.locator("#entitySearch").fill("B")
    page.locator('#searchResults [data-node="b"]').click()
    page.wait_for_timeout(100)
    before = page.evaluate("SChainExplorer.getViewState()")
    assert before["selectedNodes"] == ["b"]
    page.locator('[data-side-tab="evidence"]').click()
    assert "Quote did not match source" in page.locator("#sideContent").inner_text()
    page.locator('[data-evidence-index="0"]').click()
    assert page.locator("#evidenceModal").is_visible()
    assert page.locator("#modalQuote").inner_text() == '“q”'
    page.locator("#modalCloseBtn").click()
    assert page.evaluate("SChainExplorer.getViewState()") == before
    from playwright.sync_api import Error
    with pytest.raises(Error, match="must be a string"):
        page.evaluate("SChainExplorer.setGraph({nodes:[{id:'bad',name:7}],edges:[]})")
    assert page.evaluate("SChainExplorer.getViewState()") == before
    page.evaluate("data => SChainExplorer.setGraph(data, {preserveView:true})", data)
    assert page.evaluate("SChainExplorer.getViewState()") == before
    assert not workspace["requests"]
