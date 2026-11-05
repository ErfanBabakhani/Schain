"""Local-only proxy security and development connection tests; no live upstream."""
import logging
import threading
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from backend.dev_frontend import ConfigError, DevServer, Settings, load_settings, proxy_path
from backend import dev_frontend
from backend.frontend import build
from .test_frontend import browser

TOKEN = "fixture-only-token-" + "a" * 32
PROJECT_ID = "a" * 32


def environment(path, **changes):
    values = {"SCHAIN_API_URL": "http://127.0.0.1:8001", "SCHAIN_API_TOKEN": TOKEN,
              "SCHAIN_SSH_HOST": "fixture-alias", "SCHAIN_FRONTEND_PORT": "8080", **changes}
    path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")
    return path


def test_environment_is_private_and_not_loaded_into_process(tmp_path, monkeypatch):
    monkeypatch.delenv("SCHAIN_API_TOKEN", raising=False)
    path = environment(tmp_path / ".env.local")
    before = path.read_bytes()
    settings = load_settings(path)
    assert settings.api_url == "http://127.0.0.1:8001" and settings.port == 8080
    assert settings.token == TOKEN and TOKEN not in repr(settings)
    assert path.read_bytes() == before
    import os
    assert "SCHAIN_API_TOKEN" not in os.environ


@pytest.mark.parametrize("key,value", [
    ("SCHAIN_API_TOKEN", ""), ("SCHAIN_API_TOKEN", "short"),
    ("SCHAIN_API_TOKEN", '"token with spaces that is not a bearer value"'),
    ("SCHAIN_API_TOKEN", "replace-with-your-existing-backend-token"),
    ("SCHAIN_API_TOKEN", '"unterminated'),
    ("SCHAIN_API_URL", ""), ("SCHAIN_API_URL", "file:///etc/passwd"),
    ("SCHAIN_API_URL", "http://example.com"),
    ("SCHAIN_API_URL", "https://user:password@example.com"),
    ("SCHAIN_API_URL", "https://example.com/?url=http://other.example"),
    ("SCHAIN_API_URL", "http://127.0.0.1:bad"),
    ("SCHAIN_API_URL", "http://127.0.0.1:8080"),
    ("SCHAIN_API_URL", "https://example.com/prefix/../other"),
    ("SCHAIN_FRONTEND_PORT", "0"), ("SCHAIN_FRONTEND_PORT", "65536"),
    ("SCHAIN_FRONTEND_PORT", "abc"), ("SCHAIN_SSH_HOST", "alias -i private-key"),
])
def test_invalid_configuration(tmp_path, key, value):
    with pytest.raises(ConfigError) as error:
        load_settings(environment(tmp_path / ".env.local", **{key: value}))
    assert TOKEN not in str(error.value)


def test_missing_and_malformed_environment(tmp_path):
    path = tmp_path / ".env.local"
    with pytest.raises(ConfigError):
        load_settings(path)
    for suffix in ("not an assignment", "SCHAIN_FRONTEND_PORT=8081"):
        environment(path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(suffix)
        with pytest.raises(ConfigError):
            load_settings(path)
    path.write_text("SCHAIN_API_URL=http://127.0.0.1:8001\n")
    with pytest.raises(ConfigError):
        load_settings(path)


def test_company_backend_prefix_is_fixed_configuration(tmp_path):
    settings = load_settings(environment(tmp_path / ".env.local", SCHAIN_API_URL="https://api.example.org/schain/"))
    assert settings.api_url == "https://api.example.org/schain"
    assert proxy_path(f"/api/projects/{PROJECT_ID}/graph?limit=10&node_offset=0", "GET") == (
        f"/projects/{PROJECT_ID}/graph", "limit=10&node_offset=0")


@pytest.mark.parametrize("path,method", [
    ("http://other.example/api/projects", "GET"), ("//other.example/api/projects", "GET"),
    ("/api/../.env.local", "GET"), ("/api/projects/%2e%2e/graph", "GET"),
    ("/api/projects?url=https://other.example", "GET"),
    ("/api/projects?token=secret", "GET"), ("/api/projects#fragment", "GET"),
    (f"/api/projects/{PROJECT_ID}/graph?limit=0", "GET"),
    (f"/api/projects/{PROJECT_ID}/graph?limit=2&limit=3", "GET"),
    (f"/api/projects/{PROJECT_ID}/documents?offset=-1", "GET"),
    ("/api/openapi.json", "GET"), ("/api/projects", "DELETE"),
    (f"/api/projects/{PROJECT_ID}/start", "GET"),
])
def test_unsafe_proxy_paths(path, method):
    with pytest.raises(ValueError):
        proxy_path(path, method)


@pytest.fixture
def proxy(tmp_path):
    state = {"requests": [], "handler": None}

    def upstream(request):
        state["requests"].append(request)
        if state["handler"]:
            return state["handler"](request)
        return httpx.Response(200, json=[])

    directory = build(tmp_path / "frontend")
    server = DevServer(Settings("https://api.example.org/prefix", TOKEN, port=0), directory,
                       transport=httpx.MockTransport(upstream))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with httpx.Client(base_url=server.origin, trust_env=False) as client:
        bootstrap = client.get("/schain-dev-config")
        headers = {"Origin": server.origin, "X-SChain-CSRF": bootstrap.json()["csrf_token"]}
        yield server, client, headers, state
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_authorization_injection_no_response_or_asset_leakage(proxy, caplog):
    server, client, headers, state = proxy
    caplog.set_level(logging.INFO, logger="backend.dev_frontend")
    state["handler"] = lambda request: httpx.Response(200, json={"echo": "Bearer " + TOKEN}, headers={
        "Authorization": "Bearer " + TOKEN, "Set-Cookie": "secret=" + TOKEN,
        "Access-Control-Allow-Origin": "*"})
    response = client.get("/api/projects", headers={**headers, "Authorization": "Bearer browser-value"})
    assert response.status_code == 200 and response.json()["echo"] == "Bearer [redacted]"
    assert str(state["requests"][0].url) == "https://api.example.org/prefix/projects"
    assert state["requests"][0].headers["authorization"] == "Bearer " + TOKEN
    assert TOKEN not in response.text and TOKEN not in str(response.headers)
    assert "set-cookie" not in response.headers and "access-control-allow-origin" not in response.headers
    for path in ("/", "/assets/js/projects.js", "/assets/js/explorer.js", "/schain-dev-config"):
        response = client.get(path)
        assert response.status_code == 200 and TOKEN not in response.text
    assert TOKEN not in caplog.text
    for file in server.directory.rglob("*"):
        if file.is_file():
            assert TOKEN.encode() not in file.read_bytes()


def test_same_origin_and_csrf_protection(proxy):
    server, client, headers, state = proxy
    cases = [
        {**headers, "Origin": "http://evil.example"},
        {**headers, "Origin": "null"},
        {"Origin": server.origin},
        {"X-SChain-CSRF": headers["X-SChain-CSRF"]},
        {**headers, "Sec-Fetch-Site": "cross-site"},
        {**headers, "Host": "evil.example"},
    ]
    for supplied in cases:
        response = client.post("/api/projects", headers=supplied, json={"name": "test", "seeds": ["Seed"]})
        assert response.status_code == 403
    assert client.get("/api/projects").status_code == 403
    assert client.get("/schain-dev-config", headers={"Origin": "http://evil.example"}).status_code == 403
    assert client.options("/api/projects", headers={"Origin": "http://evil.example"}).status_code == 403
    assert state["requests"] == []


def test_static_files_are_whitelisted_and_upload_contract_preserved(proxy):
    server, client, headers, state = proxy
    (server.directory / ".env.local").write_text(TOKEN)
    for path in ("/.env.local", "/.git/config", "/README.md", "/%2e%2e/.env.local", "/assets/../.env.local"):
        response = client.get(path)
        assert response.status_code == 404 and TOKEN not in response.text
    response = client.post(f"/api/projects/{PROJECT_ID}/documents", content=b"source text", headers={
        **headers, "Content-Type": "application/octet-stream", "X-Filename": "source.txt"})
    assert response.status_code == 200
    request = state["requests"][-1]
    assert request.content == b"source text" and request.headers["x-filename"] == "source.txt"
    assert "cookie" not in request.headers and "x-schain-csrf" not in request.headers


@pytest.mark.parametrize("failure,status", [("connect", 502), ("timeout", 504), ("backend", 502), ("auth", 401), ("redirect", 502)])
def test_safe_upstream_errors(proxy, failure, status, caplog):
    _, client, headers, state = proxy
    def handler(request):
        if failure == "connect":
            raise httpx.ConnectError(TOKEN)
        if failure == "timeout":
            raise httpx.ReadTimeout(TOKEN)
        if failure == "redirect":
            return httpx.Response(302, headers={"Location": "https://other.example/" + TOKEN})
        return httpx.Response(401 if failure == "auth" else 500, json={"detail": TOKEN})
    state["handler"] = handler
    response = client.get("/api/projects", headers=headers)
    assert response.status_code == status
    assert TOKEN not in response.text and TOKEN not in str(response.headers) and TOKEN not in caplog.text
    assert len(state["requests"]) == 1


def test_port_conflict_fails_instead_of_sharing_listener(proxy):
    server, _, _, _ = proxy
    with pytest.raises(OSError):
        DevServer(Settings(server.settings.api_url, TOKEN, port=server.server_port), server.directory)


def test_running_server_cannot_be_replaced_by_a_demo_build(proxy):
    server, client, _, _ = proxy
    original = client.get("/").content
    script = client.get("/assets/js/projects.js").content
    assert client.get("/index.html").content == original
    (server.directory / "index.html").write_bytes(Path("demo/SChain_Graph_Explorer/index.html").read_bytes())
    (server.directory / "assets/js/projects.js").write_text("window.SCHAIN_DEMO_DATA = {nodes:[],edges:[]};")
    assert client.get("/").content == original
    assert client.get("/index.html").content == original
    assert client.get("/assets/js/projects.js").content == script


def test_build_identity_headers_and_explicit_entry_point(proxy):
    server, client, _, _ = proxy
    root, index = client.get("/"), client.get(server.entry_url)
    assert root.content == index.content
    assert root.headers["cache-control"] == "no-store"
    assert root.headers["x-schain-frontend"] == "projects-" + server.frontend_build
    assert root.headers["x-schain-mode"] == "server"
    config = client.get("/schain-dev-config").json()
    assert config["frontend_build"] == server.frontend_build
    assert config["application"] == "schain-projects"
    assert server.entry_url.endswith(config["entry_point"])
    assert client.get("/assets/js/projects.js?v=obsolete").status_code == 409
    assert client.get("/index.html?build=obsolete").status_code == 409


def test_listener_diagnostics_do_not_echo_untrusted_metadata():
    def existing(request):
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"mode": "local-proxy", "frontend_build": "abcdef123456", "csrf_token": TOKEN})
    assert "abcdef123456" in dev_frontend.describe_listener(8080, transport=httpx.MockTransport(existing))
    def unrelated(request):
        return httpx.Response(200, json={"mode": "other", "detail": TOKEN})
    message = dev_frontend.describe_listener(8080, transport=httpx.MockTransport(unrelated))
    assert "Another listener" in message and TOKEN not in message


@pytest.fixture
def dev_workspace(tmp_path_factory, browser):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from backend.api import create_app

    directory = tmp_path_factory.mktemp("devweb")
    client = TestClient(create_app(directory / "runtime", TOKEN))
    state = {"offline": False, "requests": []}
    def upstream(request):
        if state["offline"]:
            raise httpx.ConnectError("simulated missing tunnel")
        response = client.request(request.method, request.url.raw_path.decode(), content=request.content, headers=dict(request.headers))
        return httpx.Response(response.status_code, content=response.content, headers=dict(response.headers))
    server = DevServer(Settings("http://127.0.0.1:8001", TOKEN, port=0), build(directory / "frontend"),
                       transport=httpx.MockTransport(upstream))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    context = browser.new_context(viewport={"width": 1366, "height": 900})
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: state["requests"].append(request))
    yield server, page, client, state
    context.close()
    client.close()
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    assert errors == []


def test_browser_auto_connect_empty_create_reconnect_and_existing_projects(dev_workspace):
    server, page, client, state = dev_workspace
    response = page.goto(server.origin)
    assert response.body() == server.assets["index.html"]
    assert response.headers["x-schain-frontend"] == "projects-" + server.frontend_build
    page.wait_for_function("!document.getElementById('newProject').disabled")
    assert not page.locator("#connectionDialog").is_visible()
    assert "No projects yet" in page.locator("#graphEmptyState").inner_text()
    assert page.locator("#apiToken").input_value() == ""
    page.locator("#newProject").click()
    page.locator("#projectName").fill("Development project")
    page.locator("#projectSeeds").fill("Research seed")
    page.locator("#saveProject").click()
    page.wait_for_function("document.getElementById('visibleNodeCount').textContent === '1'")
    page.locator("#startRun").click()
    page.wait_for_function("document.getElementById('projectState').textContent === 'queued'")
    page.locator("#pauseRun").click()
    page.wait_for_function("document.getElementById('projectState').textContent === 'paused'")
    page.reload()
    page.wait_for_function("!document.getElementById('newProject').disabled")
    assert "Development project" in page.locator("#projectSelect").inner_text()
    page.locator("#connectButton").click()
    page.wait_for_function("!document.getElementById('connectButton').disabled")
    assert not page.locator("#connectionDialog").is_visible()
    for request in state["requests"]:
        assert TOKEN not in (request.post_data or "")
        assert "authorization" not in request.headers
        assert request.url.startswith(server.origin + "/")
    assert TOKEN not in page.content()


def test_browser_with_old_http_and_cache_storage_opens_canonical_entry(dev_workspace, monkeypatch):
    server, page, _, state = dev_workspace
    original = dev_frontend.Handler.do_GET
    legacy_mode = {"enabled": True}
    demo = Path("demo/SChain_Graph_Explorer/index.html").read_bytes()

    def legacy_page(handler):
        if handler.server is server and handler.path == "/" and legacy_mode["enabled"]:
            handler.send_response(200)
            handler.send_header("Content-Type", "text/html; charset=utf-8")
            handler.send_header("Cache-Control", "public, max-age=31536000")
            handler.send_header("Content-Length", str(len(demo)))
            handler.end_headers()
            handler.wfile.write(demo)
        else:
            original(handler)

    monkeypatch.setattr(dev_frontend.Handler, "do_GET", legacy_page)
    page.goto(server.origin)
    assert page.locator("body").get_attribute("data-mode") == "demo"
    assert page.locator("#newProject").count() == 0
    page.evaluate("""async html => { const cache = await caches.open('previous-demo'); await cache.put('/', new Response(html, {headers:{'Content-Type':'text/html'}})); }""", demo.decode("utf-8"))
    legacy_mode["enabled"] = False
    request_count = len(state["requests"])
    response = page.goto(server.entry_url)
    assert response.body() == server.assets["index.html"]
    page.wait_for_function("!document.getElementById('newProject').disabled")
    assert page.locator("body").get_attribute("data-mode") == "server"
    assert page.locator("#visibleNodeCount").inner_text() == "0"
    assert "No projects yet" in page.locator("#graphEmptyState").inner_text()
    assert not any("demo-data" in request.url or "/data/catalog" in request.url for request in state["requests"][request_count:])
    page.reload()
    page.wait_for_function("!document.getElementById('newProject').disabled")
    assert page.locator("body").get_attribute("data-mode") == "server"


def test_browser_unavailable_tunnel_has_backoff_and_recovers(dev_workspace):
    server, page, _, state = dev_workspace
    state["offline"] = True
    page.goto(server.origin)
    page.wait_for_function("document.getElementById('appMessage').textContent.includes('Retrying in 2s')")
    assert page.locator("#newProject").is_disabled()
    assert not page.locator("#connectionDialog").is_visible()
    page.wait_for_function("document.getElementById('appMessage').textContent.includes('Retrying in 4s')")
    state["offline"] = False
    page.locator("#connectButton").click()
    page.wait_for_function("!document.getElementById('newProject').disabled")
    assert "No projects yet" in page.locator("#graphEmptyState").inner_text()


def test_command_reports_missing_tunnel_without_leaking_or_overwriting(tmp_path, monkeypatch, capsys):
    env = environment(tmp_path / ".env.local")
    original = env.read_bytes()
    monkeypatch.setattr(dev_frontend, "ROOT", tmp_path)
    opened, requests = [], []
    monkeypatch.setattr(dev_frontend.webbrowser, "open", opened.append)
    real_server = dev_frontend.DevServer

    def upstream(request):
        requests.append(request)
        raise httpx.ConnectError(TOKEN)

    def factory(settings, directory):
        server = real_server(replace(settings, port=0), directory, transport=httpx.MockTransport(upstream))
        def stop():
            raise KeyboardInterrupt
        server.serve_forever = stop
        return server

    monkeypatch.setattr(dev_frontend, "DevServer", factory)
    dev_frontend.main(["--env", str(env)])
    output = capsys.readouterr()
    assert "Backend unavailable" in output.out and "Reconnect" in output.out
    assert len(opened) == 1 and opened[0].startswith("http://127.0.0.1:")
    assert len(requests) == 1 and requests[0].url.path == "/projects"
    assert TOKEN not in output.out + output.err
    assert env.read_bytes() == original
    assert not list((tmp_path / "outputs/server-frontend").rglob(".env*"))


def test_command_reports_occupied_port_before_connecting(tmp_path, monkeypatch, capsys):
    env = environment(tmp_path / ".env.local")
    monkeypatch.setattr(dev_frontend, "ROOT", tmp_path)
    def unavailable(*args, **kwargs):
        raise OSError(TOKEN)
    monkeypatch.setattr(dev_frontend, "DevServer", unavailable)
    monkeypatch.setattr(dev_frontend, "describe_listener", lambda port: "Existing fixture server; no process was terminated.")
    with pytest.raises(SystemExit) as error:
        dev_frontend.main(["--env", str(env), "--no-browser"])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert "port 8080 is unavailable" in output.err
    assert TOKEN not in output.out + output.err
