"""Loopback-only development frontend with a server-side authenticated API proxy."""
import argparse
import hmac
import io
import json
import logging
import mimetypes
import re
import secrets
import socket
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

import httpx
from dotenv import dotenv_values

from .frontend import ASSET_FILES, build, render_frontend

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
STATIC_FILES = frozenset({"index.html", *ASSET_FILES})
PROJECT = r"/projects/[0-9a-f]{32}"
ROUTES = (
    (r"/health", {"GET"}), (r"/projects", {"GET", "POST"}),
    (PROJECT, {"GET", "PUT"}),
    (PROJECT + r"/(start|pause|resume)", {"POST"}),
    (PROJECT + r"/documents", {"GET", "POST"}),
    (PROJECT + r"/graph", {"GET"}),
)
MAX_UPLOAD = 10 * 1024 * 1024
MAX_JSON = 256 * 1024
MAX_RESPONSE = 32 * 1024 * 1024


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    api_url: str
    token: str = field(repr=False)
    ssh_host: str = ""
    port: int = 8080


def load_settings(path: Path) -> Settings:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise ConfigError("Cannot read the local environment file; create .env.local first") from error
    seen = set()
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        if not match or match[1] in seen:
            raise ConfigError("Environment entries must be unique, single-line KEY=VALUE assignments")
        seen.add(match[1])
    values = dotenv_values(stream=io.StringIO(text), interpolate=False)
    if set(values) != seen:
        raise ConfigError("Environment entries must contain valid single-line values")
    token = values.get("SCHAIN_API_TOKEN") or ""
    if len(token) < 32 or not re.fullmatch(r"[A-Za-z0-9._~+/\-]+=*", token) or token.startswith("replace-"):
        raise ConfigError("SCHAIN_API_TOKEN must contain a valid bearer token of at least 32 characters")
    url = values.get("SCHAIN_API_URL") or ""
    try:
        parsed = urlsplit(url)
        port = parsed.port
        valid = (
            parsed.scheme in ("http", "https") and parsed.hostname
            and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment
            and not any(char.isspace() or ord(char) < 32 for char in url)
            and not any(part in (".", "..") for part in parsed.path.split("/"))
            and not any(char in url for char in ("%", "\\"))
            and token not in url
            and (port is None or 1 <= port <= 65535)
        )
        if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            valid = False
    except ValueError:
        valid = False
    if not valid:
        raise ConfigError("SCHAIN_API_URL must be a fixed HTTP loopback or HTTPS backend URL without credentials or a query")
    raw_port = values.get("SCHAIN_FRONTEND_PORT", "8080") or ""
    if not re.fullmatch(r"[0-9]{1,5}", raw_port) or not 1 <= int(raw_port) <= 65535:
        raise ConfigError("SCHAIN_FRONTEND_PORT must be between 1 and 65535")
    if parsed.hostname in ("127.0.0.1", "localhost", "::1") and int(raw_port) == (parsed.port or (443 if parsed.scheme == "https" else 80)):
        raise ConfigError("Frontend and tunneled backend must use different ports")
    ssh_host = values.get("SCHAIN_SSH_HOST") or ""
    if ssh_host and (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", ssh_host) or token in ssh_host):
        raise ConfigError("SCHAIN_SSH_HOST must be an SSH configuration alias")
    return Settings(url.rstrip("/"), token, ssh_host, int(raw_port))


def proxy_path(raw: str, method: str) -> tuple[str, str]:
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc or parsed.fragment or not parsed.path.startswith("/api/"):
        raise ValueError("Invalid API path")
    path = parsed.path[4:]
    if not any(re.fullmatch(pattern, path) and method in methods for pattern, methods in ROUTES):
        raise ValueError("API path or method is not allowed")
    pairs = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    allowed = {}
    if method == "GET" and path.endswith("/graph"):
        allowed = {"limit": 10000, "offset": 2**31 - 1, "node_limit": 10000, "node_offset": 2**31 - 1}
    elif method == "GET" and path.endswith("/documents"):
        allowed = {"limit": 200, "offset": 2**31 - 1}
    seen = set()
    for key, value in pairs:
        if key in seen or key not in allowed or not re.fullmatch(r"[0-9]{1,10}", value):
            raise ValueError("Invalid API query")
        if int(value) > allowed[key] or (key.endswith("limit") and int(value) == 0):
            raise ValueError("Invalid API query limit")
        seen.add(key)
    return path, urlencode(pairs)


def json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


class DevServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def __init__(self, settings: Settings, directory: Path, *, transport=None):
        self.settings = settings
        self.directory = directory.resolve()
        self.frontend_build, self.assets = render_frontend()
        for name, content in self.assets.items():
            path = (self.directory / name).resolve()
            if not path.is_relative_to(self.directory) or not path.is_file() or path.read_bytes() != content:
                raise ConfigError("Frontend directory is not the canonical server build; rebuild it before startup")
        self.csrf = secrets.token_urlsafe(32)
        self.transport = transport
        super().__init__(("127.0.0.1", settings.port), Handler)
        self.origin = f"http://127.0.0.1:{self.server_port}"
        self.entry_url = f"{self.origin}/index.html?build={self.frontend_build}"

    def forward(self, method, path, query="", body=None, headers=None):
        outgoing = {**(headers or {}), "Authorization": "Bearer " + self.settings.token, "Accept": "application/json"}
        url = self.settings.api_url + path + ("?" + query if query else "")
        try:
            with httpx.Client(transport=self.transport, trust_env=False, follow_redirects=False,
                              timeout=httpx.Timeout(30, connect=3)) as client:
                with client.stream(method, url, content=body, headers=outgoing) as response:
                    if 300 <= response.status_code < 400:
                        return 502, {"detail": "Backend redirects are not supported; check SCHAIN_API_URL"}
                    if response.status_code >= 500:
                        return 502, {"detail": "Backend unavailable; check the SSH tunnel and backend service"}
                    if response.status_code in (401, 403):
                        return response.status_code, {"detail": "Backend authentication failed; check .env.local and restart the local frontend"}
                    content = bytearray()
                    for chunk in response.iter_bytes():
                        content.extend(chunk)
                        if len(content) > MAX_RESPONSE:
                            return 502, {"detail": "Backend response exceeds the local proxy limit"}
                    value = json.loads(content)
                    # Backend validation errors can echo supplied values; never echo our credential.
                    safe = json_bytes(value).decode("utf-8").replace(self.settings.token, "[redacted]")
                    return response.status_code, json.loads(safe)
        except httpx.TimeoutException:
            return 504, {"detail": "Backend timed out; check the SSH tunnel and backend service"}
        except httpx.HTTPError:
            return 502, {"detail": "Cannot reach the backend; check the SSH tunnel and backend service"}
        except (ValueError, UnicodeError):
            return 502, {"detail": "Backend returned an invalid JSON response"}


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format, *args):
        logger.warning("Local frontend protocol error")

    def log_request(self, code="-", size="-"):
        method = self.command if self.command in ("GET", "POST", "PUT", "OPTIONS", "DELETE", "PATCH") else "unsupported"
        logger.info("Local frontend method=%s status=%s", method, code)

    def reply(self, status, value, content_type="application/json; charset=utf-8"):
        content = value if isinstance(value, bytes) else json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-SChain-Frontend", f"projects-{self.server.frontend_build}")
        self.send_header("X-SChain-Mode", "server")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("Local frontend client disconnected")

    def allowed_origin(self):
        hosts = self.headers.get_all("Host", [])
        if hosts != [urlsplit(self.server.origin).netloc]:
            return False
        origins = self.headers.get_all("Origin", [])
        if len(origins) > 1:
            return False
        origin = origins[0] if origins else None
        if origin is not None and origin != self.server.origin:
            return False
        return self.headers.get("Sec-Fetch-Site", "same-origin") in ("same-origin", "none")

    def serve(self):
        if not self.allowed_origin():
            self.reply(403, {"detail": "Use the local frontend URL and same-origin requests"})
            return
        parsed = urlsplit(self.path)
        if self.command == "GET" and self.path == "/schain-dev-config":
            self.reply(200, {"mode": "local-proxy", "application": "schain-projects",
                             "frontend_build": self.server.frontend_build,
                             "entry_point": "/index.html?build=" + self.server.frontend_build,
                             "api_prefix": "/api", "csrf_token": self.server.csrf})
            return
        if parsed.path.startswith("/api/"):
            self.proxy()
            return
        if self.command != "GET" or parsed.scheme or parsed.netloc or parsed.fragment:
            self.reply(404, {"detail": "Not found"})
            return
        name = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
        if name not in STATIC_FILES or parsed.path.startswith("//"):
            self.reply(404, {"detail": "Not found"})
            return
        try:
            query = dict(parse_qsl(parsed.query, strict_parsing=True))
        except ValueError:
            self.reply(400, {"detail": "Invalid frontend query"})
            return
        if any(query.get(key, self.server.frontend_build) != self.server.frontend_build for key in ("v", "build")):
            self.reply(409, {"detail": "Frontend build changed; reopen the entry URL printed by the development server"})
            return
        content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        self.reply(200, self.server.assets[name], content_type)

    def proxy(self):
        if not hmac.compare_digest(self.headers.get("X-SChain-CSRF", "").encode("utf-8"), self.server.csrf.encode("ascii")):
            self.reply(403, {"detail": "Local session expired; reconnect to the local frontend"})
            return
        if self.command != "GET" and self.headers.get("Origin") != self.server.origin:
            self.reply(403, {"detail": "State-changing requests require the local frontend origin"})
            return
        try:
            path, query = proxy_path(self.path, self.command)
        except ValueError:
            self.reply(404, {"detail": "API path, query or method is not allowed"})
            return
        headers = {}
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) > 1 or (lengths and not re.fullmatch(r"[0-9]{1,10}", lengths[0])):
            self.reply(400, {"detail": "Invalid request framing"})
            return
        length = int(lengths[0]) if lengths else 0
        upload = self.command == "POST" and path.endswith("/documents")
        if length > (MAX_UPLOAD if upload else MAX_JSON):
            self.reply(413, {"detail": "Request body exceeds the local proxy limit"})
            return
        if self.command == "GET" and length:
            self.reply(400, {"detail": "GET requests cannot contain a body"})
            return
        if upload:
            filename = self.headers.get("X-Filename", "")
            if not filename or not filename.isascii() or len(filename) > 2400 or any(ord(char) < 32 for char in filename):
                self.reply(422, {"detail": "A valid filename is required"})
                return
            headers["X-Filename"] = filename
        if length:
            expected = "application/octet-stream" if upload else "application/json"
            if self.headers.get("Content-Type", "").split(";")[0].strip() != expected:
                self.reply(415, {"detail": "Unsupported request content type"})
                return
            headers["Content-Type"] = expected
        try:
            body = self.rfile.read(length) if length else None
        except TimeoutError:
            self.reply(408, {"detail": "Request body timed out"})
            return
        if body is not None and len(body) != length:
            self.reply(400, {"detail": "Incomplete request body"})
            return
        status, value = self.server.forward(self.command, path, query, body, headers)
        self.reply(status, value)

    do_GET = serve
    do_POST = serve
    do_PUT = serve
    do_OPTIONS = serve
    do_DELETE = serve
    do_PATCH = serve


def describe_listener(port, *, transport=None):
    try:
        with httpx.Client(transport=transport, trust_env=False, follow_redirects=False, timeout=1) as client:
            response = client.get(f"http://127.0.0.1:{port}/schain-dev-config")
            data = response.json()
        if response.status_code == 200 and data.get("mode") == "local-proxy":
            revision = data.get("frontend_build")
            label = revision if isinstance(revision, str) and re.fullmatch(r"[a-f0-9]{12}", revision) else "unversioned"
            return f"An existing SChain development server is listening (build {label}); stop it in its original terminal before restarting."
    except (httpx.HTTPError, ValueError, AttributeError):
        pass
    return "Another listener owns this port; inspect its process before stopping it. No process was terminated."


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=Path(".env.local"))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(format="%(levelname)s %(message)s")
    logger.setLevel(logging.INFO)
    try:
        settings = load_settings(args.env)
    except ConfigError as error:
        parser.error(str(error))
    directory = ROOT / "outputs/server-frontend"
    if args.env.resolve().is_relative_to(directory.resolve()):
        parser.error("Keep the private environment file outside generated frontend output")
    directory = build(directory)
    try:
        server = DevServer(settings, directory)
    except OSError:
        parser.error(f"Local port {settings.port} is unavailable. {describe_listener(settings.port)}")
    except ConfigError as error:
        parser.error(str(error))
    with server:
        status, _ = server.forward("GET", "/projects")
        print(f"Development frontend: {server.entry_url}")
        print(f"Server mode | build {server.frontend_build} | assets {server.directory}")
        if status != 200:
            if status in (401, 403):
                print("Backend authentication failed; check .env.local and restart the local frontend.")
            else:
                print(f"Backend unavailable (HTTP {status}); check the SSH tunnel and backend service.")
            if settings.ssh_host:
                print(f"Check the tunnel managed by your SSH client using alias {settings.ssh_host}.")
            print("The local page remains available; use Reconnect after restoring the connection.")
        if not args.no_browser:
            webbrowser.open(server.entry_url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
