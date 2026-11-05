"""Authenticated loopback API; long-running work is submitted to the worker."""
import hmac
import os
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import DocumentResponse, GraphResponse, ProjectResponse, ProjectSettings
from .projects import Conflict, ProjectNotFound, Projects
from .storage import connect, read_graph

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def create_app(data_dir=None, token=None, config_path=None):
    data_dir = data_dir or os.environ.get("SCHAIN_DATA_DIR")
    token = token or os.environ.get("SCHAIN_API_TOKEN")
    if not data_dir or not token or len(token) < 32:
        raise ValueError("Set SCHAIN_DATA_DIR and SCHAIN_API_TOKEN (at least 32 characters)")
    config_path = config_path or os.environ.get("SCHAIN_SERVER_CONFIG")
    projects = Projects(Path(data_dir), Path(config_path) if config_path else None)
    app = FastAPI(title="SChain", docs_url=None, redoc_url=None)
    origins = os.environ.get("SCHAIN_CORS_ORIGINS", "http://127.0.0.1:8080").split(",")
    if any(origin == "*" or origin == "null" for origin in origins):
        raise ValueError("Wildcard and null origins are not supported; use the local static server")
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_methods=["GET", "POST", "PUT"],
                       allow_headers=["Authorization", "Content-Type", "X-Filename"])

    def authenticate(authorization: str = Header(default="")):
        if not hmac.compare_digest(authorization.encode(), f"Bearer {token}".encode()):
            raise HTTPException(401, "Invalid bearer token")

    @app.exception_handler(ProjectNotFound)
    async def missing(request, error):
        return JSONResponse(status_code=404, content={"detail": "Unknown project"})

    @app.exception_handler(Conflict)
    async def conflict(request, error):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    auth = [Depends(authenticate)]

    @app.get("/projects", dependencies=auth, response_model=list[ProjectResponse])
    def list_projects():
        return projects.list()

    @app.post("/projects", status_code=201, dependencies=auth, response_model=ProjectResponse)
    def create(settings: ProjectSettings):
        return projects.create(settings)

    @app.get("/projects/{project_id}", dependencies=auth, response_model=ProjectResponse)
    def detail(project_id: str):
        return projects.detail(project_id)

    @app.put("/projects/{project_id}", dependencies=auth, response_model=ProjectResponse)
    def update(project_id: str, settings: ProjectSettings):
        return projects.update(project_id, settings)

    @app.post("/projects/{project_id}/start", status_code=202, dependencies=auth, response_model=ProjectResponse)
    def start(project_id: str):
        return projects.submit(project_id)

    @app.post("/projects/{project_id}/resume", status_code=202, dependencies=auth, response_model=ProjectResponse)
    def resume(project_id: str):
        return projects.resume(project_id)

    @app.post("/projects/{project_id}/pause", dependencies=auth, response_model=ProjectResponse)
    def pause(project_id: str):
        return projects.pause(project_id)

    @app.get("/projects/{project_id}/graph", dependencies=auth, response_model=GraphResponse)
    def graph(project_id: str, limit: int = Query(5000, ge=1, le=10000),
              offset: int = Query(0, ge=0)):
        with connect(projects.database(project_id)) as connection:
            return read_graph(connection, limit, offset)

    @app.get("/projects/{project_id}/documents", dependencies=auth, response_model=list[DocumentResponse])
    def documents(project_id: str, limit: int = Query(50, ge=1, le=200),
                  offset: int = Query(0, ge=0)):
        with connect(projects.database(project_id)) as connection:
            return [dict(row) for row in connection.execute(
                "SELECT doc_id,source_url,status,error,created_at FROM documents "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
            )]

    @app.post("/projects/{project_id}/documents", status_code=202, dependencies=auth, response_model=ProjectResponse)
    async def upload(project_id: str, request: Request, x_filename: str = Header()):
        if len(x_filename) > 200 or "/" in x_filename or "\\" in x_filename:
            raise HTTPException(422, "A plain filename is required")
        suffix = Path(x_filename).suffix.lower()
        if suffix not in (".txt", ".md", ".pdf"):
            raise HTTPException(415, "Supported documents: UTF-8 .txt, .md and PDF")
        directory = projects.directory(project_id) / "uploads"
        directory.mkdir(mode=0o700, exist_ok=True)
        target = directory / (uuid.uuid4().hex + suffix)
        size = 0
        try:
            with target.open("xb") as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "Document exceeds 10 MiB")
                    handle.write(chunk)
            if size == 0:
                raise HTTPException(422, "Empty document")
            if suffix == ".pdf":
                with target.open("rb") as handle:
                    if handle.read(5) != b"%PDF-":
                        raise HTTPException(422, "Invalid PDF signature")
            else:
                try:
                    content = target.read_text(encoding="utf-8")
                except UnicodeError as error:
                    raise HTTPException(422, "Document must be UTF-8") from error
                if not content.strip() or "\x00" in content:
                    raise HTTPException(422, "Document must contain text")
            return projects.submit(project_id, "ingest", target)
        except BaseException:
            target.unlink(missing_ok=True)
            raise

    return app
