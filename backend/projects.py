"""Project isolation and transactional job submission, independent of HTTP."""
import copy
import json
import re
import uuid
from pathlib import Path

from schain.config import load_config, REPO_ROOT
from schain.expansion.frontier import seed_frontier
from schain.graph_export import build_graph_bundle
from schain.ontology import load_ontology
from schain.state import _now

from .models import ProjectSettings
from .storage import connect, dumps, initialize, save_graph

class Conflict(ValueError):
    pass


class ProjectNotFound(KeyError):
    pass


class Projects:
    def __init__(self, root: Path, config_path: Path | None = None):
        self.root = root.resolve()
        if self.root == REPO_ROOT or REPO_ROOT in self.root.parents:
            raise ValueError("Server runtime data must be outside the source repository")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.config_path = config_path or REPO_ROOT / "config/expansion.yaml"
        for directory in self.root.iterdir():
            if re.fullmatch(r"[0-9a-f]{32}", directory.name) and not directory.is_symlink() and (directory / "state.db").is_file():
                initialize(directory / "state.db")

    def directory(self, project_id):
        if not re.fullmatch(r"[0-9a-f]{32}", project_id):
            raise ProjectNotFound("Unknown project")
        path = self.root / project_id
        if path.is_symlink() or not (path / "state.db").is_file():
            raise ProjectNotFound("Unknown project")
        return path

    def database(self, project_id):
        return self.directory(project_id) / "state.db"

    def create(self, settings: ProjectSettings):
        project_id = uuid.uuid4().hex
        directory = self.root / project_id
        directory.mkdir(mode=0o700)
        config = load_config(self.config_path)
        config["storage"] = {
            key: str(directory / "artifacts" / Path(value).name)
            for key, value in config["storage"].items()
        }
        config["storage"].update(state_db=str(directory / "state.db"), logs=str(directory / "logs"))
        config["storage"]["sqlite_graph"] = True
        config["state"] = {"backend": "sqlite"}
        config["save"] = {"enabled": False}
        config["checkpoint_store"] = {"enabled": False}
        config["distributed"] = {"enabled": False}
        config["runtime"].update(resume=True, restart_from_scratch=False)
        config["audit"] = {"enabled": False}
        config = self.apply_settings(config, settings)
        initialize(directory / "state.db")
        with connect(directory / "state.db") as connection:
            with connection:
                connection.execute("INSERT INTO project VALUES(?,?,?)", (
                    project_id, settings.model_dump_json(), dumps(config)
                ))
            seed_frontier(connection, config, load_ontology(config))
            save_graph(connection, build_graph_bundle(config))
        return self.detail(project_id)

    @staticmethod
    def apply_settings(config, settings):
        config = copy.deepcopy(config)
        limits = settings.limits
        config["seeds"] = settings.seeds
        expansion = config["expansion"]
        expansion.pop("max_documents", None)
        expansion.update(max_tier=limits.max_tier, max_documents_per_run=limits.max_documents_per_run)
        retrieval = expansion.setdefault("retrieval", {})
        retrieval.update(max_documents_per_entity=limits.max_documents_per_entity,
                         target_documents_per_entity=limits.max_documents_per_entity)
        retrieval.pop("max_documents_by_tier", None)
        retrieval.pop("target_documents_by_tier", None)
        config["concurrency"].update(retrieval=limits.retrieval_concurrency,
                                     llm_max_concurrency=limits.llm_max_concurrency)
        config["runtime"]["max_minutes"] = limits.max_minutes
        return config

    def identifiers(self):
        return [p.name for p in sorted(self.root.iterdir())
                if re.fullmatch(r"[0-9a-f]{32}", p.name) and not p.is_symlink() and (p / "state.db").is_file()]

    def list(self):
        return [self.detail(project_id) for project_id in self.identifiers()]

    def detail(self, project_id):
        with connect(self.database(project_id)) as connection:
            row = connection.execute("SELECT * FROM project").fetchone()
            runs = [dict(r) for r in connection.execute(
                "SELECT * FROM runs ORDER BY rowid DESC LIMIT 20"
            )]
            counts = {r[0]: r[1] for r in connection.execute(
                "SELECT status,COUNT(*) FROM documents GROUP BY status"
            )}
            errors = [dict(r) for r in connection.execute(
                "SELECT doc_id,status,error FROM documents WHERE error IS NOT NULL "
                "ORDER BY updated_at DESC LIMIT 20"
            )]
            errors.extend(dict(r) for r in connection.execute(
                "SELECT entity_key,status,error FROM frontier WHERE error IS NOT NULL "
                "ORDER BY updated_at DESC LIMIT 20"
            ))
            graph_counts = {key: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                            for key, table in (("nodes", "graph_nodes"), ("edges", "graph_edges"))}
            revision = connection.execute("SELECT payload FROM graph_meta WHERE key='revision'").fetchone()
        for run in runs:
            run["summary"] = json.loads(run["summary"]) if run["summary"] else None
        return {"id": project_id, **json.loads(row["settings"]),
                "status": runs[0]["status"] if runs else "ready", "runs": runs,
                "statistics": {**graph_counts, "documents": counts}, "errors": errors,
                "graph_revision": json.loads(revision[0]) if revision else None}

    @staticmethod
    def require_idle(connection):
        if connection.execute("SELECT 1 FROM runs WHERE status IN ('queued','running','pausing')").fetchone():
            raise Conflict("Project has an active job; pause it and wait for the worker to finish")

    def update(self, project_id, settings):
        with connect(self.database(project_id)) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                self.require_idle(connection)
                row = connection.execute("SELECT * FROM project").fetchone()
                old = ProjectSettings.model_validate_json(row["settings"])
                if connection.execute("SELECT 1 FROM runs LIMIT 1").fetchone() and (
                    old.seeds != settings.seeds or old.limits.max_tier != settings.limits.max_tier
                ):
                    raise Conflict("Seeds and depth are immutable after the first job; create a new project")
                config = self.apply_settings(json.loads(row["config"]), settings)
                connection.execute("UPDATE project SET settings=?,config=?", (
                    settings.model_dump_json(), dumps(config)
                ))
                if old.seeds != settings.seeds or old.limits.max_tier != settings.limits.max_tier:
                    connection.execute("DELETE FROM frontier")
                    seed_frontier(connection, config, load_ontology(config), commit=False)
                    save_graph(connection, build_graph_bundle(config))
        return self.detail(project_id)

    def submit(self, project_id, kind="expand", input_path=None):
        with connect(self.database(project_id)) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                self.require_idle(connection)
                run_id = uuid.uuid4().hex
                connection.execute(
                    "INSERT INTO runs(id,kind,status,input_path,created_at) VALUES(?,?,'queued',?,?)",
                    (run_id, kind, str(input_path) if input_path else None, _now())
                )
        return self.detail(project_id)

    def pause(self, project_id):
        with connect(self.database(project_id)) as connection:
            with connection:
                connection.execute("UPDATE runs SET status='paused',finished_at=? WHERE status='queued'", (_now(),))
                connection.execute("UPDATE runs SET status='pausing' WHERE status='running'")
        return self.detail(project_id)

    def resume(self, project_id):
        with connect(self.database(project_id)) as connection:
            last = connection.execute("SELECT * FROM runs ORDER BY rowid DESC LIMIT 1").fetchone()
        if last and last["kind"] == "ingest" and last["status"] in ("failed", "paused"):
            return self.submit(project_id, "ingest", last["input_path"])
        return self.submit(project_id)
