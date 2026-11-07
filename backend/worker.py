"""Dedicated serial worker. Run under the institution's approved service/job manager."""
import argparse
import json
import logging
import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path

from .models import ProjectSettings

logger = logging.getLogger(__name__)


@contextmanager
def worker_lock(root):
    """Kernel-owned lock: process death releases ownership without lease guessing."""
    handle = (root / "worker.lock").open("a+b")
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise RuntimeError("Another worker owns this runtime directory") from error
    try:
        yield
    finally:
        handle.close()


class Worker:
    def __init__(self, projects, llm_factory=None):
        self.projects = projects
        if llm_factory is None:
            from schain.llm.provider import build_llm
            llm_factory = build_llm
        self.llm_factory = llm_factory
        self.stopping = False

    def recover(self):
        from .activity import finish
        from .storage import connect
        # Caller holds the global kernel lock; no live job can own these rows.
        for project_id in self.projects.identifiers():
            with connect(self.projects.database(project_id)) as connection:
                interrupted = list(connection.execute(
                    "SELECT DISTINCT r.id,r.status FROM runs r LEFT JOIN live_documents d ON d.run_id=r.id "
                    "WHERE r.status IN ('running','pausing') OR d.stage NOT IN "
                    "('resolved','failed','paused','completed','interrupted')"
                ))
                with connection:
                    connection.execute("UPDATE runs SET status='queued' WHERE status='running'")
            for run_id, status in interrupted:
                finish(self.projects.database(project_id), run_id,
                       "interrupted" if status in ('running', 'pausing', 'queued') else status)

    def tick(self):
        from .storage import connect
        for project_id in self.projects.identifiers():
            with connect(self.projects.database(project_id)) as connection:
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    row = connection.execute("SELECT * FROM runs WHERE status IN ('queued','pausing') ORDER BY rowid LIMIT 1").fetchone()
                    if row is None:
                        continue
                    if row["status"] == "queued":
                        connection.execute("UPDATE runs SET status='running' WHERE id=?", (row["id"],))
                self.execute(project_id, dict(row))
                return True
        return False

    def execute(self, project_id, run):
        from .activity import finish, record
        import pandas as pd
        from schain import state
        from schain.documents import load_document
        from schain.expansion.frontier import discover_upstream
        from schain.expansion.loop import run_expand
        from schain.graph_export import build_graph_bundle
        from schain.kg.support import refresh_edge_support
        from schain.ontology import load_ontology
        from schain.pipeline import process_document
        from schain.sqlite_graph import restore_artifacts
        from .storage import connect, dumps, save_graph
        path = self.projects.database(project_id)
        with connect(path) as connection:
            project = connection.execute("SELECT * FROM project").fetchone()
        config = json.loads(project["config"])
        settings = ProjectSettings.model_validate_json(project["settings"])
        connection = state.connect(path, journal_mode="DELETE")

        def pause_requested():
            with connect(path) as control:
                status = control.execute("SELECT status FROM runs WHERE id=?", (run["id"],)).fetchone()[0]
            return self.stopping or status == "pausing"

        def publish(resolved_document=None):
            facts_path = Path(config["storage"]["extracted_facts"])
            facts = pd.read_parquet(facts_path) if facts_path.exists() else pd.DataFrame()
            # A manual document is explicitly associated only with witnessed customers
            # already on the frontier; the shared grounding/direction gate still applies.
            if not facts.empty:
                parents = {r["entity_key"]: r["tier"] for r in connection.execute("SELECT * FROM frontier")}
                for fact in facts.to_dict(orient="records"):
                    if fact["relation"] == "suppliesTo" and fact["object_id"] in parents:
                        state.add_document_origin(connection, fact["doc_id"], fact["object_id"], parents[fact["object_id"]])
            refresh_edge_support(config)
            # Shared frontier policy handles manual documents and crash recovery too.
            ontology = load_ontology(config)
            for tier in range(settings.limits.max_tier):
                for parent in connection.execute("SELECT * FROM frontier WHERE tier=?", (tier,)).fetchall():
                    discover_upstream(connection, config, ontology, parent["entity_name"],
                                      parent["entity_key"], tier, pd.DataFrame())
            save_graph(connection, build_graph_bundle(config), resolved_document=resolved_document)

        def checkpoint(stage, document, payload):
            if stage == "resolved":
                publish(document["doc_id"])

        def progress(stage, document, payload):
            record(path, run["id"], stage, document, payload)

        final_status, summary, error_text = "failed", None, None
        try:
            record(path, run["id"], "run_running")
            restore_artifacts(config, connection)
            publish()  # Repair a crash between core checkpoint and graph publication.
            if pause_requested():
                final_status, summary = "paused", {"stop_reason": "pause_requested"}
                return
            used = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            remaining = settings.limits.max_total_documents - used
            incomplete = connection.execute(
                "SELECT 1 FROM documents WHERE status NOT IN ('RESOLVED','COMMITTED') LIMIT 1"
            ).fetchone()
            manual_document = load_document(Path(run["input_path"])) if run["kind"] == "ingest" else None
            if manual_document:
                manual_document["title"] = run.get("input_name") or Path(run["input_path"]).name
            duplicate = manual_document and connection.execute(
                "SELECT 1 FROM documents WHERE content_hash=?", (manual_document["content_hash"],)
            ).fetchone()
            if remaining <= 0 and not duplicate and (run["kind"] == "ingest" or not incomplete):
                final_status, summary = "paused", {"stop_reason": "total_document_budget"}
            elif pause_requested():
                final_status, summary = "paused", {"stop_reason": "pause_requested"}
            else:
                llm = self.llm_factory(config)
                ontology = load_ontology(config)
                if run["kind"] == "ingest":
                    config["_runtime_deadline_monotonic"] = time.monotonic() + settings.limits.max_minutes * 60
                    summary = process_document(manual_document, config, connection, llm, ontology,
                                               checkpoint_sink=checkpoint, progress_sink=progress)
                    final_status = "completed"
                else:
                    # Core runtime is cumulative. A session adds its allowance to persisted time.
                    elapsed = float(state.run_value(connection, "elapsed_seconds", 0))
                    config["runtime"]["max_minutes"] += elapsed / 60
                    summary = run_expand(config, connection, llm, ontology,
                                         document_checkpoint_sink=checkpoint, should_pause=pause_requested,
                                         progress_sink=progress,
                                         max_total_documents=settings.limits.max_total_documents)
                    if summary["completion"]["failed_frontier"] or summary["documents_failed"]:
                        final_status = "failed"
                        error_text = "Expansion has failed documents or frontier entities; inspect project errors"
                    else:
                        final_status = "completed" if summary["completion"]["complete"] else "paused"
                publish()
        except Exception as error:  # Job boundary: persist failure and keep other projects available.
            logger.exception("Job %s failed", run["id"])
            error_text = f"{type(error).__name__}: {error}"
        finally:
            with connection:
                connection.execute("UPDATE runs SET status=?,finished_at=?,summary=?,error=? WHERE id=?",
                                   (final_status, state._now(), dumps(summary), error_text, run["id"]))
            connection.close()
            finish(path, run["id"], final_status)
            logger.info("job=%s project=%s status=%s", run["id"], project_id, final_status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("SCHAIN_DATA_DIR"), required=not os.environ.get("SCHAIN_DATA_DIR"))
    parser.add_argument("--once", action="store_true", help="Process at most one queued job")
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(args.cpu_threads)
    logging.basicConfig(level=logging.INFO)
    from .projects import Projects
    projects = Projects(args.data_dir)
    worker = Worker(projects)
    def stop(signum, frame):
        worker.stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with worker_lock(projects.root):
        worker.recover()
        while not worker.stopping:
            worked = worker.tick()
            if args.once:
                break
            if not worked:
                time.sleep(2)


if __name__ == "__main__":
    main()
