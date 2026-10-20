"""Static notebook safety and portable export checks."""

from __future__ import annotations

import json
from pathlib import Path

from schain.distributed.export import export_distributed_artifacts


ROOT = Path(__file__).resolve().parents[1]


def test_distributed_notebook_is_valid_and_has_no_drive_or_session_lock():
    path = ROOT / "notebooks" / "SChain_Expand_Neon_Distributed.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    source = "\n".join(
        "".join(cell.get("source") or []) for cell in notebook["cells"]
    )
    assert "drive.mount" not in source
    assert "session.lock.json" not in source
    assert "NEON_DATABASE_URL" in source
    assert "run_distributed_expand" in source
    assert "TASKS_THIS_SESSION" in source
    assert "SChain_Expand_Neon_Distributed" in path.name


def test_distributed_export_contains_graph_and_audit(tmp_path):
    class Repository:
        def export_facts(self, _experiment, *, accepted=True):
            if not accepted:
                return [{"candidate_fact_id": "rejected-1", "doc_id": "doc-1"}]
            return [{
                "fact_id": "fact-1", "doc_id": "doc-1",
                "subject_id": "company:a", "subject_name": "A", "subject_type": "Company",
                "relation": "suppliesTo",
                "object_id": "company:b", "object_name": "B", "object_type": "Company",
            }]

        def export_documents(self, _experiment):
            return [{"doc_id": "doc-1", "text": "A supplies B"}]

        def progress(self, _experiment):
            return {"status": "completed", "accepted_facts": 1}

        def experiment_manifest(self, _experiment):
            return {"manifest": {"model": "test"}, "status": "completed"}

        def export_events(self, _experiment):
            return [{"event_type": "experiment_completed"}]

    paths = export_distributed_artifacts(Repository(), "experiment", tmp_path)
    graph = json.loads(Path(paths["graph_json"]).read_text(encoding="utf-8"))
    audit = json.loads(Path(paths["audit_json"]).read_text(encoding="utf-8"))
    assert len(graph["nodes"]) == 2
    assert graph["edges"][0]["relation"] == "suppliesTo"
    assert audit["events"][0]["event_type"] == "experiment_completed"
    assert audit["rejected_facts"][0]["candidate_fact_id"] == "rejected-1"
