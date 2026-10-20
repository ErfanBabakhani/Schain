"""Portable Graph JSON and distributed Audit JSON exports from PostgreSQL."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from ..checkpoint import atomic_write_json


def _edge_id(subject: str, relation: str, object_: str) -> str:
    raw = f"{subject}|{relation}|{object_}".encode("utf-8")
    return "edge-" + hashlib.sha256(raw).hexdigest()[:20]


def export_distributed_artifacts(
    repository: Any, experiment_id: str, output_directory: str | Path,
) -> dict[str, str]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    facts = repository.export_facts(experiment_id, accepted=True)
    rejected = repository.export_facts(experiment_id, accepted=False)
    documents = repository.export_documents(experiment_id)
    progress = repository.progress(experiment_id)
    manifest = repository.experiment_manifest(experiment_id)
    events = repository.export_events(experiment_id)
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    for fact in facts:
        for side in ("subject", "object"):
            entity_id = fact.get(f"{side}_id")
            if entity_id:
                nodes[str(entity_id)] = {
                    "id": entity_id,
                    "label": fact.get(f"{side}_name") or entity_id,
                    "type": fact.get(f"{side}_type"),
                }
        subject = str(fact.get("subject_id") or "")
        relation = str(fact.get("relation") or "")
        object_ = str(fact.get("object_id") or "")
        if not subject or not relation or not object_:
            continue
        edge_id = _edge_id(subject, relation, object_)
        edge = edges.setdefault(edge_id, {
            "id": edge_id, "source": subject, "target": object_,
            "relation": relation, "facts": [],
        })
        edge["facts"].append(fact)
    generated_at = datetime.now(timezone.utc).isoformat()
    graph_path = output / f"{experiment_id}.graph.json"
    audit_path = output / f"{experiment_id}.audit.json"
    atomic_write_json(graph_path, {
        "schema_version": "distributed-1.0",
        "experiment_id": experiment_id,
        "generated_at": generated_at,
        "nodes": list(nodes.values()), "edges": list(edges.values()),
        "documents": documents, "statistics": progress,
    })
    atomic_write_json(audit_path, {
        "schema_version": "distributed-audit-1.0",
        "experiment_id": experiment_id,
        "generated_at": generated_at,
        "experiment": manifest, "progress": progress,
        "accepted_facts": facts, "rejected_facts": rejected,
        "events": events,
    })
    return {"graph_json": str(graph_path), "audit_json": str(audit_path)}
