"""Adapter from a fenced PostgreSQL task to the existing scientific Expand loop."""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import re
import tempfile
from typing import Any

import pandas as pd

from .. import state
from ..checkpoint import atomic_write_json
from ..documents import append_raw_jsonl, upsert_normalized_parquet
from ..kg.facts import upsert_facts_parquet
from ..kg.validate import persist_rejected
from ..names import mention_key
from ..pipeline import _write_resolution_parquet
from ..sources.base import SearchResult
from ..expansion.loop import run_expand
from .models import TaskLease, TaskResult

logger = logging.getLogger(__name__)


def _safe_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")[:80] or "worker"


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def worker_config(
    base: dict[str, Any], lease: TaskLease, *, worker_id: str,
) -> tuple[dict[str, Any], Path]:
    """Return an isolated, disposable local cache configuration for one lease."""
    config = copy.deepcopy(base)
    configured = (base.get("distributed") or {}).get("local_work_root")
    root = Path(configured) if configured else Path(tempfile.gettempdir()) / "schain-workers"
    workspace = (
        root / _safe_segment(lease.experiment_id) / _safe_segment(worker_id)
        / f"{lease.task_id}-{lease.lease_token}"
    ).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    storage = {
        "raw_documents": workspace / "raw" / "documents.jsonl",
        "normalized_documents": workspace / "processed" / "documents.parquet",
        "extracted_facts": workspace / "processed" / "facts.parquet",
        "extractions": workspace / "processed" / "extractions",
        "search_results": workspace / "processed" / "search_results.parquet",
        "entity_resolution": workspace / "processed" / "entity_resolution.parquet",
        "rejected_facts": workspace / "processed" / "rejected_facts.parquet",
        "edge_support": workspace / "processed" / "edge_support.parquet",
        "verification_evidence": workspace / "processed" / "verification_evidence.parquet",
        "extraction_integrity": workspace / "processed" / "extraction_integrity.json",
        "state_db": workspace / "state" / "state.db",
        "graph_export": workspace / "outputs" / "supply_chain_graph.v1.json",
    }
    config["storage"] = {key: str(path) for key, path in storage.items()}
    config.setdefault("audit", {})["path"] = str(workspace / "audit")
    config.setdefault("checkpoint_store", {})["enabled"] = False
    # Graph persistence is owned by the PostgreSQL outbox, never this local loop.
    config.setdefault("save", {})["enabled"] = False
    return config, workspace


class RetrievalCheckpoint:
    def __init__(self, repository: Any, lease: TaskLease) -> None:
        self.repository = repository
        self.lease = lease

    def load(self, entity_key: str, tier: int) -> list[SearchResult] | None:
        rows = self.repository.load_retrieval_checkpoint(
            self.lease.experiment_id, entity_key, tier,
        )
        if rows is None:
            return None
        fields = set(SearchResult.__dataclass_fields__)
        results: list[SearchResult] = []
        for row in rows:
            result = SearchResult(**{key: value for key, value in row.items() if key in fields})
            for key, value in (row.get("_runtime") or {}).items():
                setattr(result, key, value)
            results.append(result)
        return results

    def save(self, entity_key: str, tier: int, results: list[SearchResult]) -> None:
        rows = []
        runtime_fields = (
            "_source_known", "_source_class", "_source_base_score",
            "_source_effective_score", "_source_group_id", "_discovery_direction",
            "_discovery_rank",
        )
        for result in results:
            row = result.as_row()
            row["_runtime"] = {
                key: getattr(result, key) for key in runtime_fields if hasattr(result, key)
            }
            rows.append(_json_value(row))
        self.repository.save_retrieval_checkpoint(self.lease, rows)


class DocumentCheckpointSink:
    def __init__(self, repository: Any, lease: TaskLease) -> None:
        self.repository = repository
        self.lease = lease

    def __call__(self, stage: str, document: dict[str, Any], payload: dict[str, Any]) -> None:
        extraction = payload if stage == "extracted" else None
        resolution = payload if stage == "resolved" else None
        self.repository.save_document_checkpoint(
            self.lease,
            document=_json_value(document),
            stage=stage,
            extraction=_json_value(extraction) if extraction is not None else None,
            resolution=_json_value(resolution) if resolution is not None else None,
        )


class DistributedExpandProcessor:
    """Process exactly the company represented by a PostgreSQL task lease."""

    def __init__(
        self, repository: Any, config: dict[str, Any], llm: Any,
        ontology: dict[str, Any], *, worker_id: str,
    ) -> None:
        self.repository = repository
        self.config = config
        self.llm = llm
        self.ontology = ontology
        self.worker_id = worker_id

    def __call__(self, lease: TaskLease) -> TaskResult:
        config, workspace = worker_config(self.config, lease, worker_id=self.worker_id)
        connection = state.connect(Path(config["storage"]["state_db"]))
        try:
            state.enqueue_entity(
                connection, lease.entity_name, lease.tier,
                entity_id=lease.entity_key, entity_type="Company", expandable=True,
            )
            self._materialize_aliases(connection, lease)
            self._materialize_documents(config, connection, lease)
            summary = run_expand(
                config, connection, self.llm, self.ontology, None,
                allowed_entity_keys={lease.entity_key}, initialize_seeds=False,
                retrieval_checkpoint=RetrievalCheckpoint(self.repository, lease),
                document_checkpoint_sink=DocumentCheckpointSink(self.repository, lease),
            )
            local = connection.execute(
                "SELECT status,stop_reason FROM frontier WHERE entity_key=?",
                (lease.entity_key,),
            ).fetchone()
            if local is None or local["status"] != state.DONE:
                raise RuntimeError(
                    "company task paused before completion: "
                    + str(local["stop_reason"] if local else summary.get("stop_reason"))
                )
            return self._collect_result(config, connection, lease, summary)
        finally:
            connection.close()

    def _materialize_aliases(self, connection: Any, lease: TaskLease) -> None:
        entries = []
        for alias in self.repository.shared_aliases(lease.experiment_id):
            entries.append({
                "label": alias["entity_type"], "mention": alias["mention"],
                "canonical_id": alias["canonical_id"],
                "canonical_name": alias["canonical_name"],
                "method": alias["method"],
            })
        if entries:
            state.save_resolution(connection, entries)

    def _materialize_documents(
        self, config: dict[str, Any], connection: Any, lease: TaskLease,
    ) -> None:
        for stored in self.repository.task_documents(lease):
            document = dict(stored["document"])
            doc_id, _ = state.register_document(
                connection, document, lease.entity_key, lease.tier,
            )
            append_raw_jsonl(document, Path(config["storage"]["raw_documents"]))
            upsert_normalized_parquet(document, Path(config["storage"]["normalized_documents"]))
            extraction = stored.get("extraction")
            if extraction:
                target = Path(config["storage"]["extractions"]) / f"{doc_id}.json"
                atomic_write_json(target, dict(extraction))
                state.set_document_status(connection, doc_id, state.EXTRACTED)
            resolved = dict(stored.get("resolution") or {})
            if resolved:
                resolution = dict(resolved.get("resolution") or {})
                if resolution:
                    _write_resolution_parquet(config, doc_id, resolution)
                    state.save_resolution(connection, [
                        {
                            "label": entry.get("label") or entry.get("entity_type"),
                            "mention": entry["mention"],
                            "canonical_id": entry["canonical_id"],
                            "canonical_name": entry["canonical_name"],
                            "method": entry.get("method", "deterministic"),
                        }
                        for entry in resolution.values()
                        if entry.get("mention") and (entry.get("label") or entry.get("entity_type"))
                    ])
                accepted = pd.DataFrame(resolved.get("accepted_facts") or [])
                rejected = pd.DataFrame(resolved.get("rejected_facts") or [])
                if not accepted.empty:
                    upsert_facts_parquet(accepted, Path(config["storage"]["extracted_facts"]))
                if not rejected.empty:
                    persist_rejected(config, rejected)
                state.set_document_status(connection, doc_id, state.RESOLVED)
                state.reserve_document(connection, doc_id, lease.entity_key)
                state.finish_document_usage(
                    connection, doc_id, lease.entity_key, completed=True, elapsed_seconds=0,
                )

    def _collect_result(
        self, config: dict[str, Any], connection: Any, lease: TaskLease,
        summary: dict[str, Any],
    ) -> TaskResult:
        facts_path = Path(config["storage"]["extracted_facts"])
        rejected_path = Path(config["storage"]["rejected_facts"])
        accepted = pd.read_parquet(facts_path).to_dict("records") if facts_path.exists() else []
        rejected = pd.read_parquet(rejected_path).to_dict("records") if rejected_path.exists() else []
        normalized_path = Path(config["storage"]["normalized_documents"])
        documents: list[dict[str, Any]] = []
        if normalized_path.exists():
            for document in pd.read_parquet(normalized_path).to_dict("records"):
                doc_id = document["doc_id"]
                extraction_path = Path(config["storage"]["extractions"]) / f"{doc_id}.json"
                documents.append({
                    "document": _json_value(document),
                    "extraction": (
                        json.loads(extraction_path.read_text(encoding="utf-8"))
                        if extraction_path.exists() else None
                    ),
                    "metrics": {},
                })
        aliases = []
        for row in state.load_resolution_cache(connection).values():
            aliases.append({
                "mention_key": row["mention_key"], "entity_type": row["label"],
                "mention": row["mention"], "canonical_id": row["canonical_id"],
                "canonical_name": row["canonical_name"], "method": row["method"],
                "evidence": [],
            })
        discoveries = []
        for row in connection.execute(
            "SELECT * FROM frontier WHERE entity_key<>? ORDER BY tier,entity_key",
            (lease.entity_key,),
        ).fetchall():
            discoveries.append({
                "entity_key": row["entity_key"], "entity_name": row["entity_name"],
                "entity_type": row["entity_type"], "tier": int(row["tier"]),
                "priority": float(row["priority"] or 0),
                "provenance": {"parent": lease.entity_key, "task_id": lease.task_id},
            })
        return TaskResult(
            documents=_json_value(documents),
            accepted_facts=_json_value(accepted),
            rejected_facts=_json_value(rejected),
            aliases=_json_value(aliases),
            discoveries=_json_value(discoveries),
            metrics=_json_value(summary),
        )
