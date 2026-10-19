"""Run-scoped incremental audit tracing and companion JSON assembly."""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import math
import os
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urlsplit, urlunsplit

from . import __version__
from .checkpoint import atomic_write_json
from .config import REPO_ROOT, resolve_path


_call_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "schain_audit_call_context", default={},
)
_SECRET_KEY = re.compile(
    r"(?:authorization|password|secret|api[_-]?key|access[_-]?token|refresh[_-]?token)", re.I,
)
_SECRET_TEXT = re.compile(
    r"\b(authorization|password|api[_-]?key|access[_-]?token|bearer)\b"
    r"\s*[:=]?\s*[^\s,;]+", re.I,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def _safe_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"}:
            return value
        return urlunsplit(
            (parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", "")
        )
    except ValueError:
        return "<configured endpoint>"


def json_value(value: Any, key: str = "") -> Any:
    """Convert pipeline objects to secret-safe strict JSON values."""
    if _SECRET_KEY.search(key):
        return "<redacted>"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (Path, datetime)):
        return str(value)
    if isinstance(value, str):
        value = _safe_url(value)
        value = re.sub(r"(?<=://)[^/@\s]+@", "<redacted>@", value)
        return _SECRET_TEXT.sub(lambda match: f"{match.group(1)}=<redacted>", value)
    if isinstance(value, dict):
        return {
            str(item_key): json_value(item, str(item_key))
            for item_key, item in value.items()
            if str(item_key) != "_audit_trace"
        }
    if isinstance(value, (list, tuple, set)):
        return [json_value(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return json_value(value.model_dump(mode="json"))
        except TypeError:
            return json_value(value.model_dump())
    if hasattr(value, "item"):
        try:
            return json_value(value.item(), key)
        except (TypeError, ValueError):
            pass
    return str(value)


def sanitized_config(config: dict[str, Any]) -> dict[str, Any]:
    return json_value({key: value for key, value in config.items() if not key.startswith("_")})


def prompt_hash(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(
        json_value(value), ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@contextlib.contextmanager
def llm_call_context(**values: Any) -> Iterator[None]:
    token = _call_context.set({**_call_context.get(), **values})
    try:
        yield
    finally:
        _call_context.reset(token)


def current_llm_context() -> dict[str, Any]:
    return dict(_call_context.get())


def audit_path_for_graph(graph_path: Path) -> Path:
    graph_path = Path(graph_path)
    return graph_path.with_name(graph_path.stem + ".audit.json")


def _git_metadata() -> tuple[Optional[str], Optional[str]]:
    if not (REPO_ROOT / ".git").exists():
        return None, None
    values = []
    for args in (("rev-parse", "HEAD"), ("branch", "--show-current")):
        try:
            values.append(subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                timeout=3, check=True,
            ).stdout.strip() or None)
        except (OSError, subprocess.SubprocessError):
            values.append(None)
    return values[0], values[1]


class RunAuditTrace:
    """Thread-safe run journal plus deterministic final JSON assembler."""

    def __init__(self, *, output_path: Path, graph_path: Path, mode: str,
                 config: dict[str, Any], input_path: Optional[Path],
                 neo4j_enabled: bool, capture_content: bool,
                 run_id: Optional[str] = None) -> None:
        self.enabled = True
        self.capture_content = capture_content
        self.output_path = Path(output_path)
        self.graph_path = Path(graph_path)
        self.journal_path = self.output_path.with_suffix(self.output_path.suffix + ".journal.jsonl")
        self.run_id = run_id or uuid.uuid4().hex
        self.mode = mode
        self._lock = threading.Lock()
        self._sequence = 0
        self._call_sequence = 0
        self._call_ids: dict[tuple[str, str], list[str]] = {}
        self._context_call_ids: dict[tuple[str, str, str, str], list[str]] = {}
        self.started_at = _now()
        self._started_monotonic = time.monotonic()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # A new run owns a fresh journal.  A stale journal can remain after an
        # ungraceful process termination and must never be mixed into this run.
        self.journal_path.unlink(missing_ok=True)
        commit, branch = _git_metadata()
        llm = config.get("llm") or {}
        self.run_metadata = {
            "run_id": self.run_id, "started_at": self.started_at,
            "schain_version": __version__, "git_commit": commit, "git_branch": branch,
            "mode": mode, "model": llm.get("model"),
            "provider": llm.get("provider", "local"),
            "endpoint": _safe_url(str(llm.get("base_url") or "")) or None,
            "inference_parameters": json_value({
                key: llm.get(key) for key in (
                    "temperature", "max_tokens", "json_mode", "structured_output",
                    "enable_thinking", "request_timeout_seconds", "client_max_retries",
                    "rate_limit_max_attempts",
                ) if key in llm
            }),
            "input_paths": [str(input_path)] if input_path is not None else [],
            "graph_output_path": str(self.graph_path),
            "audit_output_path": str(self.output_path),
            "neo4j_enabled": neo4j_enabled, "full_content": capture_content,
        }
        self.configuration = sanitized_config(config)
        self.emit("run_started", stage="run", **self.run_metadata)

    def document(self, document_id: str) -> "AuditTrail":
        return AuditTrail(coordinator=self, document_id=document_id)

    def emit(self, event: str, *, stage: str,
             document_id: Optional[str] = None, **payload: Any) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            record = {
                "event_id": f"event-{self._sequence:08d}", "sequence": self._sequence,
                "timestamp": _now(), "run_id": self.run_id, "event_type": event,
                "pipeline_stage": stage, "document_id": document_id,
                **json_value(payload),
            }
            with self.journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return record

    def start_llm_call(self, *, document_id: str, stage: str,
                       request: dict[str, Any], model: Any,
                       parameters: Any) -> dict[str, Any]:
        context = current_llm_context()
        with self._lock:
            self._call_sequence += 1
            call_id = f"llm-{self._call_sequence:08d}"
            self._call_ids.setdefault((document_id, stage), []).append(call_id)
            for key, value in context.items():
                values = value if isinstance(value, (list, tuple, set)) else [value]
                for item in values:
                    if item is not None:
                        self._context_call_ids.setdefault(
                            (document_id, stage, str(key), str(item)), [],
                        ).append(call_id)
        input_value = request.get("input")
        payload = {
            **context, "llm_call_id": call_id,
            "attempt_number": int(context.get("retry_attempt", 0)) + 1,
            "prompt_template_id": context.get("prompt_template_id", stage),
            "prompt_hash": prompt_hash(input_value), "model": model,
            "inference_parameters": parameters or {},
            "request_constraints": {
                key: request.get("keyword_arguments", {}).get(key)
                for key in ("response_format", "response_schema")
                if request.get("keyword_arguments", {}).get(key) is not None
            },
        }
        if self.capture_content:
            payload["request"] = request
            payload["rendered_prompt" if isinstance(input_value, str) else "rendered_messages"] = input_value
        else:
            payload["content_capture"] = "disabled"
        self.emit("llm_call_started", stage=stage, document_id=document_id, **payload)
        return {"llm_call_id": call_id, "started": time.monotonic()}

    def finish_llm_call(self, *, document_id: str, stage: str,
                        handle: dict[str, Any], response: Any = None,
                        error: Optional[BaseException] = None) -> None:
        payload: dict[str, Any] = {
            **current_llm_context(), "llm_call_id": handle["llm_call_id"],
            "ended_at": _now(),
            "duration_seconds": round(time.monotonic() - handle["started"], 6),
            "status": "error" if error else "success",
            "provider_raw_response": None,
            "provider_raw_response_unavailable_reason": (
                "neo4j-graphrag exposes LLMResponse content, not the underlying HTTP response"
            ),
            "token_usage": None, "finish_reason": None,
            "provider_metadata_unavailable_reason": (
                "not exposed by the configured neo4j-graphrag LLMInterface response"
            ),
        }
        if error is not None:
            payload.update(error_type=type(error).__name__, error=str(error))
        elif response is not None:
            content = getattr(response, "content", None)
            payload["assistant_message_content_sha256"] = prompt_hash(content or "")
            if self.capture_content:
                payload["assistant_message_content"] = content
                payload["response_capture_level"] = "neo4j_graphrag.LLMResponse.content"
            metadata = getattr(response, "response_metadata", None)
            if metadata is not None:
                payload["response_metadata"] = metadata
                payload["provider_metadata_unavailable_reason"] = None
        self.emit("llm_call_finished", stage=stage, document_id=document_id, **payload)

    def call_ids(self, document_id: str, stage: str) -> list[str]:
        with self._lock:
            return list(self._call_ids.get((document_id, stage), []))

    def context_call_ids(
        self, document_id: str, stage: str, key: str, value: Any,
    ) -> list[str]:
        with self._lock:
            return list(self._context_call_ids.get(
                (document_id, stage, str(key), str(value)), [],
            ))

    def _read_events(self) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        with self.journal_path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @staticmethod
    def _payload(event: dict[str, Any]) -> dict[str, Any]:
        fixed = {"event_id", "sequence", "timestamp", "run_id", "event_type",
                 "pipeline_stage", "document_id"}
        return {key: value for key, value in event.items() if key not in fixed}

    def _assemble(self, events: list[dict[str, Any]], *, complete: bool,
                  status: str, summary: Optional[dict[str, Any]],
                  error: Optional[BaseException]) -> dict[str, Any]:
        documents: dict[str, dict[str, Any]] = {}
        chunks: dict[tuple[str, str], dict[str, Any]] = {}
        calls: dict[str, dict[str, Any]] = {}
        categories = {key: [] for key in (
            "extraction_graphs", "integrity_checks", "entity_resolution",
            "candidate_facts", "validation_decisions", "repair_attempts",
            "accepted_facts", "rejected_facts", "relationship_outcomes", "errors",
        )}
        graph_export: dict[str, Any] = {"path": str(self.graph_path), "status": "not_started"}
        for event in events:
            kind, doc_id = event["event_type"], event.get("document_id")
            payload = self._payload(event)
            if doc_id:
                document = documents.setdefault(doc_id, {
                    "document_id": doc_id, "doc_id": doc_id,
                    "lifecycle": [], "errors": [],
                })
                if kind == "document_observed":
                    document.update(payload)
                if kind.startswith("document_"):
                    document["lifecycle"].append({
                        "event": kind, "timestamp": event["timestamp"], **payload,
                    })
                if kind == "document_failed":
                    document["errors"].append(payload)
            chunk_id = event.get("chunk_id")
            if doc_id and chunk_id:
                chunk = chunks.setdefault((doc_id, str(chunk_id)), {
                    "document_id": doc_id, "chunk_id": str(chunk_id), "history": [],
                })
                if kind == "chunk_discovered":
                    chunk.update(payload)
                chunk["history"].append({
                    "event": kind, "timestamp": event["timestamp"], **payload,
                })
            call_id = event.get("llm_call_id")
            if call_id:
                call = calls.setdefault(call_id, {
                    "llm_call_id": call_id, "document_id": doc_id,
                    "pipeline_stage": event["pipeline_stage"],
                })
                if kind == "llm_call_started":
                    call.update(payload)
                    call["started_at"] = event["timestamp"]
                elif kind == "llm_call_finished":
                    call.update(payload)
                else:
                    call.setdefault("related_events", []).append(event["event_id"])
            if kind in {"chunk_extraction_parsed", "parsed_extraction_graph"}:
                categories["extraction_graphs"].append(event)
            if kind in {
                "structural_integrity_checked", "relationship_integrity_result",
                "entity_resolution_integrity_checked",
            } or kind.startswith("endpoint_"):
                categories["integrity_checks"].append(event)
            if kind.startswith("entity_resolution") or kind == "alias_merge_evaluated":
                categories["entity_resolution"].append(event)
            if kind == "candidate_fact_created":
                categories["candidate_facts"].append(event)
            if kind in {
                "validation_check", "validation_completed",
                "semantic_validation_batch_response",
                "semantic_validation_targeted_retry",
                "semantic_validation_unresolved",
            }:
                categories["validation_decisions"].append(event)
            if "repair" in kind:
                categories["repair_attempts"].append(event)
            if kind == "candidate_accepted":
                categories["accepted_facts"].append(event)
            if kind == "candidate_rejected":
                categories["rejected_facts"].append(event)
            if (kind.startswith("relationship_") or kind.startswith("endpoint_")
                    or kind == "structural_integrity_checked"):
                categories["relationship_outcomes"].append(event)
            if kind.endswith("failed") or payload.get("status") == "error":
                categories["errors"].append(event)
            if kind == "graph_export_started":
                graph_export = {"status": "started", "started_at": event["timestamp"], **payload}
            elif kind in {"graph_export_completed", "graph_export_failed"}:
                graph_export.update(payload)
                graph_export["status"] = "success" if kind.endswith("completed") else "failed"
                graph_export["ended_at"] = event["timestamp"]
        doc_ids, call_ids = set(documents), set(calls)
        unresolved = []
        for call in calls.values():
            if call.get("document_id") and call["document_id"] not in doc_ids:
                unresolved.append({"type": "document_id", "value": call["document_id"],
                                   "owner": call["llm_call_id"]})
        for item in events:
            linked_ids = (
                list(item.get("related_llm_call_ids") or [])
                + list(item.get("originating_llm_call_ids") or [])
            )
            for linked in linked_ids:
                if linked not in call_ids:
                    unresolved.append({"type": "llm_call_id", "value": linked,
                                       "owner": item["event_id"]})
        relationship_ids = {
            item.get("extraction_relationship_id")
            for item in categories["relationship_outcomes"]
            if item.get("extraction_relationship_id")
        }
        for item in categories["candidate_facts"]:
            linked = item.get("extraction_relationship_id")
            if linked and linked not in relationship_ids:
                unresolved.append({
                    "type": "extraction_relationship_id", "value": linked,
                    "owner": item["event_id"],
                })
        fact_ids = {
            str(value)
            for event in categories["candidate_facts"]
            for value in (event.get("fact_id"), event.get("candidate_fact_id"))
            if value
        }
        final_fact_ids = {
            str(item.get("fact_id")) for item in categories["accepted_facts"]
            if item.get("fact_id")
        }
        for mapping in graph_export.get("edge_fact_map") or []:
            for linked in mapping.get("fact_ids") or []:
                if str(linked) not in final_fact_ids:
                    unresolved.append({
                        "type": "fact_id", "value": str(linked),
                        "owner": str(mapping.get("edge_id") or "graph_export"),
                    })
        run = {
            **self.run_metadata, "ended_at": _now(),
            "elapsed_seconds": round(time.monotonic() - self._started_monotonic, 6),
            "completion_status": status, "complete": complete,
            "error": ({"type": type(error).__name__, "message": str(error)} if error else None),
        }
        result = {
            "audit_schema_version": "1.0", "run": run,
            "configuration": self.configuration,
            "documents": list(documents.values()), "chunks": list(chunks.values()),
            "llm_calls": list(calls.values()), **categories,
            "graph_export": graph_export, "events": events,
            "summary": {
                "pipeline_summary": summary or {},
                "counts": {
                    "documents": len(documents), "chunks": len(chunks),
                    "llm_calls": len(calls),
                    "candidate_facts": len(categories["candidate_facts"]),
                    "accepted_facts": len(categories["accepted_facts"]),
                    "rejected_facts": len(categories["rejected_facts"]),
                    "errors": len(categories["errors"]),
                },
                "known_document_ids": sorted(doc_ids), "known_llm_call_ids": sorted(call_ids),
                "known_fact_ids": sorted(fact_ids),
                "reference_integrity": {
                    "valid": not unresolved, "unresolved_references": unresolved,
                    "unavailable_references_are_explicit": True,
                },
                "relationship_loss_observability": {
                    "raw_response_to_parser_exclusions": "not_observable",
                    "reason": (
                        "neo4j-graphrag returns the parsed Neo4jGraph but does not "
                        "expose a field-level parser rejection ledger"
                    ),
                    "parsed_structural_losses": "relationship_outcomes",
                    "unresolved_endpoint_losses": "relationship_outcomes",
                    "candidate_rejections": "rejected_facts",
                    "accepted_export_inclusion": "graph_export.edge_fact_map",
                    "accepted_export_omissions": "graph_export.accepted_facts_omitted",
                },
            },
        }
        return json_value(result)

    def finalize(self, *, complete: bool, status: str,
                 summary: Optional[dict[str, Any]] = None,
                 error: Optional[BaseException] = None) -> Path:
        self.emit("run_finished" if complete else "run_failed", stage="run",
                  status=status, complete=complete,
                  error_type=type(error).__name__ if error else None,
                  error=str(error) if error else None)
        payload = self._assemble(self._read_events(), complete=complete, status=status,
                                 summary=summary, error=error)
        atomic_write_json(self.output_path, payload)
        self.journal_path.unlink(missing_ok=True)
        return self.output_path


class AuditTrail:
    """Document view over a run trace, or a disabled no-op object."""

    def __init__(self, *, coordinator: Optional[RunAuditTrace], document_id: str,
                 enabled: bool = False, path: Optional[Path] = None,
                 run_id: str = "disabled", capture_content: bool = False) -> None:
        self.coordinator, self.document_id = coordinator, document_id
        self.enabled = coordinator is not None or enabled
        self.capture_content = coordinator.capture_content if coordinator else capture_content
        self.run_id = coordinator.run_id if coordinator else run_id
        self.path = path
        self._sequence = 0
        self._lock = threading.Lock()

    def emit(self, event: str, *, stage: str, **payload: Any) -> None:
        if self.coordinator:
            self.coordinator.emit(event, stage=stage, document_id=self.document_id, **payload)
        elif self.enabled and self.path is not None:
            with self._lock:
                self._sequence += 1
                record = {
                    "schema_version": 1, "run_id": self.run_id,
                    "document_id": self.document_id, "sequence": self._sequence,
                    "timestamp": _now(), "pipeline_stage": stage, "event": event,
                    **json_value(payload),
                }
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def llm_started(self, *, stage: str, request: dict[str, Any],
                    model: Any, parameters: Any) -> Optional[dict[str, Any]]:
        if self.coordinator:
            return self.coordinator.start_llm_call(
                document_id=self.document_id, stage=stage, request=request,
                model=model, parameters=parameters,
            )
        if not self.enabled:
            return None
        handle = {"llm_call_id": f"llm-{uuid.uuid4().hex}", "started": time.monotonic()}
        input_value = request.get("input")
        self.emit(
            "llm_call_started", stage=stage, llm_call_id=handle["llm_call_id"],
            prompt_hash=prompt_hash(input_value), model=model,
            inference_parameters=parameters or {},
            rendered_prompt=input_value if self.capture_content else None,
            request=request if self.capture_content else None,
        )
        return handle

    def llm_finished(self, *, stage: str, started: dict[str, Any],
                     response: Any = None,
                     error: Optional[BaseException] = None) -> None:
        if self.coordinator:
            self.coordinator.finish_llm_call(
                document_id=self.document_id, stage=stage, handle=started,
                response=response, error=error,
            )
        elif self.enabled:
            self.emit(
                "llm_call_finished", stage=stage,
                llm_call_id=started["llm_call_id"],
                duration_seconds=round(time.monotonic() - started["started"], 6),
                status="error" if error else "success",
                error_type=type(error).__name__ if error else None,
                error=str(error) if error else None,
                raw_model_response=(
                    getattr(response, "content", None)
                    if self.capture_content and response is not None else None
                ),
                raw_response_capture=(
                    "provider_response_content"
                    if self.capture_content and response is not None else None
                ),
            )

    def llm_call_ids(self, stage: str) -> list[str]:
        return self.coordinator.call_ids(self.document_id, stage) if self.coordinator else []

    def contextual_llm_call_ids(self, stage: str, key: str, value: Any) -> list[str]:
        return self.coordinator.context_call_ids(
            self.document_id, stage, key, value,
        ) if self.coordinator else []


def audit_for_document(config: dict[str, Any], document_id: str) -> AuditTrail:
    coordinator = config.get("_audit_trace")
    if isinstance(coordinator, RunAuditTrace):
        return AuditTrail(coordinator=coordinator, document_id=document_id)
    audit = config.get("audit") or {}
    enabled = bool(audit.get("enabled", False))
    run_id = str(audit.get("run_id") or config.get("_audit_run_id") or "")
    if enabled and not run_id:
        run_id = uuid.uuid4().hex
        config["_audit_run_id"] = run_id
    root = Path(resolve_path(audit.get("path") or "data/processed/audit"))
    return AuditTrail(
        coordinator=None, document_id=document_id, enabled=enabled,
        path=root / (run_id or "disabled") / f"{document_id}.jsonl",
        run_id=run_id or "disabled",
        capture_content=bool(audit.get("capture_content", False)),
    )


def configure_run_audit(config: dict[str, Any], *, mode: str, graph_path: Path,
                        input_path: Optional[Path], neo4j_enabled: bool) -> Optional[RunAuditTrace]:
    audit_config = config.get("audit") or {}
    enabled = _truthy(os.environ.get("SCHAIN_AUDIT_TRACE")) or bool(audit_config.get("enabled", False))
    if not enabled:
        return None
    capture_content = _truthy(os.environ.get("SCHAIN_AUDIT_FULL_CONTENT")) or bool(
        audit_config.get("capture_content", False)
    )
    trace = RunAuditTrace(
        output_path=audit_path_for_graph(graph_path), graph_path=graph_path,
        mode=mode, config=config, input_path=input_path,
        neo4j_enabled=neo4j_enabled, capture_content=capture_content,
        run_id=str(audit_config.get("run_id") or "") or None,
    )
    config["_audit_trace"] = trace
    config["_audit_run_id"] = trace.run_id
    return trace
