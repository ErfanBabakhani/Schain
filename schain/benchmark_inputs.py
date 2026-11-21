"""Versioned input archives for the selection and expansion comparisons."""

import base64
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import closing
import time

import pyarrow as pa
from pyarrow.parquet import ParquetFile

from . import state
from .checkpoint import atomic_write_json
from .expansion.frontier import seed_frontier
from .expansion.loop import _ESCALATION_LADDER
from .run_config import _safe
from .sources.base import SearchResult
from .sources.retrieve import fetch_selected_full_text, retrieve_for_entity

POLICIES = ("legacy", "adaptive_linucb")
MODES = ("selection-only", "full-pipeline")
PRIMARY = tuple(_ESCALATION_LADDER[0][1])
RESULT_FIELDS = {field.name for field in fields(SearchResult)}


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def result_key(result):
    return digest([result.provider, result.query, result.url, result.search_id])


def result_row(result):
    return {
        "result": result.as_row(),
        "content_sha256": digest({
            name: getattr(result, name)
            for name in ("title", "snippet", "text", "cached_full_text", "fetched_text")
        }),
    }


def restore_result(row):
    result = SearchResult(**{key: value for key, value in row["result"].items() if key in RESULT_FIELDS})
    if result_row(result)["content_sha256"] != row["content_sha256"]:
        raise ValueError(f"invalid frozen content hash for {result.url}")
    return result


def entry_key(profile, entity_key, tier, stages):
    return digest([profile, entity_key, tier, list(stages)])


def recording_config(config, directory):
    return {
        **config,
        "runtime": {**config.get("runtime", {}), "resume": False},
        "storage": {
            "state_db": str(directory / "state.db"),
            "search_results": str(directory / "search_results.parquet"),
        },
    }


def record_dataset(path, mode, configs, targets, inventory, version, *,
                   initial_facts=None, allow_search_errors=False, provenance=None):
    """Run shared searches explicitly. No LLM or model initialization occurs here."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    initial = None
    if initial_facts:
        data = initial_facts.read_bytes()
        validate_initial_parquet(data)
        initial = {"parquet_base64": base64.b64encode(data).decode("ascii"),
                   "sha256": hashlib.sha256(data).hexdigest()}
    entries = {}
    downloads_by_source = {}
    legacy = configs["legacy"]["expansion"]
    can_share_metadata = (
        legacy["selection"].get("candidate_pool_policy") == "title_snippet"
        and not legacy["retrieval"].get("max_candidates_per_domain")
    )
    profiles = POLICIES if mode == "full-pipeline" else ("shared",)
    for profile in profiles:
        config = configs["legacy" if profile == "shared" else profile]
        stage_sets = [PRIMARY] if profile == "shared" else [stages for _, stages in _ESCALATION_LADDER]
        items = targets if profile == "shared" else inventory
        for name, entity_key, tier in items:
            for stages in stage_sets:
                key = entry_key(profile, entity_key, tier, stages)
                shared = entries.get(entry_key("legacy", entity_key, tier, stages)) if can_share_metadata else None
                with TemporaryDirectory(prefix="schain-record-", dir=path.parent) as scratch:
                    record_config = recording_config(config, Path(scratch))
                    if shared is not None:
                        captured = {field: shared[field] for field in ("query_plan", "searches", "provider_results")}
                        results = [restore_result(row) for row in shared["candidates"]]
                    else:
                        captured = {}

                        def capture(plan, searches, provider_rows, _selected):
                            captured.update(query_plan=plan, searches=searches, provider_results=[
                                result_row(SearchResult(**{
                                    key: value for key, value in row.items() if key in RESULT_FIELDS
                                })) for row in provider_rows
                            ])

                        with closing(state.connect(Path(scratch) / "state.db")) as db:
                            results = retrieve_for_entity(
                                record_config, name, entity_key, tier, db,
                                escalation_stages=list(stages), record_snapshot=capture,
                            )
                        failures = [row for row in captured["searches"] if row["status"] == state.FAILED]
                        if failures and not allow_search_errors:
                            first = failures[0]
                            raise RuntimeError(f"recording failed for {name}: {first['provider']} "
                                               f"{first['query']!r}: {first['error']}")
                    downloads = {}
                    if profile == "adaptive_linucb":
                        for result in results:
                            # Discovery rank can vary by stage, but the download inputs cannot.
                            source = digest([result.provider, result.url, result.canonical_source_url,
                                             result.title, result.snippet, result.cached_full_text])
                            if source not in downloads_by_source:
                                candidate = replace(result)
                                fetch_selected_full_text(candidate, record_config)
                                downloads_by_source[source] = candidate
                            candidate = replace(downloads_by_source[source], entity_key=result.entity_key,
                                                tier=result.tier, query=result.query, search_id=result.search_id,
                                                provider_rank=result.provider_rank)
                            downloads[result_key(result)] = result_row(candidate)
                    entries[key] = {
                        "profile": profile, "target": name, "entity_key": entity_key, "tier": tier,
                        "stages": list(stages), **captured,
                        "provider_parameters": _safe(config["expansion"].get("sources", {})),
                        "search_reused_from": entry_key("legacy", entity_key, tier, stages) if shared else None,
                        "candidates": [result_row(result) for result in results], "downloads": downloads,
                    }
    payload = {
        "schema_version": 1, "dataset_version": version, "mode": mode,
        "targets": targets, "inventory": inventory, "entries": entries, "initial_facts": initial,
        "preparation": {
            **(provenance or {}),
            "llm_requests": 0, "input_tokens": 0, "output_tokens": 0,
            "elapsed_seconds": time.monotonic() - started,
            "provider_searches": sum(len(entry["searches"]) for entry in entries.values()
                                     if not entry["search_reused_from"]),
            "provider_search_errors": sum(row["status"] == state.FAILED for entry in entries.values()
                                          if not entry["search_reused_from"]
                                          for row in entry["searches"]),
            "adaptive_full_text_outcomes": len(downloads_by_source),
        },
        "retrieval_config_sha256": retrieval_fingerprint(configs, mode),
    }
    # Reserve the name so concurrent preparations cannot replace one another.
    with path.open("x", encoding="utf-8"):
        pass
    atomic_write_json(path, {**payload, "dataset_sha256": digest(payload)})
    return payload


def retrieval_fingerprint(configs, mode):
    profiles = POLICIES if mode == "full-pipeline" else ("legacy",)
    return digest(_safe({
        profile: {name: configs[profile]["expansion"].get(name)
                  for name in ("search", "retrieval", "sources", "relevance")}
        for profile in profiles
    }))


def load_dataset(path):
    archive = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(archive, dict):
        raise ValueError(f"frozen dataset must be a JSON object: {path}")
    fingerprint = archive.pop("dataset_sha256", None)
    if archive.get("schema_version") != 1 or digest(archive) != fingerprint:
        raise ValueError(f"invalid frozen dataset version or SHA-256: {path}")
    archive["dataset_sha256"] = fingerprint
    for key, entry in archive["entries"].items():
        expected = entry_key(entry["profile"], entry["entity_key"], entry["tier"], entry["stages"])
        if expected != key:
            raise ValueError(f"invalid frozen retrieval key: {key}")
        for row in [*entry["provider_results"], *entry["candidates"], *entry["downloads"].values()]:
            restore_result(row)
    data = initial_bytes(archive)
    if data is not None:
        validate_initial_parquet(data)
    return archive


def validate_initial_parquet(data):
    names = set(ParquetFile(pa.BufferReader(data)).schema_arrow.names)
    required = {"fact_id", "doc_id", "subject_id", "subject_name", "subject_type",
                "object_id", "object_name", "object_type", "relation", "evidence"}
    if missing := required - names:
        raise ValueError(f"initial facts Parquet is missing columns: {', '.join(sorted(missing))}")


def initial_bytes(archive):
    initial = archive["initial_facts"]
    if initial is None:
        return None
    data = base64.b64decode(initial["parquet_base64"], validate=True)
    if hashlib.sha256(data).hexdigest() != initial["sha256"]:
        raise ValueError("invalid frozen initial graph SHA-256")
    return data


class Replay:
    """Only the archive can supply search results or downloaded bodies."""

    def __init__(self, archive, profile, *, tokenizer=None, event=None):
        self.archive = archive
        self.profile = profile
        self.tokenizer = tokenizer
        self.event = event or (lambda *args, **kwargs: None)

    def entry(self, entity_key, tier, stages):
        key = entry_key(self.profile, entity_key, tier, stages)
        try:
            return self.archive["entries"][key]
        except KeyError as error:
            raise ValueError(f"missing frozen retrieval for {self.profile} {entity_key} "
                             f"at tier {tier}, stages {list(stages)}") from error

    def load(self, entity_key, tier, *, stages):
        candidates = [restore_result(row) for row in self.entry(entity_key, tier, stages)["candidates"]]
        for result in candidates:
            self.event(entity_key, tier, result, "considered")
        return candidates

    def fetch(self, entity_key, tier, stages, candidate):
        downloads = self.entry(entity_key, tier, stages)["downloads"]
        try:
            fetched = restore_result(downloads[result_key(candidate)])
        except KeyError as error:
            raise ValueError(f"missing frozen full-text outcome for {candidate.url}") from error
        for name in RESULT_FIELDS:
            setattr(candidate, name, getattr(fetched, name))


def targets_for(config, ontology):
    with closing(state.connect(":memory:")) as db:
        seed_frontier(db, config, ontology)
        return [tuple(row) for row in db.execute(
            "SELECT entity_name,entity_key,tier FROM frontier ORDER BY rowid"
        )]
