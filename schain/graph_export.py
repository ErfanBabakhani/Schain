"""Export a frontend-ready semantic knowledge-graph JSON bundle.

Reads only durable Parquet / SQLite artifacts. Does not call an LLM, open Neo4j, or
mutate extraction results. The bundle carries graph semantics and provenance only —
no layout, colours, or rendering hints.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from . import __version__
from .config import seed_entities, storage_path
from .documents import normalize_url
from .kg.facts import FACT_COLUMNS
from .ontology import load_ontology

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
DEFAULT_EXPORT_RELATIVE = "outputs/supply_chain_graph.v1.json"


# --- JSON-safe helpers -------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def json_safe(value: Any) -> Any:
    """Convert pandas / numpy values into plain JSON-serializable Python objects."""
    if _is_missing(value):
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    # numpy scalars expose item()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except (ValueError, TypeError):
            pass
    text = str(value)
    return text if text else None


def _as_str(value: Any, default: str = "") -> str:
    cleaned = json_safe(value)
    if cleaned is None:
        return default
    return str(cleaned)


def _unique_sorted(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _as_str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return sorted(result, key=str.casefold)


def semantic_edge_id(source_id: str, relation: str, target_id: str) -> str:
    """Deterministic id for one semantic edge; independent of supporting documents."""
    payload = "|".join([source_id, relation, target_id])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"edge:{digest}"


def graph_fingerprint(node_ids: list[str], edge_ids: list[str], fact_ids: list[str]) -> str:
    """Stable hash of semantic graph content (excludes generated_at)."""
    payload = "\n".join(
        [
            "nodes:" + ",".join(sorted(node_ids)),
            "edges:" + ",".join(sorted(edge_ids)),
            "facts:" + ",".join(sorted(fact_ids)),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- artifact loading --------------------------------------------------------------


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_facts(config: dict[str, Any]) -> pd.DataFrame:
    path = storage_path(config, "extracted_facts")
    if not path.exists():
        logger.warning("facts.parquet not found at %s; exporting an empty graph", path)
        return pd.DataFrame(columns=FACT_COLUMNS)
    frame = pd.read_parquet(path)
    if frame.empty:
        return pd.DataFrame(columns=FACT_COLUMNS)
    return frame


def _load_documents(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    frame = _read_parquet(storage_path(config, "normalized_documents"))
    if frame.empty or "doc_id" not in frame.columns:
        return {}
    docs: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        doc_id = _as_str(row.get("doc_id"))
        if doc_id:
            docs[doc_id] = row
    return docs


def _load_resolution_aliases(config: dict[str, Any]) -> dict[str, set[str]]:
    """canonical_id -> set of alias / mention strings from entity_resolution.parquet."""
    frame = _read_parquet(storage_path(config, "entity_resolution"))
    aliases: dict[str, set[str]] = {}
    if frame.empty:
        return aliases
    for row in frame.to_dict(orient="records"):
        canonical_id = _as_str(row.get("canonical_id"))
        if not canonical_id:
            continue
        bucket = aliases.setdefault(canonical_id, set())
        mention = _as_str(row.get("mention")).strip()
        if mention:
            bucket.add(mention)
        canonical_name = _as_str(row.get("canonical_name")).strip()
        if canonical_name:
            bucket.add(canonical_name)
        raw_aliases = row.get("aliases")
        if _is_missing(raw_aliases):
            continue
        if isinstance(raw_aliases, str):
            try:
                raw_aliases = json.loads(raw_aliases)
            except json.JSONDecodeError:
                raw_aliases = [raw_aliases]
        elif hasattr(raw_aliases, "tolist"):
            raw_aliases = raw_aliases.tolist()
        if isinstance(raw_aliases, (list, tuple, set)):
            for alias in raw_aliases:
                text = _as_str(alias).strip()
                if text:
                    bucket.add(text)
    return aliases


def _load_search_by_url(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """normalized URL -> first matching search-result row (provider / published)."""
    frame = _read_parquet(storage_path(config, "search_results"))
    by_url: dict[str, dict[str, Any]] = {}
    if frame.empty or "url" not in frame.columns:
        return by_url
    for row in frame.to_dict(orient="records"):
        url = normalize_url(_as_str(row.get("url")) or None)
        if not url or url in by_url:
            continue
        by_url[url] = row
    return by_url


def _load_frontier_tiers(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """entity_key -> {tier, entity_name} from state.db when present."""
    path = storage_path(config, "state_db")
    if not path.exists():
        return {}
    try:
        connection = sqlite3.connect(str(path))
        connection.row_factory = sqlite3.Row
    except sqlite3.Error as error:
        logger.warning("could not open state.db for frontier tiers: %s", error)
        return {}
    try:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='frontier'"
        ).fetchone()
        if not exists:
            return {}
        rows = connection.execute(
            "SELECT entity_key, entity_name, tier FROM frontier"
        ).fetchall()
    except sqlite3.Error as error:
        logger.warning("could not read frontier table: %s", error)
        return {}
    finally:
        connection.close()

    tiers: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["entity_key"])
        tiers[key] = {"tier": int(row["tier"]), "entity_name": str(row["entity_name"])}
    return tiers


# --- bundle construction -----------------------------------------------------------


def _build_nodes_and_edges(
    facts: pd.DataFrame,
    resolution_aliases: dict[str, set[str]],
    frontier: dict[str, dict[str, Any]],
    seed_names: list[str],
    documents: dict[str, dict[str, Any]],
    ontology: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    entity_types = set(ontology.get("entity_types") or [])
    relation_types = set(ontology.get("relation_types") or [])

    # Seed matching uses canonical company ids for is_seed.
    from .names import canonical_id

    seed_ids = {canonical_id("Company", name) for name in seed_names}

    node_meta: dict[str, dict[str, Any]] = {}
    edge_buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for row in facts.to_dict(orient="records"):
        subject_id = _as_str(row.get("subject_id"))
        object_id = _as_str(row.get("object_id"))
        relation = _as_str(row.get("relation"))
        subject_type = _as_str(row.get("subject_type"))
        object_type = _as_str(row.get("object_type"))
        if not subject_id or not object_id or not relation:
            continue
        if entity_types and subject_type not in entity_types:
            raise ValueError(f"unknown subject entity type {subject_type!r} on {subject_id}")
        if entity_types and object_type not in entity_types:
            raise ValueError(f"unknown object entity type {object_type!r} on {object_id}")
        if relation_types and relation not in relation_types:
            raise ValueError(f"unknown relation type {relation!r}")

        for entity_id, entity_type, entity_name, mention in (
            (subject_id, subject_type, row.get("subject_name"), row.get("subject_mention")),
            (object_id, object_type, row.get("object_name"), row.get("object_mention")),
        ):
            meta = node_meta.setdefault(
                entity_id,
                {
                    "id": entity_id,
                    "type": entity_type,
                    "name": _as_str(entity_name, entity_id),
                    "aliases": set(),
                    "source_doc_ids": set(),
                    "in_degree": 0,
                    "out_degree": 0,
                },
            )
            name = _as_str(entity_name)
            if name and len(name) > len(meta["name"]):
                meta["name"] = name
            if _as_str(mention).strip():
                meta["aliases"].add(_as_str(mention).strip())
            if name:
                meta["aliases"].add(name)
            doc_id = _as_str(row.get("doc_id"))
            if doc_id:
                meta["source_doc_ids"].add(doc_id)

        key = (subject_id, relation, object_id)
        if key not in edge_buckets:
            edge_buckets[key] = []
            node_meta[subject_id]["out_degree"] += 1
            node_meta[object_id]["in_degree"] += 1
        edge_buckets[key].append(row)

    for entity_id, extra in resolution_aliases.items():
        if entity_id in node_meta:
            node_meta[entity_id]["aliases"].update(extra)

    nodes: list[dict[str, Any]] = []
    for entity_id, meta in sorted(node_meta.items(), key=lambda item: item[0]):
        frontier_row = frontier.get(entity_id)
        discovery_tier: Optional[int] = None
        if frontier_row is not None:
            discovery_tier = int(frontier_row["tier"])
        is_seed = entity_id in seed_ids or discovery_tier == 0
        aliases = _unique_sorted(list(meta["aliases"]))
        source_doc_ids = sorted(meta["source_doc_ids"])
        nodes.append(
            {
                "id": entity_id,
                "type": meta["type"],
                "name": meta["name"],
                "aliases": aliases,
                "discovery_tier": discovery_tier,
                "is_seed": bool(is_seed),
                "source_doc_ids": source_doc_ids,
                "stats": {
                    "in_degree": int(meta["in_degree"]),
                    "out_degree": int(meta["out_degree"]),
                    "edge_count": int(meta["in_degree"] + meta["out_degree"]),
                    "support_documents": len(source_doc_ids),
                },
            }
        )

    edges: list[dict[str, Any]] = []
    all_fact_ids: list[str] = []
    for (source_id, relation, target_id), rows in sorted(
        edge_buckets.items(), key=lambda item: item[0]
    ):
        evidence_items: list[dict[str, Any]] = []
        fact_ids: list[str] = []
        source_doc_ids: list[str] = []
        source_urls: list[str] = []
        for row in rows:
            fact = _as_str(row.get("fact_id"))
            doc_id = _as_str(row.get("doc_id"))
            url = _as_str(row.get("source_url"))
            if fact:
                fact_ids.append(fact)
                all_fact_ids.append(fact)
            if doc_id:
                source_doc_ids.append(doc_id)
            if url:
                source_urls.append(url)
            doc_meta = documents.get(doc_id, {})
            evidence_items.append(
                {
                    "fact_id": fact or None,
                    "doc_id": doc_id or None,
                    "chunk_index": json_safe(row.get("chunk_index")),
                    "quote": _as_str(row.get("evidence")) or None,
                    "source_url": url or None,
                    "document_title": _as_str(doc_meta.get("title")) or None,
                    "content_source": _as_str(doc_meta.get("content_source")) or None,
                    "provider": None,  # filled below when search metadata exists
                    "model": _as_str(row.get("model")) or None,
                    "extracted_at": _as_str(row.get("extracted_at")) or None,
                }
            )

        # Deterministic evidence order: doc_id, chunk_index, fact_id.
        evidence_items.sort(
            key=lambda item: (
                item.get("doc_id") or "",
                -1 if item.get("chunk_index") is None else item["chunk_index"],
                item.get("fact_id") or "",
            )
        )
        unique_docs = _unique_sorted(source_doc_ids)
        edges.append(
            {
                "id": semantic_edge_id(source_id, relation, target_id),
                "source": source_id,
                "target": target_id,
                "relation": relation,
                "support_count": len(unique_docs),
                "evidence_count": len(evidence_items),
                "fact_ids": sorted(set(fact_ids)),
                "source_doc_ids": unique_docs,
                "source_urls": _unique_sorted(source_urls),
                "evidence": evidence_items,
            }
        )

    return nodes, edges, all_fact_ids


def _enrich_evidence_providers(
    edges: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
    search_by_url: dict[str, dict[str, Any]],
) -> None:
    for edge in edges:
        for item in edge["evidence"]:
            provider = None
            url = item.get("source_url") or _as_str(
                documents.get(item.get("doc_id") or "", {}).get("source_url")
            )
            if url:
                hit = search_by_url.get(normalize_url(url) or "")
                if hit:
                    provider = _as_str(hit.get("provider")) or None
                    if not item.get("source_url"):
                        item["source_url"] = url
                    if not item.get("document_title"):
                        item["document_title"] = _as_str(hit.get("title")) or None
            if provider is None:
                content_source = item.get("content_source")
                if content_source == "file":
                    provider = "file"
            item["provider"] = provider


def _build_documents(
    doc_ids: set[str],
    documents: dict[str, dict[str, Any]],
    search_by_url: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for doc_id in sorted(doc_ids):
        row = documents.get(doc_id, {})
        source_path = _as_str(row.get("source_path")) or None
        source_file = Path(source_path).name if source_path else None
        source_url = _as_str(row.get("source_url")) or None
        provider = None
        published_at = None
        if source_url:
            hit = search_by_url.get(normalize_url(source_url) or "")
            if hit:
                provider = _as_str(hit.get("provider")) or None
                published_at = _as_str(hit.get("published")) or None
        if provider is None and _as_str(row.get("content_source")) == "file":
            provider = "file"
        exported.append(
            {
                "doc_id": doc_id,
                "title": _as_str(row.get("title")) or None,
                "source_url": source_url,
                "source_file": source_file,
                "content_source": _as_str(row.get("content_source")) or None,
                "char_count": json_safe(row.get("char_count")),
                "content_hash": _as_str(row.get("content_hash")) or None,
                "ingested_at": _as_str(row.get("ingested_at")) or None,
                "provider": provider,
                "published_at": published_at,
            }
        )
    return exported


def _statistics(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    fact_count: int,
) -> dict[str, Any]:
    entity_types: dict[str, int] = {}
    for node in nodes:
        entity_types[node["type"]] = entity_types.get(node["type"], 0) + 1

    relation_types: dict[str, int] = {}
    evidence_items = 0
    providers: dict[str, int] = {}
    for edge in edges:
        relation_types[edge["relation"]] = relation_types.get(edge["relation"], 0) + 1
        evidence_items += edge["evidence_count"]
        for item in edge["evidence"]:
            provider = item.get("provider") or "unknown"
            providers[provider] = providers.get(provider, 0) + 1

    discovery_tiers: dict[str, int] = {}
    for node in nodes:
        tier = node.get("discovery_tier")
        if tier is None:
            continue
        key = str(tier)
        discovery_tiers[key] = discovery_tiers.get(key, 0) + 1

    return {
        "facts": fact_count,
        "nodes": len(nodes),
        "edges": len(edges),
        "documents": len(documents),
        "evidence_items": evidence_items,
        "entity_types": dict(sorted(entity_types.items())),
        "relation_types": dict(sorted(relation_types.items())),
        "discovery_tiers": dict(sorted(discovery_tiers.items(), key=lambda kv: int(kv[0]))),
        "source_providers": dict(sorted(providers.items())),
    }


def _validate_bundle(bundle: dict[str, Any], ontology: dict[str, Any]) -> None:
    nodes = bundle["nodes"]
    edges = bundle["edges"]
    node_ids = [node["id"] for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("exported nodes contain duplicate ids")
    edge_ids = [edge["id"] for edge in edges]
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("exported edges contain duplicate ids")

    allowed_entities = set(ontology.get("entity_types") or [])
    allowed_relations = set(ontology.get("relation_types") or [])
    id_set = set(node_ids)
    for node in nodes:
        if allowed_entities and node["type"] not in allowed_entities:
            raise ValueError(f"node {node['id']} has illegal type {node['type']!r}")
    for edge in edges:
        if allowed_relations and edge["relation"] not in allowed_relations:
            raise ValueError(f"edge {edge['id']} has illegal relation {edge['relation']!r}")
        if edge["source"] not in id_set or edge["target"] not in id_set:
            raise ValueError(
                f"edge {edge['id']} references missing endpoint "
                f"{edge['source']!r} -> {edge['target']!r}"
            )


def build_graph_bundle(
    config: dict[str, Any],
    *,
    mode: str = "export",
    pipeline_summary: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the versioned semantic graph dict from persisted artifacts."""
    ontology = load_ontology(config)
    facts = _load_facts(config)
    documents = _load_documents(config)
    resolution_aliases = _load_resolution_aliases(config)
    frontier = _load_frontier_tiers(config)
    search_by_url = _load_search_by_url(config)
    seeds = seed_entities(config) if config.get("seeds") else []

    nodes, edges, fact_ids = _build_nodes_and_edges(
        facts, resolution_aliases, frontier, seeds, documents, ontology
    )
    _enrich_evidence_providers(edges, documents, search_by_url)

    referenced_docs: set[str] = set()
    for edge in edges:
        referenced_docs.update(edge["source_doc_ids"])
    document_entries = _build_documents(referenced_docs, documents, search_by_url)

    models = _unique_sorted(
        [_as_str(row.get("model")) for row in facts.to_dict(orient="records")]
        if not facts.empty
        else [_as_str(config.get("llm", {}).get("model"))]
    )
    model = models[0] if models else _as_str(config.get("llm", {}).get("model")) or None

    fingerprint = graph_fingerprint(
        [node["id"] for node in nodes],
        [edge["id"] for edge in edges],
        fact_ids,
    )
    stats = _statistics(nodes, edges, document_entries, len(facts) if not facts.empty else 0)

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "generator": {"name": "SChain", "version": __version__},
            "mode": mode,
            "model": model,
            "graph_fingerprint": fingerprint,
            "seeds": list(seeds),
            "pipeline_summary": json_safe(pipeline_summary or {}),
        },
        "ontology": {
            "entity_types": list(ontology.get("entity_types") or []),
            "relation_types": list(ontology.get("relation_types") or []),
            "tier_advancing_relations": list(ontology.get("tier_advancing_relations") or []),
            "patterns": [list(pattern) for pattern in (ontology.get("patterns") or [])],
        },
        "statistics": stats,
        "nodes": nodes,
        "edges": edges,
        "documents": document_entries,
    }
    _validate_bundle(bundle, ontology)
    return bundle


def export_path(config: dict[str, Any], override: Optional[Path] = None) -> Path:
    """Resolve the graph-export destination path."""
    from .config import resolve_path

    if override is not None:
        path = Path(override)
        return path if path.is_absolute() else resolve_path(path)
    if "graph_export" in config.get("storage", {}):
        return storage_path(config, "graph_export")
    return resolve_path(DEFAULT_EXPORT_RELATIVE)


def write_graph_bundle(bundle: dict[str, Any], path: Path) -> Path:
    """Atomically write the JSON bundle (temp file + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    logger.info(
        "wrote graph export %s (%d nodes, %d edges)",
        path, bundle["statistics"]["nodes"], bundle["statistics"]["edges"],
    )
    return path


def export_graph(
    config: dict[str, Any],
    *,
    output: Optional[Path] = None,
    mode: str = "export",
    pipeline_summary: Optional[dict[str, Any]] = None,
) -> Path:
    """Build and write the frontend-ready graph JSON. Returns the output path."""
    path = export_path(config, output)
    bundle = build_graph_bundle(config, mode=mode, pipeline_summary=pipeline_summary)
    return write_graph_bundle(bundle, path)
