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
from .config import seed_entities, storage_dir, storage_path
from .documents import normalize_url
from .kg.facts import FACT_COLUMNS
from .kg.support import aggregate_edge_support, contextualize_facts, load_document_origins
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


def _optional_bool(value: Any) -> Optional[bool]:
    """Keep absent legacy validation flags unknown, including pandas nulls."""
    value = json_safe(value)
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


def _any_flag(rows: list[dict[str, Any]], field: str) -> Optional[bool]:
    flags = [_optional_bool(row.get(field)) for row in rows]
    if True in flags:
        return True
    return False if False in flags else None


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
    if "doc_id" in frame:
        frame = frame[~frame.doc_id.fillna("").astype(str).str.startswith("verify:")]
    return frame


def _load_documents(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    frame = _read_parquet(storage_path(config, "normalized_documents"))
    if frame.empty or "doc_id" not in frame.columns:
        return {}
    docs: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        doc_id = _as_str(row.get("doc_id"))
        if doc_id:
            metric_path = storage_dir(config, "extractions") / "metrics" / f"{doc_id}.json"
            if metric_path.exists():
                try:
                    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                    latest = metrics.get("extraction_metrics") or metrics.get("last_attempt") or {}
                    row["extraction_metrics"] = {
                        key: latest.get(key) for key in (
                            "extraction_chunks_available", "extraction_chunks",
                            "extraction_chunks_omitted", "extraction_chunk_indices",
                            "extraction_coverage", "extraction_chunk_selection",
                        ) if key in latest
                    }
                    row["extraction_metrics"].update({
                        key: metrics.get(key) for key in ("total_elapsed_seconds", "total_llm_calls")
                        if key in metrics
                    })
                except (OSError, ValueError, AttributeError) as error:
                    logger.warning("could not read extraction metrics for %s: %s", doc_id, error)
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
    """entity_key -> {tier, entity_name, discovered_from} from state.db when present."""
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
            "SELECT entity_key, entity_name, tier, discovered_from FROM frontier"
        ).fetchall()
    except sqlite3.Error as error:
        logger.warning("could not read frontier table: %s", error)
        return {}
    finally:
        connection.close()

    tiers: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["entity_key"])
        tiers[key] = {
            "tier": int(row["tier"]),
            "entity_name": str(row["entity_name"]),
            "discovered_from": (
                str(row["discovered_from"]) if row["discovered_from"] is not None else None
            ),
        }
    return tiers


def _load_document_states(config: dict[str, Any]) -> Optional[dict[str, int]]:
    """Count completed documents, including those that yielded zero accepted facts."""
    path = storage_path(config, "state_db")
    if not path.exists():
        return None
    connection = None
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        rows = connection.execute(
            "SELECT status, COUNT(*) FROM documents GROUP BY status"
        ).fetchall()
        return {str(status): int(count) for status, count in rows}
    except sqlite3.Error as error:
        logger.warning("could not read document states for export: %s", error)
        return None
    finally:
        if connection is not None:
            connection.close()


def _load_run_metrics(config: dict[str, Any]) -> Optional[dict[str, Any]]:
    path = storage_path(config, "state_db").parent / "run_metrics.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError) as error:
        logger.warning("could not read run metrics: %s", error)
        return None


TIERABLE_TYPES = frozenset({"Company", "Mine"})


def _supply_parents(edges: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Index eligible supply edges once; use legacy edges only if flags are absent."""
    supply_edges = [edge for edge in edges if edge.get("relation") == "suppliesTo"]
    flags = [
        _optional_bool(edge.get("expansion_relevant"))
        if _optional_bool(edge.get("expansion_relevant")) is not None
        else _optional_bool((edge.get("quality") or {}).get("expansion_relevant"))
        for edge in supply_edges
    ]
    has_flags = any(flag is not None for flag in flags)
    parents: dict[str, set[str]] = {}
    for edge, relevant in zip(supply_edges, flags):
        quality = edge.get("quality") or {}
        # Current bundles retain unknown legacy validation metadata for audit, but
        # unknown quote grounding cannot establish a defensible supply-chain tier.
        if "evidence_quote_grounded" in quality and _optional_bool(
            quality.get("evidence_quote_grounded")
        ) is not True:
            continue
        if has_flags and relevant is not True:
            continue
        if _optional_bool(quality.get("frontier_eligible")) is False:
            continue
        if quality.get("contested") is True or quality.get("status") == "contested":
            continue
        parents.setdefault(edge["source"], set()).add(edge["target"])
    return parents


def _recompute_shortest_tiers(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    seed_ids: set[str],
    max_tier: Optional[int] = None,
) -> None:
    """Assign discovery_tier as min suppliesTo distance from any seed (BFS).

    Prefer expansion-relevant suppliesTo edges. When no edge carries an explicit
    expansion_relevant flag (legacy exports), all suppliesTo edges are used.
    Seeds stay at 0. Contextual types keep discovery_tier=null.
    """
    from collections import defaultdict, deque

    by_id = {node["id"]: node for node in nodes}
    upstream: dict[str, set[str]] = defaultdict(set)
    for source, targets in _supply_parents(edges).items():
        if by_id.get(source, {}).get("type") not in TIERABLE_TYPES:
            continue
        for target in targets:
            if by_id.get(target, {}).get("type") in TIERABLE_TYPES:
                upstream[target].add(source)

    tiers: dict[str, int] = {}
    parents: dict[str, set[str]] = defaultdict(set)
    queue: deque[str] = deque()
    for seed_id in sorted(seed_ids):
        if seed_id in by_id:
            tiers[seed_id] = 0
            queue.append(seed_id)

    while queue:
        customer = queue.popleft()
        customer_tier = tiers[customer]
        next_tier = customer_tier + 1
        if max_tier is not None and next_tier > int(max_tier):
            continue
        for supplier in upstream.get(customer, ()):
            if supplier not in by_id:
                continue
            if supplier not in tiers or next_tier < tiers[supplier]:
                tiers[supplier] = next_tier
                parents[supplier] = {customer}
                queue.append(supplier)
            elif next_tier == tiers[supplier]:
                parents[supplier].add(customer)

    for node in nodes:
        if node.get("type") not in TIERABLE_TYPES:
            node["discovery_tier"] = None
            node["tier_parent_id"] = None
            node["tier_parent_ids"] = []
            node["is_supply_chain_node"] = bool(node.get("is_seed"))
            continue
        node_id = node["id"]
        if node_id in seed_ids:
            node["discovery_tier"] = 0
            node["is_seed"] = True
            node["is_supply_chain_node"] = True
            node["tier_parent_ids"] = []
            node["tier_parent_id"] = None
            node["discovered_from"] = None
            continue
        if node_id in tiers:
            node["discovery_tier"] = tiers[node_id]
            parent_list = sorted(parents.get(node_id) or [])
            node["tier_parent_ids"] = parent_list
            node["tier_parent_id"] = parent_list[0] if parent_list else None
            if parent_list:
                node["discovered_from"] = parent_list[0]
            node["is_supply_chain_node"] = True
        else:
            node["discovery_tier"] = None
            node["tier_parent_id"] = None
            node["tier_parent_ids"] = []
            node["discovered_from"] = None
            node["is_supply_chain_node"] = False
            node["is_seed"] = False


def _apply_tier_semantics(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    max_tier: Optional[int] = None,
    expandable_types: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Enforce supply-chain tier invariants; demote invalid tiers to context nodes."""
    by_id = {node["id"]: node for node in nodes}
    supplies_parents = _supply_parents(edges)
    expandable_types = {"Company"} if expandable_types is None else expandable_types

    checked = 0
    violations = 0
    # Validate shallow parents before their descendants, independent of node ID order.
    for node in sorted(nodes, key=lambda n: (
        n.get("discovery_tier") is None, n.get("discovery_tier") or 0, n["id"]
    )):
        ntype = node.get("type")
        # Contextual types never carry supply-chain discovery tiers.
        if ntype not in TIERABLE_TYPES:
            if node.get("discovery_tier") is not None and not node.get("is_seed"):
                node["discovery_tier"] = None
                node["tier_parent_id"] = None
                node["tier_parent_ids"] = []
                node["discovered_from"] = None
            node["is_expandable"] = False
            node["is_supply_chain_node"] = bool(node.get("is_seed"))
            continue

        tier = node.get("discovery_tier")
        node["is_expandable"] = (
            ntype in expandable_types and tier is not None
            and (max_tier is None or int(tier) < max_tier)
        )
        if tier is None:
            node["is_supply_chain_node"] = bool(node.get("is_seed"))
            continue

        checked += 1
        if int(tier) == 0 and node.get("is_seed"):
            node["is_supply_chain_node"] = True
            node.setdefault("tier_parent_ids", [])
            continue

        parents = supplies_parents.get(node["id"], set())
        valid_parents = []
        for parent_id in parents:
            parent = by_id.get(parent_id)
            if parent is None:
                continue
            parent_tier = parent.get("discovery_tier")
            if parent_tier is not None and int(parent_tier) == int(tier) - 1:
                valid_parents.append(parent_id)
        if not valid_parents or (max_tier is not None and int(tier) > max_tier):
            violations += 1
            logger.warning(
                "tier invariant violation: %s tier=%s has no expansion-relevant "
                "suppliesTo parent at tier %s",
                node["id"], tier, int(tier) - 1,
            )
            node["discovery_tier"] = None
            node["tier_parent_id"] = None
            node["tier_parent_ids"] = []
            node["discovered_from"] = None
            node["is_supply_chain_node"] = False
            node["is_expandable"] = False
        else:
            node["tier_parent_ids"] = sorted(set(valid_parents))
            node["tier_parent_id"] = node["tier_parent_ids"][0]
            node["is_supply_chain_node"] = True

    return {"tier_nodes_checked": checked, "tier_invariant_violations": violations}


# --- bundle construction -----------------------------------------------------------


def _build_nodes_and_edges(
    facts: pd.DataFrame,
    resolution_aliases: dict[str, set[str]],
    frontier: dict[str, dict[str, Any]],
    seed_names: list[str],
    documents: dict[str, dict[str, Any]],
    ontology: dict[str, Any],
    edge_support: Optional[pd.DataFrame] = None,
    max_tier: Optional[int] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, Any]]:
    entity_types = set(ontology.get("entity_types") or [])
    relation_types = set(ontology.get("relation_types") or [])

    # Seed matching uses canonical company ids for is_seed.
    from .names import canonical_id, normalization_key

    seed_ids: set[str] = set()
    support_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    if edge_support is not None and not edge_support.empty:
        for row in edge_support.to_dict(orient="records"):
            support_by_key[
                (_as_str(row.get("subject_id")), _as_str(row.get("relation")), _as_str(row.get("object_id")))
            ] = row
    fresh_support = aggregate_edge_support(facts, config={})
    fresh_by_key = {
        (row["subject_id"], row["relation"], row["object_id"]): row
        for row in fresh_support.to_dict(orient="records")
    }

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

    # Resolve seed aliases against the already persisted canonical entities. An alias
    # is used only when unambiguous; do not invent a second isolated CATL node when
    # Prompt 2 stored its longer canonical company name.
    alias_ids: dict[str, set[str]] = {}
    for entity_id, meta in node_meta.items():
        if meta["type"] == "Company":
            for alias in meta["aliases"]:
                alias_ids.setdefault(normalization_key(alias), set()).add(entity_id)

    # Always emit configured seed companies, even with degree zero / no facts yet.
    for seed_name in seed_names:
        seed_id = canonical_id("Company", seed_name)
        candidates = alias_ids.get(normalization_key(seed_name), set())
        if seed_id not in node_meta and len(candidates) == 1:
            seed_id = next(iter(candidates))
        seed_ids.add(seed_id)
        meta = node_meta.setdefault(
            seed_id,
            {
                "id": seed_id,
                "type": "Company",
                "name": seed_name,
                "aliases": {seed_name},
                "source_doc_ids": set(),
                "in_degree": 0,
                "out_degree": 0,
            },
        )
        meta["aliases"].add(seed_name)

    nodes: list[dict[str, Any]] = []
    for entity_id, meta in sorted(node_meta.items(), key=lambda item: item[0]):
        frontier_row = frontier.get(entity_id)
        discovery_tier: Optional[int] = None
        discovered_from = None
        tier_parent_ids: list[str] = []
        if frontier_row is not None and meta["type"] in TIERABLE_TYPES:
            discovery_tier = int(frontier_row["tier"])
            discovered_from = frontier_row.get("discovered_from")
            if discovered_from:
                tier_parent_ids = [discovered_from]
        elif entity_id in seed_ids:
            discovery_tier = 0
        # Non-tierable types keep discovery_tier null even if somehow frontier-linked.
        if meta["type"] not in TIERABLE_TYPES:
            discovery_tier = None
            discovered_from = None
            tier_parent_ids = []
        is_seed = entity_id in seed_ids
        aliases = _unique_sorted(list(meta["aliases"]))
        source_doc_ids = sorted(meta["source_doc_ids"])
        nodes.append(
            {
                "id": entity_id,
                "type": meta["type"],
                "name": meta["name"],
                "aliases": aliases,
                "discovery_tier": discovery_tier,
                "tier_parent_id": tier_parent_ids[0] if tier_parent_ids else None,
                "tier_parent_ids": tier_parent_ids,
                "discovered_from": discovered_from,
                "is_seed": bool(is_seed),
                "is_expandable": meta["type"] == "Company",
                "is_supply_chain_node": bool(
                    is_seed or (discovery_tier is not None and meta["type"] in TIERABLE_TYPES)
                ),
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
            doc_meta = documents.get(doc_id, {})
            url = _as_str(row.get("source_url")) or _as_str(doc_meta.get("source_url"))
            if fact:
                fact_ids.append(fact)
                all_fact_ids.append(fact)
            if doc_id:
                source_doc_ids.append(doc_id)
            if url:
                source_urls.append(url)
            provider = (
                _as_str(row.get("provider"))
                or _as_str(doc_meta.get("provider"))
                or None
            )
            evidence_items.append(
                {
                    "fact_id": fact or None,
                    "doc_id": doc_id or None,
                    "chunk_index": json_safe(row.get("chunk_index")),
                    "quote": _as_str(row.get("evidence")) or None,
                    "source_url": url or None,
                    "retrieval_url": (
                        _as_str(row.get("retrieval_url"))
                        or _as_str(doc_meta.get("retrieval_url"))
                        or None
                    ),
                    "canonical_source_url": (
                        _as_str(row.get("canonical_source_url"))
                        or _as_str(doc_meta.get("canonical_source_url"))
                        or None
                    ),
                    "source_domain": (
                        _as_str(row.get("source_domain"))
                        or _as_str(doc_meta.get("source_domain"))
                        or None
                    ),
                    "source_class": (
                        _as_str(row.get("source_class"))
                        or _as_str(doc_meta.get("source_class"))
                        or None
                    ),
                    "source_score": json_safe(
                        row.get("source_effective_score")
                        if not _is_missing(row.get("source_effective_score"))
                        else doc_meta.get("source_effective_score")
                    ),
                    "source_known": json_safe(
                        row.get("source_known")
                        if not _is_missing(row.get("source_known"))
                        else doc_meta.get("source_known")
                    ),
                    "source_group_id": (
                        _as_str(row.get("source_group_id"))
                        or _as_str(doc_meta.get("source_group_id"))
                        or None
                    ),
                    "document_title": _as_str(doc_meta.get("title")) or None,
                    "content_source": (
                        _as_str(row.get("content_source"))
                        or _as_str(doc_meta.get("content_source"))
                        or None
                    ),
                    "provider": provider,
                    "model": _as_str(row.get("model")) or None,
                    "extracted_at": _as_str(row.get("extracted_at")) or None,
                    "validation_status": _as_str(row.get("validation_status")) or None,
                    "ontology_valid": _optional_bool(row.get("ontology_valid")),
                    "evidence_validated": _optional_bool(row.get("evidence_validated")),
                    "evidence_quote_grounded": _optional_bool(row.get("evidence_quote_grounded")),
                    "direction_explicit": _optional_bool(row.get("direction_explicit")),
                    "origin_entity": _as_str(row.get("origin_entity")) or None,
                    "expansion_origin_ids": json_safe(row.get("expansion_origin_ids")) or [],
                    "expansion_relevant": _optional_bool(row.get("expansion_relevant")),
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
        support = support_by_key.get((source_id, relation, target_id), {})
        fresh = fresh_by_key.get((source_id, relation, target_id), {})
        quality_status = (
            _as_str(fresh.get("quality_status"))
            or _as_str(support.get("verification_status"))
            or "accepted"
        )
        # Legacy multi-source labels → simplified statuses.
        if quality_status in ("candidate", "supported"):
            quality_status = "accepted"
        elif quality_status in ("strong_candidate", "verified"):
            quality_status = "accepted_high_reputation"
        contested = support.get("quality_status") == "contested" or _optional_bool(
            support.get("has_supporting_conflict")
        ) is True
        if contested:
            quality_status = "contested"
        # Facts are the source of truth; a cached edge-support table can lag behind
        # after an interrupted run. Missing flags must never turn into True via NaN.
        support_count = len(unique_docs)
        expansion_relevant = _any_flag(rows, "expansion_relevant")
        ontology_valid = _any_flag(rows, "ontology_valid")
        evidence_validated = _any_flag(rows, "evidence_validated")
        evidence_quote_grounded = _any_flag(rows, "evidence_quote_grounded")
        direction_explicit = _any_flag(rows, "direction_explicit")
        frontier_eligible: Optional[bool] = None
        if any(value is not None for value in (
            ontology_valid, evidence_validated, direction_explicit, expansion_relevant
        )):
            frontier_eligible = bool(fresh.get("frontier_eligible", False))
        if contested:
            frontier_eligible = False
        scores_by_doc: dict[str, float] = {}
        for item in evidence_items:
            score = item.get("source_score")
            if item.get("doc_id") and isinstance(score, (int, float)) and math.isfinite(score):
                scores_by_doc[item["doc_id"]] = max(scores_by_doc.get(item["doc_id"], score), score)
        scores = list(scores_by_doc.values())
        edge = {
            "id": semantic_edge_id(source_id, relation, target_id),
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "support_count": support_count,
            # Deprecated: independent-source counting removed; always null in new exports.
            "independent_support_count": None,
            "evidence_count": len(evidence_items),
            "fact_ids": sorted(set(fact_ids)),
            "source_doc_ids": unique_docs,
            "source_urls": _unique_sorted(source_urls),
            "evidence": evidence_items,
            "expansion_relevant": expansion_relevant,
            "quality": {
                "status": quality_status,
                # Compatibility for older frontend readers; not computed from source counts.
                "verification_status": "not_required",
                "max_source_score": max(scores) if scores else None,
                "average_source_score": sum(scores) / len(scores) if scores else None,
                "full_text_support_count": len({
                    item["doc_id"] for item in evidence_items
                    if item.get("content_source") == "full_text" and item.get("doc_id")
                }),
                "ontology_valid": ontology_valid,
                "evidence_validated": evidence_validated,
                "evidence_quote_grounded": evidence_quote_grounded,
                "direction_explicit": direction_explicit,
                "frontier_eligible": frontier_eligible,
                "expansion_relevant": expansion_relevant,
                "frontier_reason": json_safe(fresh.get("frontier_reason")),
                "frontier_rejection_reason": json_safe(
                    "contested" if contested else fresh.get("frontier_rejection_reason")
                ),
                "contested": contested,
            },
        }
        edges.append(edge)

    _recompute_shortest_tiers(nodes, edges, seed_ids, max_tier=max_tier)
    tier_stats = _apply_tier_semantics(
        nodes, edges, max_tier=max_tier,
        expandable_types=set(ontology.get("expandable_types", ["Company"])),
    )
    node_by_id = {node["id"]: node for node in nodes}
    for edge in edges:
        parent = node_by_id[edge["target"]]
        supplier = node_by_id[edge["source"]]
        parent_tier = parent.get("discovery_tier")
        supplier_tier = supplier.get("discovery_tier")
        reason = edge["quality"].get("frontier_reason") or {}
        reason["parent_tier_valid"] = (
            parent.get("type") in TIERABLE_TYPES and parent_tier is not None
            and (max_tier is None or parent_tier < max_tier)
        )
        reason["parent_tier_check"] = "exported_shortest_supply_path"
        reason["parent_tier"] = parent_tier
        # Hard structural invariant: supplier_tier must equal customer_tier + 1
        # for an edge to remain expansion-/frontier-relevant.
        structural_ok = (
            supplier_tier is not None
            and parent_tier is not None
            and int(supplier_tier) == int(parent_tier) + 1
        )
        reason["supplier_tier"] = supplier_tier
        reason["structural_upstream"] = structural_ok
        # Only demote known-true eligibility; preserve legacy unknown (None) flags.
        if (
            edge["relation"] == "suppliesTo"
            and supplier_tier is not None
            and parent_tier is not None
            and not structural_ok
        ):
            if edge["quality"].get("frontier_eligible") is True:
                edge["quality"]["frontier_eligible"] = False
                if edge["quality"].get("frontier_rejection_reason") in (None, "", "null"):
                    edge["quality"]["frontier_rejection_reason"] = "tier_direction_invariant"
            if edge["quality"].get("expansion_relevant") is True:
                edge["quality"]["expansion_relevant"] = False
            if edge.get("expansion_relevant") is True:
                edge["expansion_relevant"] = False
        edge["quality"]["frontier_reason"] = reason
        edge["is_supply_chain_edge"] = bool(
            edge["relation"] == "suppliesTo"
            and edge["quality"].get("frontier_eligible") is True
            and structural_ok
            and edge["target"] in supplier.get("tier_parent_ids", [])
        )
    return nodes, edges, all_fact_ids, tier_stats


def _enrich_evidence_providers(
    edges: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
    search_by_url: dict[str, dict[str, Any]],
) -> None:
    """Fill missing provider from document / search metadata; never overwrite known values."""
    for edge in edges:
        for item in edge["evidence"]:
            if item.get("provider"):
                continue
            doc = documents.get(item.get("doc_id") or "", {})
            provider = _as_str(doc.get("provider")) or None
            url = (
                item.get("source_url")
                or item.get("retrieval_url")
                or _as_str(doc.get("source_url"))
                or _as_str(doc.get("retrieval_url"))
            )
            if not provider and url:
                hit = search_by_url.get(normalize_url(url) or "")
                if hit:
                    provider = _as_str(hit.get("provider")) or None
                    if not item.get("source_url"):
                        item["source_url"] = url
                    if not item.get("document_title"):
                        item["document_title"] = _as_str(hit.get("title")) or None
            if provider is None and item.get("content_source") == "file":
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
        retrieval_url = _as_str(row.get("retrieval_url")) or None
        provider = _as_str(row.get("provider")) or None
        published_at = None
        lookup_url = source_url or retrieval_url
        if lookup_url:
            hit = search_by_url.get(normalize_url(lookup_url) or "")
            if hit:
                if not provider:
                    provider = _as_str(hit.get("provider")) or None
                published_at = _as_str(hit.get("published")) or None
        if provider is None and _as_str(row.get("content_source")) == "file":
            provider = "file"
        exported.append(
            {
                "doc_id": doc_id,
                "title": _as_str(row.get("title")) or None,
                "source_url": source_url,
                "retrieval_url": retrieval_url,
                "canonical_source_url": _as_str(row.get("canonical_source_url")) or None,
                "source_file": source_file,
                "content_source": _as_str(row.get("content_source")) or None,
                "char_count": json_safe(row.get("char_count")),
                "content_hash": _as_str(row.get("content_hash")) or None,
                "ingested_at": _as_str(row.get("ingested_at")) or None,
                "provider": provider,
                "published_at": published_at,
                "source_domain": _as_str(row.get("source_domain")) or None,
                "source_class": _as_str(row.get("source_class")) or None,
                "source_base_score": json_safe(row.get("source_base_score")),
                "source_effective_score": json_safe(row.get("source_effective_score")),
                "source_known": json_safe(row.get("source_known")),
                "publisher_name": _as_str(row.get("publisher_name")) or None,
                "publisher_domain": _as_str(row.get("publisher_domain")) or None,
                "extraction_metrics": json_safe(row.get("extraction_metrics")),
                "origin_entity": _as_str(row.get("origin_entity")) or None,
                "query": _as_str(row.get("query")) or None,
                "tier": json_safe(row.get("tier")),
            }
        )
    return exported


def _statistics(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    fact_count: int,
    *,
    rejected_count: int = 0,
    config: Optional[dict[str, Any]] = None,
    tier_stats: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    entity_types: dict[str, int] = {}
    for node in nodes:
        entity_types[node["type"]] = entity_types.get(node["type"], 0) + 1

    relation_types: dict[str, int] = {}
    evidence_items = 0
    providers: dict[str, int] = {}
    status_counts = {
        "accepted": 0, "accepted_high_reputation": 0, "contested": 0,
    }
    source_classes: dict[str, int] = {}
    evidence_full = evidence_snip = 0
    known_evidence = unknown_evidence = 0
    expansion_relevant_edges = context_only_edges = 0
    score_buckets = {"0": 0, "1-49": 0, "50-84": 0, "85-100": 0}
    for edge in edges:
        relation_types[edge["relation"]] = relation_types.get(edge["relation"], 0) + 1
        evidence_items += edge["evidence_count"]
        if edge.get("expansion_relevant") or (edge.get("quality") or {}).get("expansion_relevant"):
            expansion_relevant_edges += 1
        else:
            context_only_edges += 1
        quality = edge.get("quality") or {}
        status = quality.get("status") or "accepted"
        if status in status_counts:
            status_counts[status] += 1
        for item in edge["evidence"]:
            provider = item.get("provider") or "unknown"
            providers[provider] = providers.get(provider, 0) + 1
            cls = item.get("source_class") or "unknown"
            source_classes[cls] = source_classes.get(cls, 0) + 1
            cs = item.get("content_source")
            if cs == "full_text":
                evidence_full += 1
            elif cs == "title_snippet":
                evidence_snip += 1
            # Evidence-item known/unknown based on source_known when present, else score.
            known_flag = item.get("source_known")
            if known_flag is True:
                known_evidence += 1
            elif known_flag is False:
                unknown_evidence += 1
            elif item.get("source_score") is None:
                unknown_evidence += 1
            else:
                known_evidence += 1
            try:
                if item.get("source_score") is not None:
                    value = float(item["source_score"])
                    if value <= 0:
                        score_buckets["0"] += 1
                    elif value < 50:
                        score_buckets["1-49"] += 1
                    elif value < 85:
                        score_buckets["50-84"] += 1
                    else:
                        score_buckets["85-100"] += 1
            except (TypeError, ValueError):
                pass

    discovery_tiers: dict[str, int] = {}
    configured_depth = int(((config or {}).get("expansion") or {}).get("max_tier", 5))
    observed_depth = max((int(node["discovery_tier"]) for node in nodes
                          if node.get("discovery_tier") is not None), default=0)
    tier_counts = {i: 0 for i in range(max(5, configured_depth, observed_depth) + 1)}
    tiered_supply = context_nodes = 0
    for node in nodes:
        tier = node.get("discovery_tier")
        if node.get("is_supply_chain_node") and tier is not None:
            tiered_supply += 1
            key = str(tier)
            discovery_tiers[key] = discovery_tiers.get(key, 0) + 1
            ti = int(tier)
            if ti in tier_counts:
                tier_counts[ti] += 1
        else:
            context_nodes += 1

    known_docs = sum(1 for d in documents if d.get("source_known") is True)
    unknown_docs = sum(
        1 for d in documents
        if d.get("source_known") is False or d.get("source_known") is None
    )
    doc_full = sum(1 for d in documents if d.get("content_source") == "full_text")
    doc_snip = sum(1 for d in documents if d.get("content_source") == "title_snippet")
    doc_wrapper = sum(1 for d in documents if d.get("source_class") == "retrieval_wrapper")
    docs_by_provider: dict[str, int] = {}
    docs_by_origin: dict[str, int] = {}
    for doc in documents:
        provider = doc.get("provider") or "unknown"
        docs_by_provider[provider] = docs_by_provider.get(provider, 0) + 1
        origin = doc.get("origin_entity") or "unknown"
        docs_by_origin[str(origin)] = docs_by_origin.get(str(origin), 0) + 1

    tier_stats = tier_stats or {}
    return {
        "facts": fact_count,
        "accepted_facts": fact_count,
        "rejected_facts": rejected_count,
        "nodes": len(nodes),
        "edges": len(edges),
        "documents": len(documents),
        # Filled from checkpoint completion states by build_graph_bundle.
        "documents_processed": None,
        "evidence_items": evidence_items,
        "entity_types": dict(sorted(entity_types.items())),
        "relation_types": dict(sorted(relation_types.items())),
        "discovery_tiers": dict(sorted(discovery_tiers.items(), key=lambda kv: int(kv[0]))),
        "tier_0_nodes": tier_counts[0],
        "tier_1_nodes": tier_counts[1],
        "tier_2_nodes": tier_counts[2],
        "tier_3_nodes": tier_counts[3],
        "tier_4_nodes": tier_counts[4],
        "tier_5_nodes": tier_counts[5],
        **{f"tier_{tier}_nodes": count for tier, count in tier_counts.items() if tier > 5},
        "seed_count": tier_counts[0],
        "seeds_by_group": (config or {}).get("_seeds_by_group") or {},
        "tiered_supply_chain_nodes": tiered_supply,
        "context_nodes": context_nodes,
        "expansion_relevant_edges": expansion_relevant_edges,
        "context_only_edges": context_only_edges,
        "tier_nodes_checked": tier_stats.get("tier_nodes_checked", 0),
        "tier_invariant_violations": tier_stats.get("tier_invariant_violations", 0),
        "source_providers": dict(sorted(providers.items())),
        "documents_by_provider": dict(sorted(docs_by_provider.items())),
        "documents_by_origin_entity": dict(sorted(docs_by_origin.items())),
        "documents_by_content_source": {
            "full_text": doc_full,
            "title_snippet": doc_snip,
        },
        "quality_status": status_counts,
        "document_content": {
            "full_text": doc_full,
            "title_snippet": doc_snip,
        },
        "source_classes": dict(sorted(source_classes.items())),
        "source_score_ranges": score_buckets,
        "known_source_documents": known_docs,
        "unknown_source_documents": unknown_docs,
        "known_source_evidence_items": known_evidence,
        "unknown_source_evidence_items": unknown_evidence,
        "wrapper_documents": doc_wrapper,
        "full_text_documents": doc_full,
        "title_snippet_documents": doc_snip,
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
    from .run_config import effective_export_config

    config, run_snapshot = effective_export_config(config)
    ontology = load_ontology(config)
    facts = _load_facts(config)
    facts = contextualize_facts(facts, load_document_origins(config))
    documents = _load_documents(config)
    resolution_aliases = _load_resolution_aliases(config)
    frontier = _load_frontier_tiers(config)
    search_by_url = _load_search_by_url(config)
    seeds = seed_entities(config) if config.get("seeds") else []
    from .config import seed_groups
    from .kg.support import load_edge_support

    edge_support = load_edge_support(config)
    max_tier = (config.get("expansion") or {}).get("max_tier")

    nodes, edges, fact_ids, tier_stats = _build_nodes_and_edges(
        facts, resolution_aliases, frontier, seeds, documents, ontology,
        edge_support=edge_support,
        max_tier=int(max_tier) if max_tier is not None else None,
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
    rejected_path = storage_path(config, "rejected_facts")
    rejected_count = len(pd.read_parquet(rejected_path)) if rejected_path.exists() else 0
    groups = seed_groups(config) if config.get("seeds") else {}
    stats = _statistics(
        nodes, edges, document_entries,
        len(facts) if not facts.empty else 0,
        rejected_count=rejected_count,
        config={**config, "_seeds_by_group": groups},
        tier_stats=tier_stats,
    )
    document_states = _load_document_states(config)
    stats["documents_available"] = len(documents)
    stats["documents_with_facts"] = len(referenced_docs)
    stats["document_states"] = document_states
    stats["documents_processed"] = (
        sum(document_states.get(status, 0) for status in ("RESOLVED", "COMMITTED"))
        if document_states is not None else None
    )
    stats["documents_committed_to_neo4j"] = (
        document_states.get("COMMITTED", 0) if document_states is not None else None
    )
    all_documents = _build_documents(set(documents), documents, search_by_url)
    selected_by_provider: dict[str, int] = {}
    full_by_provider: dict[str, int] = {}
    for doc in all_documents:
        provider = doc.get("provider") or "unknown"
        selected_by_provider[provider] = selected_by_provider.get(provider, 0) + 1
        if doc.get("content_source") == "full_text":
            full_by_provider[provider] = full_by_provider.get(provider, 0) + 1
    accepted_by_provider: dict[str, set[str]] = {}
    for edge in edges:
        if edge["quality"].get("frontier_eligible") is not True:
            continue
        for item in edge["evidence"]:
            if all(item.get(flag) is True for flag in (
                "ontology_valid", "evidence_validated", "direction_explicit", "expansion_relevant"
            )) and item.get("evidence_quote_grounded") is not False:
                accepted_by_provider.setdefault(item.get("provider") or "unknown", set()).add(edge["id"])
    stats["selected_documents_by_provider"] = dict(sorted(selected_by_provider.items()))
    stats["selected_full_text_documents_by_provider"] = {
        key: full_by_provider.get(key, 0) for key in sorted(selected_by_provider)
    }
    stats["accepted_expansion_edges_by_provider"] = {
        key: len(value) for key, value in sorted(accepted_by_provider.items())
    }
    stats["supply_chain_core_edges"] = sum(edge["is_supply_chain_edge"] for edge in edges)
    stats["provider_statistics_scope"] = {
        "documents_by_provider": "documents referenced by accepted facts (legacy field)",
        "selected_documents_by_provider": "all normalized selected documents, including zero-fact documents",
        "source_providers": "accepted evidence items (legacy field; quotes may repeat documents)",
        "accepted_expansion_edges_by_provider": "unique eligible semantic edges witnessed by each provider; providers can overlap",
    }
    run_metrics = _load_run_metrics(config)
    stats["run_metrics"] = json_safe(run_metrics)
    completion = None
    if isinstance(pipeline_summary, dict) and pipeline_summary.get("completion"):
        completion = json_safe(pipeline_summary["completion"])
    elif isinstance(run_metrics, dict) and run_metrics.get("completion"):
        completion = json_safe(run_metrics["completion"])
    else:
        # Derive from frontier checkpoint when summary is absent.
        try:
            from . import state as state_mod
            from .config import storage_path as _storage_path
            db = _storage_path(config, "state_db")
            if db.exists():
                conn = state_mod.connect(db)
                try:
                    counts = state_mod.frontier_counts(conn)
                    tiers = state_mod.frontier_tier_counts(conn)
                    pending = int(state_mod.pending_frontier_count(conn, int(max_tier or 5)))
                    completion = {
                        "complete": pending == 0 and int(counts.get(state_mod.PROCESSING, 0) or 0) == 0,
                        "stop_reason": (run_metrics or {}).get("stop_reason"),
                        "pending_frontier": pending,
                        "processing_frontier": int(counts.get(state_mod.PROCESSING, 0) or 0),
                        "done_frontier": int(counts.get(state_mod.DONE, 0) or 0),
                        "failed_frontier": int(counts.get(state_mod.FAILED, 0) or 0),
                        "highest_reached_tier": max(tiers) if tiers else 0,
                        "highest_discovery_tier": max(tiers) if tiers else 0,
                    }
                finally:
                    conn.close()
        except Exception:  # noqa: BLE001
            completion = None

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "generator": {"name": "SChain", "version": __version__},
            "mode": mode,
            "model": model,
            "graph_fingerprint": fingerprint,
            "seeds": list(seeds),
            "seed_groups": seed_groups(config) if config.get("seeds") else {},
            "seed_count": len(seeds),
            "max_tier": max_tier,
            "run_config": json_safe(run_snapshot),
            "run_config_provenance": (
                "expansion_snapshot_legacy_artifacts_unverified"
                if run_snapshot and run_snapshot.get("legacy_artifacts_unverified")
                else "expansion_snapshot" if run_snapshot else "legacy_export_config_no_snapshot"
            ),
            "pipeline_summary": json_safe(pipeline_summary or {}),
            "completion": completion,
            "experiment_complete": bool((completion or {}).get("complete")) if completion else None,
            "quality_schema_version": "2.0",
        },
        "ontology": {
            "entity_types": list(ontology.get("entity_types") or []),
            "relation_types": list(ontology.get("relation_types") or []),
            "tier_advancing_relations": list(ontology.get("tier_advancing_relations") or []),
            "expandable_types": list(ontology.get("expandable_types") or []),
            "patterns": [list(pattern) for pattern in (ontology.get("patterns") or [])],
        },
        "statistics": stats,
        "completion": completion,
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
