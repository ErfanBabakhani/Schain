"""Build an aggregated directed graph from the facts Parquet layer.

Visualization only: one node per canonical entity, one edge per
(subject_id, relation, object_id) with all supporting evidence retained.
Does not call the LLM, write Neo4j, or mutate the source DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from ..kg.facts import FACT_COLUMNS

# Stable colours by entity type (never randomised between runs).
ENTITY_COLORS: dict[str, str] = {
    "Company": "#1B4F72",
    "Location": "#1E8449",
    "Material": "#B9770E",
    "Person": "#6C3483",
    "Product": "#148F77",
    "Mine": "#922B21",
}
DEFAULT_ENTITY_COLOR = "#5D6D7E"

# Known ontology relations get a preferred filter order; unknown types still work.
PREFERRED_RELATIONS = ("suppliesTo", "locatedIn", "owns", "produces")


@dataclass(frozen=True)
class GraphNode:
    id: str
    name: str
    entity_type: str
    degree: int = 0


@dataclass
class EdgeSupport:
    """One fact row that supports a semantic edge."""

    fact_id: str
    evidence: str
    source_url: Optional[str]
    doc_id: str
    chunk_index: Any
    model: Optional[str]
    extracted_at: Optional[str]
    subject_mention: str
    object_mention: str


@dataclass
class GraphEdge:
    source_id: str
    source_name: str
    source_type: str
    relation: str
    target_id: str
    target_name: str
    target_type: str
    supports: list[EdgeSupport] = field(default_factory=list)

    @property
    def support_count(self) -> int:
        return len(self.supports)

    @property
    def edge_key(self) -> tuple[str, str, str]:
        return (self.source_id, self.relation, self.target_id)


@dataclass
class DocumentGraph:
    """Directed knowledge graph derived from final facts only."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    doc_ids: list[str]
    models: list[str]
    relation_types: list[str]

    @property
    def entity_count(self) -> int:
        return len(self.nodes)

    @property
    def fact_count(self) -> int:
        return sum(edge.support_count for edge in self.edges)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


def _as_str(value: Any, default: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return str(value)


def build_document_graph(facts: pd.DataFrame) -> DocumentGraph:
    """Aggregate facts into nodes and directed edges.

    Duplicate semantic edges (same subject_id, relation, object_id) collapse to one
    visible edge; every supporting fact's evidence is kept on that edge.
    """
    if facts is None or facts.empty:
        return DocumentGraph(nodes=[], edges=[], doc_ids=[], models=[], relation_types=[])

    required = {
        "subject_id", "subject_type", "subject_name",
        "relation",
        "object_id", "object_type", "object_name",
    }
    missing = required - set(facts.columns)
    if missing:
        raise ValueError(f"facts frame is missing columns: {sorted(missing)}")

    working = facts.copy()
    node_meta: dict[str, dict[str, str]] = {}
    degree: dict[str, int] = {}
    edges: dict[tuple[str, str, str], GraphEdge] = {}

    for row in working.itertuples(index=False):
        subject_id = _as_str(getattr(row, "subject_id"))
        object_id = _as_str(getattr(row, "object_id"))
        relation = _as_str(getattr(row, "relation"))
        if not subject_id or not object_id or not relation:
            continue

        subject_name = _as_str(getattr(row, "subject_name"), subject_id)
        object_name = _as_str(getattr(row, "object_name"), object_id)
        subject_type = _as_str(getattr(row, "subject_type"), "Unknown")
        object_type = _as_str(getattr(row, "object_type"), "Unknown")

        node_meta.setdefault(subject_id, {"name": subject_name, "entity_type": subject_type})
        node_meta.setdefault(object_id, {"name": object_name, "entity_type": object_type})
        # Prefer a longer display name if a later fact provides one.
        if len(subject_name) > len(node_meta[subject_id]["name"]):
            node_meta[subject_id]["name"] = subject_name
        if len(object_name) > len(node_meta[object_id]["name"]):
            node_meta[object_id]["name"] = object_name

        key = (subject_id, relation, object_id)
        edge = edges.get(key)
        if edge is None:
            edge = GraphEdge(
                source_id=subject_id,
                source_name=subject_name,
                source_type=subject_type,
                relation=relation,
                target_id=object_id,
                target_name=object_name,
                target_type=object_type,
            )
            edges[key] = edge
            degree[subject_id] = degree.get(subject_id, 0) + 1
            degree[object_id] = degree.get(object_id, 0) + 1

        edge.supports.append(
            EdgeSupport(
                fact_id=_as_str(getattr(row, "fact_id", "")),
                evidence=_as_str(getattr(row, "evidence", "")),
                source_url=_as_str(getattr(row, "source_url", "")) or None,
                doc_id=_as_str(getattr(row, "doc_id", "")),
                chunk_index=getattr(row, "chunk_index", None),
                model=_as_str(getattr(row, "model", "")) or None,
                extracted_at=_as_str(getattr(row, "extracted_at", "")) or None,
                subject_mention=_as_str(getattr(row, "subject_mention", subject_name)),
                object_mention=_as_str(getattr(row, "object_mention", object_name)),
            )
        )

    nodes = [
        GraphNode(
            id=node_id,
            name=meta["name"],
            entity_type=meta["entity_type"],
            degree=degree.get(node_id, 0),
        )
        for node_id, meta in sorted(node_meta.items(), key=lambda item: item[1]["name"].lower())
    ]
    edge_list = sorted(
        edges.values(),
        key=lambda edge: (edge.relation, edge.source_name.lower(), edge.target_name.lower()),
    )
    doc_ids = sorted({s.doc_id for e in edge_list for s in e.supports if s.doc_id})
    models = sorted({s.model for e in edge_list for s in e.supports if s.model})
    relation_types = sorted({e.relation for e in edge_list})
    # Prefer ontology order, then any extras alphabetically.
    ordered_relations = [r for r in PREFERRED_RELATIONS if r in relation_types]
    ordered_relations += [r for r in relation_types if r not in ordered_relations]

    return DocumentGraph(
        nodes=nodes,
        edges=edge_list,
        doc_ids=doc_ids,
        models=models,
        relation_types=ordered_relations,
    )


def empty_facts_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=FACT_COLUMNS)
