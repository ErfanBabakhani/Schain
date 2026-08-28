"""The V1 static ontology and Prompt 1.

Entity types, relation types and the allowed (source, relation, target) patterns are
declared in ``config/ontology.yaml`` so they are stated once, in one readable place.
They are static for V1: nothing here grows the ontology at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from neo4j_graphrag.components.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
from neo4j_graphrag.generation.prompts import ERExtractionTemplate

from .config import REPO_ROOT, _read_yaml, resolve_path

DEFAULT_ONTOLOGY_PATH = REPO_ROOT / "config" / "ontology.yaml"


def load_ontology(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Load the ontology block, from the path named in ``config.ontology_file``."""
    if config and config.get("ontology"):
        ontology = config["ontology"]
    else:
        path = resolve_path((config or {}).get("ontology_file", DEFAULT_ONTOLOGY_PATH))
        ontology = _read_yaml(Path(path)).get("ontology", {})

    missing = [key for key in ("entity_types", "relation_types") if not ontology.get(key)]
    if missing:
        raise ValueError(f"ontology is missing {', '.join(missing)}")
    return ontology


def build_graph_schema(ontology: dict[str, Any]) -> GraphSchema:
    """Build the schema handed to the extractor.

    Every entity carries ``name``; every relation carries ``evidence``, the verbatim
    sentence supporting it, which is what makes an extracted fact auditable later.
    """
    node_types = [
        NodeType(label=label, properties=[PropertyType(name="name", type="STRING", required=True)])
        for label in ontology["entity_types"]
    ]
    relationship_types = [
        RelationshipType(label=label, properties=[PropertyType(name="evidence", type="STRING")])
        for label in ontology["relation_types"]
    ]
    patterns = tuple(tuple(pattern) for pattern in ontology.get("patterns", []) or [])
    return GraphSchema(
        node_types=node_types,
        relationship_types=relationship_types,
        patterns=patterns,
    )


def render_schema(schema: dict) -> str:
    """Render the schema as a short readable block instead of a raw dict dump.

    The library passes ``schema.model_dump()`` into the prompt, which is verbose and
    hard to read. Compacting it keeps the prompt reviewable by a human and shorter.
    """

    def props(entry: dict) -> str:
        names = [p["name"] for p in entry.get("properties", [])]
        return f"({', '.join(names)})" if names else ""

    lines = ["Node labels:"]
    for node in schema.get("node_types", []):
        lines.append(f"  {node['label']}{props(node)}")
    lines.append("Relationship types:")
    for rel in schema.get("relationship_types", []):
        lines.append(f"  {rel['label']}{props(rel)}")
    patterns = schema.get("patterns", ())
    if patterns:
        lines.append("Allowed patterns:")
        for pattern in patterns:
            # model_dump() turns each pattern into a dict; a plain tuple is also accepted.
            if isinstance(pattern, dict):
                source, relation, target = (
                    pattern["source"],
                    pattern["relationship"],
                    pattern["target"],
                )
            else:
                source, relation, target = pattern
            lines.append(f"  ({source})-[:{relation}]->({target})")
    return "\n".join(lines)


# Worked example from the architecture brief, shown to the model so the expected shape
# (including evidence) is unambiguous.
EXTRACTION_EXAMPLES = """\
Input text: "CATL supplies battery cells to BMW. The plant is located in Germany."
Output:
{"nodes": [
  {"id": "0", "label": "Company", "properties": {"name": "CATL"}},
  {"id": "1", "label": "Company", "properties": {"name": "BMW"}},
  {"id": "2", "label": "Location", "properties": {"name": "Germany"}}
],
"relationships": [
  {"type": "suppliesTo", "start_node_id": "0", "end_node_id": "1",
   "properties": {"evidence": "CATL supplies battery cells to BMW."}}
]}\
"""


class SupplyChainERTemplate(ERExtractionTemplate):
    """Prompt 1 - schema-constrained entity and relation extraction.

    Extends the library template with supply-chain framing, an explicit direction rule
    for ``suppliesTo`` (the most error-prone relation), and the evidence requirement.
    The ``{schema}``, ``{examples}`` and ``{text}`` placeholders are required by the
    library and must be kept.
    """

    DEFAULT_TEMPLATE = """\
You are an information extraction system building a supply-chain knowledge graph.

Extract entities and relationships from the input text below. Return JSON only:
{{"nodes": [{{"id": "0", "label": "Company", "properties": {{"name": "CATL"}}}}],
"relationships": [{{"type": "suppliesTo", "start_node_id": "0", "end_node_id": "1",
"properties": {{"evidence": "verbatim sentence from the input text"}}}}]}}

Use only the following node labels, relationship types and allowed patterns:
{schema}

Rules:
- Extract only facts stated in the input text. Never use outside knowledge, and never guess.
- Use the entity name as written in the text; do not expand or abbreviate it.
- suppliesTo points from the supplier to the customer. In "X supplies Y to Z" the
  relationship is X suppliesTo Z, and Y is the supplied product or material.
- Every relationship must include an "evidence" property containing the verbatim
  sentence from the input text that supports it.
- Assign a unique string id to each node and reuse it in the relationships.
- Skip anything that does not fit the allowed labels, types and patterns.
- Return the JSON object on its own, with no backticks and no commentary.

Examples:
{examples}

Input text:

{text}
"""

    def format(self, *args, **kwargs):  # type: ignore[override]
        """Compact the schema dict the extractor passes in before rendering."""
        schema = kwargs.get("schema")
        if isinstance(schema, dict):
            kwargs["schema"] = render_schema(schema)
        return super().format(*args, **kwargs)
