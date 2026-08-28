"""The V1 static ontology, expressed as a neo4j-graphrag ``GraphSchema``.

Entity and relation types are frozen for V1 (see PROJECT_DECISIONS.md section 4).
The schema object is what constrains the extraction prompt, so this module is also
where Prompt 1 lives.
"""

from __future__ import annotations

from neo4j_graphrag.components.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
from neo4j_graphrag.generation.prompts import ERExtractionTemplate

ENTITY_TYPES = ("Company", "Location", "Material", "Person", "Product", "Mine")
RELATION_TYPES = ("suppliesTo", "locatedIn", "owns", "produces")

# Which (source, relation, target) combinations are accepted. The frozen decisions fix
# the type lists but not the combinations, so these are stated explicitly here rather
# than left implicit in the prompt. Edit this list to widen or narrow extraction.
PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("Company", "suppliesTo", "Company"),
    ("Mine", "suppliesTo", "Company"),
    ("Company", "produces", "Product"),
    ("Company", "produces", "Material"),
    ("Mine", "produces", "Material"),
    ("Company", "locatedIn", "Location"),
    ("Mine", "locatedIn", "Location"),
    ("Company", "owns", "Company"),
    ("Company", "owns", "Mine"),
    ("Person", "owns", "Company"),
)


def build_graph_schema() -> GraphSchema:
    """Build the static schema handed to the extractor.

    Every entity carries only ``name``; every relation carries ``evidence``, the verbatim
    sentence that supports it, which is what makes an extracted fact auditable later.
    """
    node_types = [
        NodeType(
            label=label,
            properties=[PropertyType(name="name", type="STRING", required=True)],
        )
        for label in ENTITY_TYPES
    ]
    relationship_types = [
        RelationshipType(
            label=label,
            properties=[PropertyType(name="evidence", type="STRING")],
        )
        for label in RELATION_TYPES
    ]
    return GraphSchema(
        node_types=node_types,
        relationship_types=relationship_types,
        patterns=PATTERNS,
    )


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


class SupplyChainERTemplate(ERExtractionTemplate):
    """Prompt 1 - schema-guided entity and relation extraction.

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
