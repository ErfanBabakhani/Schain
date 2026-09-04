"""Interactive knowledge-graph visualization from the facts Parquet layer.

Typical Colab usage after an ingest run::

    from IPython.display import IFrame, display
    from schain.visualization import render_all_graph, load_facts

    facts = load_facts("data/processed/facts.parquet")
    paths = render_all_graph(facts, "data/visualizations")
    display(IFrame(src=str(paths["all"]), width="100%", height=720))
"""

from .graph import DocumentGraph, GraphEdge, GraphNode, build_document_graph
from .render import load_facts, render_all_graph, render_document_graph, render_html

__all__ = [
    "DocumentGraph",
    "GraphEdge",
    "GraphNode",
    "build_document_graph",
    "load_facts",
    "render_all_graph",
    "render_document_graph",
    "render_html",
]
