"""Build a self-contained, offline graph explorer from a SChain JSON export.

This module uses only the Python standard library; no model or Neo4j is needed.
"""
from __future__ import annotations

import argparse
import json
import math
from importlib.resources import files
from pathlib import Path
from typing import Any


def validate_graph(data: Any) -> dict:
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list) or not isinstance(data.get("edges"), list):
        raise ValueError("Expected a SChain graph object with nodes and edges arrays")
    ids = [n.get("id") for n in data["nodes"] if isinstance(n, dict)]
    if len(ids) != len(data["nodes"]) or any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Node IDs must be unique, non-empty strings")
    known = set(ids)
    nodes = []
    for n in data["nodes"]:
        for key in ("name", "type"):
            if n.get(key) is not None and not isinstance(n[key], str):
                raise ValueError(f"Node {key} must be a string when present")
        if n.get("aliases") is not None and (
            not isinstance(n["aliases"], list) or any(not isinstance(a, str) for a in n["aliases"])
        ):
            raise ValueError("Node aliases must be an array of strings")
        tier = n.get("discovery_tier")
        if tier is not None and (
            isinstance(tier, bool) or not isinstance(tier, (int, float))
            or (isinstance(tier, float) and not math.isfinite(tier))
            or tier < 0 or int(tier) != tier
        ):
            raise ValueError("Node discovery_tier must be a non-negative integer or null")
        nodes.append({**n, "name": n.get("name") or n["id"], "type": n.get("type") or "Unknown"})
    for key in ("metadata", "statistics"):
        if data.get(key) is not None and not isinstance(data[key], dict):
            raise ValueError(f"Graph {key} must be an object")
    generated = (data.get("metadata") or {}).get("generated_at")
    if generated is not None and not isinstance(generated, str):
        raise ValueError("metadata.generated_at must be a string")
    if data.get("documents") is not None and (
        not isinstance(data["documents"], list)
        or any(not isinstance(d, dict) for d in data["documents"])
    ):
        raise ValueError("Graph documents must be an array of objects")
    edge_ids = set()
    for e in data["edges"]:
        if not isinstance(e, dict) or any(
            not isinstance(e.get(key), str) or e[key] not in known for key in ("source", "target")
        ):
            raise ValueError("Each edge must reference two existing node IDs")
        if not isinstance(e.get("id"), str) or not e["id"] or e["id"] in edge_ids:
            raise ValueError("Edge IDs must be unique, non-empty strings")
        edge_ids.add(e["id"])
        if e.get("quality") is not None and not isinstance(e["quality"], dict):
            raise ValueError("Edge quality must be an object")
        if e.get("source_doc_ids") is not None and (
            not isinstance(e["source_doc_ids"], list)
            or any(not isinstance(d, str) for d in e["source_doc_ids"])
        ):
            raise ValueError("Edge source_doc_ids must be an array of strings")
        if e.get("evidence") is not None and (
            not isinstance(e["evidence"], list)
            or any(not isinstance(item, dict) for item in e["evidence"])
        ):
            raise ValueError("Edge evidence must be an array of objects")
    return {**data, "nodes": nodes}


def write_viewer(graph: dict, output: Path) -> Path:
    """Embed graph data safely: source text cannot close the JSON script element."""
    graph = validate_graph(graph)
    assets = files("schain").joinpath("viewer_assets")
    payload = json.dumps(graph, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = assets.joinpath("index.html").read_text(encoding="utf-8")
    # Insert data last so strings in uploaded documents cannot become template code.
    html = html.replace("/*__STYLE__*/", assets.joinpath("style.css").read_text(encoding="utf-8"))
    html = html.replace("/*__SCRIPT__*/", assets.joinpath("app.js").read_text(encoding="utf-8"))
    html = html.replace("__GRAPH_DATA__", payload)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(html, encoding="utf-8")
    temp.replace(output)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", default=Path("outputs/graph-explorer.html"), type=Path)
    args = parser.parse_args(argv)
    try:
        graph = json.loads(args.input.read_text(encoding="utf-8"))
        print(write_viewer(graph, args.output))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Cannot build graph explorer: {error}\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
