"""Render a DocumentGraph to interactive self-contained HTML."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from .graph import (
    DEFAULT_ENTITY_COLOR,
    ENTITY_COLORS,
    DocumentGraph,
    build_document_graph,
)

PathLike = Union[str, Path]


def _safe_filename(doc_id: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", doc_id.strip()) or "document"
    return cleaned[:120]


def _node_size(degree: int) -> int:
    # Mild scaling: base 18, +3 per incident edge, capped.
    return min(36, 18 + max(degree, 1) * 3)


def _graph_payload(graph: DocumentGraph) -> dict[str, Any]:
    nodes = []
    for node in graph.nodes:
        color = ENTITY_COLORS.get(node.entity_type, DEFAULT_ENTITY_COLOR)
        nodes.append(
            {
                "id": node.id,
                "label": node.name,
                "title": f"{node.name} ({node.entity_type})",
                "group": node.entity_type,
                "entityType": node.entity_type,
                "degree": node.degree,
                "size": _node_size(node.degree),
                "color": {
                    "background": color,
                    "border": color,
                    "highlight": {"background": "#F4D03F", "border": "#7D6608"},
                    "hover": {"background": color, "border": "#F4D03F"},
                },
                "font": {"color": "#F8F9F9", "size": 14, "face": "IBM Plex Sans, Segoe UI, sans-serif"},
                "shape": "dot",
                "borderWidth": 2,
            }
        )

    edges = []
    for index, edge in enumerate(graph.edges):
        supports = [
            {
                "fact_id": support.fact_id,
                "evidence": support.evidence,
                "source_url": support.source_url,
                "doc_id": support.doc_id,
                "chunk_index": support.chunk_index,
                "model": support.model,
                "extracted_at": support.extracted_at,
                "subject_mention": support.subject_mention,
                "object_mention": support.object_mention,
            }
            for support in edge.supports
        ]
        edges.append(
            {
                "id": f"e{index}",
                "from": edge.source_id,
                "to": edge.target_id,
                "label": edge.relation,
                "relation": edge.relation,
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.85}},
                "color": {"color": "#85929E", "highlight": "#D4AC0D", "hover": "#F5B041"},
                "font": {
                    "align": "middle",
                    "size": 12,
                    "color": "#2C3E50",
                    "strokeWidth": 3,
                    "strokeColor": "#FBFCFC",
                    "face": "IBM Plex Sans, Segoe UI, sans-serif",
                },
                "width": min(5, 1.5 + 0.4 * edge.support_count),
                "smooth": {"type": "cubicBezier", "forceDirection": "horizontal", "roundness": 0.35},
                "sourceName": edge.source_name,
                "targetName": edge.target_name,
                "sourceType": edge.source_type,
                "targetType": edge.target_type,
                "supportCount": edge.support_count,
                "supports": supports,
            }
        )

    legend = [
        {"type": entity_type, "color": color}
        for entity_type, color in ENTITY_COLORS.items()
        if any(node.entity_type == entity_type for node in graph.nodes)
    ]
    extras = sorted(
        {
            node.entity_type
            for node in graph.nodes
            if node.entity_type not in ENTITY_COLORS
        }
    )
    for entity_type in extras:
        legend.append({"type": entity_type, "color": DEFAULT_ENTITY_COLOR})

    return {
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "entities": graph.entity_count,
            "facts": graph.fact_count,
            "edges": graph.edge_count,
            "relations": graph.relation_types,
            "doc_ids": graph.doc_ids,
            "models": graph.models,
        },
        "legend": legend,
        "relationTypes": graph.relation_types,
    }


def _html_document(title: str, payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    safe_title = html.escape(title)
    # vis-network from CDN (works in Colab and browsers with network access).
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{safe_title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&display=swap" rel="stylesheet"/>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
:root {{
  --bg: #0f1419;
  --panel: #18212b;
  --panel-2: #1f2a36;
  --ink: #e8eef4;
  --muted: #8fa3b5;
  --accent: #d4a017;
  --line: #2a3847;
  --good: #3d8b6e;
}}
* {{ box-sizing: border-box; }}
html, body {{
  margin: 0; height: 100%;
  background: radial-gradient(1200px 600px at 10% -10%, #1a2733 0%, var(--bg) 55%);
  color: var(--ink);
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
}}
.app {{
  display: grid;
  grid-template-columns: 280px 1fr 340px;
  grid-template-rows: auto 1fr;
  height: 100%;
  gap: 0;
}}
header {{
  grid-column: 1 / -1;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.9rem 1.25rem;
  border-bottom: 1px solid var(--line);
  background: rgba(15,20,25,0.92);
  backdrop-filter: blur(8px);
}}
header h1 {{
  margin: 0;
  font-family: "IBM Plex Serif", Georgia, serif;
  font-size: 1.25rem;
  font-weight: 600;
  letter-spacing: 0.01em;
}}
header .meta {{
  color: var(--muted);
  font-size: 0.85rem;
}}
.sidebar, .detail {{
  background: var(--panel);
  border-right: 1px solid var(--line);
  padding: 1rem;
  overflow: auto;
}}
.detail {{
  border-right: none;
  border-left: 1px solid var(--line);
}}
.canvas-wrap {{
  position: relative;
  min-width: 0;
  min-height: 0;
}}
#network {{
  width: 100%;
  height: 100%;
  background:
    linear-gradient(180deg, rgba(26,39,51,0.35), transparent 40%),
    repeating-linear-gradient(0deg, transparent, transparent 31px, rgba(42,56,71,0.35) 32px),
    repeating-linear-gradient(90deg, transparent, transparent 31px, rgba(42,56,71,0.35) 32px);
}}
.section-title {{
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin: 0 0 0.6rem;
}}
.stats {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  margin-bottom: 1.1rem;
}}
.stat {{
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.65rem 0.7rem;
}}
.stat .n {{
  font-size: 1.35rem;
  font-weight: 600;
  color: var(--accent);
}}
.stat .l {{
  font-size: 0.72rem;
  color: var(--muted);
}}
.legend-item, .filter-item {{
  display: flex;
  align-items: center;
  gap: 0.55rem;
  margin: 0.35rem 0;
  font-size: 0.9rem;
}}
.swatch {{
  width: 12px; height: 12px; border-radius: 50%;
  box-shadow: 0 0 0 2px rgba(255,255,255,0.08);
}}
.controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin: 0.8rem 0 1rem;
}}
button, .search input {{
  font: inherit;
}}
button {{
  background: var(--panel-2);
  color: var(--ink);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.4rem 0.7rem;
  cursor: pointer;
}}
button:hover {{ border-color: var(--accent); color: #fff; }}
button.active {{
  background: #2d2410;
  border-color: var(--accent);
  color: var(--accent);
}}
.search input {{
  width: 100%;
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 8px;
  color: var(--ink);
  padding: 0.5rem 0.65rem;
  margin-bottom: 0.8rem;
}}
.search input:focus {{ outline: 1px solid var(--accent); }}
.hint {{
  color: var(--muted);
  font-size: 0.82rem;
  line-height: 1.45;
}}
.detail h2 {{
  margin: 0 0 0.4rem;
  font-size: 1.05rem;
}}
.kv {{
  display: grid;
  grid-template-columns: 7.5rem 1fr;
  gap: 0.25rem 0.6rem;
  font-size: 0.86rem;
  margin: 0.8rem 0;
}}
.kv dt {{ color: var(--muted); }}
.kv dd {{ margin: 0; word-break: break-word; }}
.evidence-card {{
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.75rem;
  margin: 0.55rem 0;
}}
.evidence-card .badge {{
  display: inline-block;
  font-size: 0.7rem;
  color: var(--accent);
  border: 1px solid rgba(212,160,23,0.35);
  border-radius: 999px;
  padding: 0.1rem 0.45rem;
  margin-bottom: 0.45rem;
}}
.evidence-text {{
  white-space: pre-wrap;
  line-height: 1.45;
  font-size: 0.9rem;
}}
.empty {{
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  color: var(--muted); font-size: 1.05rem;
}}
@media (max-width: 1100px) {{
  .app {{ grid-template-columns: 1fr; grid-template-rows: auto auto 55vh auto; }}
  .sidebar, .detail {{ border: none; border-bottom: 1px solid var(--line); }}
}}
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>{safe_title}</h1>
    <div class="meta" id="header-meta"></div>
  </header>
  <aside class="sidebar">
    <p class="section-title">Summary</p>
    <div class="stats" id="stats"></div>
    <p class="section-title">Search</p>
    <div class="search"><input id="search" type="search" placeholder="Find entity by name…"/></div>
    <p class="section-title">Layout</p>
    <div class="controls">
      <button type="button" id="layout-hier" class="active">Hierarchical</button>
      <button type="button" id="layout-force">Force</button>
      <button type="button" id="fit">Fit</button>
      <button type="button" id="reset">Reset</button>
    </div>
    <p class="section-title">Relations</p>
    <div id="filters"></div>
    <p class="section-title" style="margin-top:1rem">Entity types</p>
    <div id="legend"></div>
    <p class="hint" style="margin-top:1.2rem">
      Click an edge to inspect evidence. Direction is exactly as stored in facts
      (supplier → customer for <code>suppliesTo</code>).
    </p>
  </aside>
  <div class="canvas-wrap">
    <div id="network"></div>
    <div class="empty" id="empty" hidden>No facts to visualize yet.</div>
  </div>
  <aside class="detail">
    <p class="section-title">Selection</p>
    <div id="detail">
      <p class="hint">Select a node or edge to inspect provenance.</p>
    </div>
  </aside>
</div>
<script>
const DATA = {data_json};

function esc(s) {{
  return String(s ?? "").replace(/[&<>"']/g, c => ({{
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  }})[c]);
}}

const stats = DATA.summary;
document.getElementById("header-meta").textContent = [
  stats.doc_ids.length ? ("docs: " + stats.doc_ids.join(", ")) : null,
  stats.models.length ? ("model: " + stats.models.join(", ")) : null,
].filter(Boolean).join(" · ");

document.getElementById("stats").innerHTML = `
  <div class="stat"><div class="n">${{stats.entities}}</div><div class="l">entities</div></div>
  <div class="stat"><div class="n">${{stats.facts}}</div><div class="l">validated facts</div></div>
  <div class="stat"><div class="n">${{stats.edges}}</div><div class="l">graph edges</div></div>
  <div class="stat"><div class="n">${{stats.relations.length}}</div><div class="l">relation types</div></div>
`;

document.getElementById("legend").innerHTML = DATA.legend.map(item => `
  <div class="legend-item"><span class="swatch" style="background:${{item.color}}"></span>${{esc(item.type)}}</div>
`).join("");

const activeRelations = new Set(DATA.relationTypes);
const filters = document.getElementById("filters");
filters.innerHTML = DATA.relationTypes.map(rel => `
  <label class="filter-item">
    <input type="checkbox" data-rel="${{esc(rel)}}" checked/>
    <span>${{esc(rel)}}</span>
  </label>
`).join("") || '<p class="hint">No relations</p>';

const nodes = new vis.DataSet(DATA.nodes);
const edges = new vis.DataSet(DATA.edges);
const container = document.getElementById("network");
const empty = document.getElementById("empty");
if (!DATA.nodes.length) empty.hidden = false;

const hierOptions = {{
  layout: {{
    hierarchical: {{
      enabled: true,
      direction: "LR",
      sortMethod: "directed",
      levelSeparation: 180,
      nodeSpacing: 120,
      treeSpacing: 160,
      blockShifting: true,
      edgeMinimization: true,
      parentCentralization: true,
    }}
  }},
  physics: {{ enabled: false }},
}};
const forceOptions = {{
  layout: {{ hierarchical: {{ enabled: false }} }},
  physics: {{
    enabled: true,
    solver: "forceAtlas2Based",
    forceAtlas2Based: {{
      gravitationalConstant: -42,
      centralGravity: 0.01,
      springLength: 140,
      springConstant: 0.06,
      avoidOverlap: 0.7,
    }},
    stabilization: {{ iterations: 120 }},
  }},
}};

const baseOptions = {{
  interaction: {{ hover: true, tooltipDelay: 120, navigationButtons: true, keyboard: true }},
  nodes: {{ shadow: {{ enabled: true, color: "rgba(0,0,0,0.35)", size: 8, x: 0, y: 2 }} }},
  edges: {{
    selectionWidth: 2,
    hoverWidth: 1.5,
  }},
}};

let currentLayout = "hier";
const network = new vis.Network(
  container,
  {{ nodes, edges }},
  Object.assign({{}}, baseOptions, hierOptions)
);

function applyRelationFilter() {{
  const visible = DATA.edges.filter(e => activeRelations.has(e.relation));
  edges.clear();
  edges.add(visible);
  network.fit({{ animation: false }});
}}

filters.addEventListener("change", (ev) => {{
  const input = ev.target;
  if (!input.matches("input[data-rel]")) return;
  const rel = input.getAttribute("data-rel");
  if (input.checked) activeRelations.add(rel); else activeRelations.delete(rel);
  applyRelationFilter();
}});

function setLayout(mode) {{
  currentLayout = mode;
  document.getElementById("layout-hier").classList.toggle("active", mode === "hier");
  document.getElementById("layout-force").classList.toggle("active", mode === "force");
  network.setOptions(Object.assign({{}}, baseOptions, mode === "hier" ? hierOptions : forceOptions));
  network.fit({{ animation: true }});
}}
document.getElementById("layout-hier").onclick = () => setLayout("hier");
document.getElementById("layout-force").onclick = () => setLayout("force");
document.getElementById("fit").onclick = () => network.fit({{ animation: true }});
document.getElementById("reset").onclick = () => {{
  DATA.relationTypes.forEach(r => activeRelations.add(r));
  filters.querySelectorAll("input[data-rel]").forEach(i => {{ i.checked = true; }});
  applyRelationFilter();
  setLayout("hier");
}};

document.getElementById("search").addEventListener("input", (ev) => {{
  const q = ev.target.value.trim().toLowerCase();
  if (!q) {{
    network.unselectAll();
    return;
  }}
  const match = DATA.nodes.find(n => n.label.toLowerCase().includes(q));
  if (!match) return;
  network.selectNodes([match.id]);
  network.focus(match.id, {{ scale: 1.25, animation: true }});
  showNode(match.id);
}});

function showNode(id) {{
  const node = DATA.nodes.find(n => n.id === id);
  if (!node) return;
  document.getElementById("detail").innerHTML = `
    <h2>${{esc(node.label)}}</h2>
    <dl class="kv">
      <dt>Type</dt><dd>${{esc(node.entityType)}}</dd>
      <dt>Canonical id</dt><dd><code>${{esc(node.id)}}</code></dd>
      <dt>Degree</dt><dd>${{node.degree}}</dd>
    </dl>
  `;
}}

function showEdge(id) {{
  const edge = DATA.edges.find(e => e.id === id);
  if (!edge) return;
  const cards = edge.supports.map((s, i) => `
    <div class="evidence-card">
      <div class="badge">support ${{i + 1}} / ${{edge.supportCount}}</div>
      <div class="evidence-text">${{esc(s.evidence) || "<em>No evidence text</em>"}}</div>
      <dl class="kv">
        <dt>Mentions</dt><dd>${{esc(s.subject_mention)}} → ${{esc(s.object_mention)}}</dd>
        <dt>doc_id</dt><dd>${{esc(s.doc_id)}}</dd>
        <dt>chunk</dt><dd>${{esc(s.chunk_index)}}</dd>
        <dt>source URL</dt><dd>${{s.source_url ? `<a href="${{esc(s.source_url)}}" target="_blank" rel="noopener">${{esc(s.source_url)}}</a>` : "—"}}</dd>
        <dt>model</dt><dd>${{esc(s.model) || "—"}}</dd>
        <dt>extracted_at</dt><dd>${{esc(s.extracted_at) || "—"}}</dd>
        <dt>fact_id</dt><dd><code>${{esc(s.fact_id)}}</code></dd>
      </dl>
    </div>
  `).join("");
  document.getElementById("detail").innerHTML = `
    <h2>${{esc(edge.sourceName)}} → ${{esc(edge.targetName)}}</h2>
    <dl class="kv">
      <dt>Relation</dt><dd><strong>${{esc(edge.relation)}}</strong></dd>
      <dt>Source</dt><dd>${{esc(edge.sourceName)}} <span class="hint">(${{esc(edge.sourceType)}})</span></dd>
      <dt>Target</dt><dd>${{esc(edge.targetName)}} <span class="hint">(${{esc(edge.targetType)}})</span></dd>
      <dt>Support</dt><dd>${{edge.supportCount}} fact(s)</dd>
    </dl>
    ${{cards}}
  `;
}}

network.on("selectNode", params => {{
  if (params.nodes.length) showNode(params.nodes[0]);
}});
network.on("selectEdge", params => {{
  if (params.edges.length) showEdge(params.edges[0]);
}});
</script>
</body>
</html>
"""


def render_html(graph: DocumentGraph, title: str = "SChain knowledge graph") -> str:
    return _html_document(title, _graph_payload(graph))


def render_document_graph(
    facts: pd.DataFrame,
    output_path: PathLike,
    *,
    title: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> Path:
    """Write one HTML graph for ``facts`` (optionally filtered to ``doc_id``)."""
    frame = facts
    if doc_id is not None:
        if "doc_id" not in facts.columns:
            raise ValueError("facts frame has no doc_id column")
        frame = facts[facts["doc_id"] == doc_id]
    # Never mutate the caller's frame: filtering already returns a view/copy; graph build copies.
    graph = build_document_graph(frame)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if title is None:
        if doc_id:
            title = f"SChain · {doc_id}"
        elif graph.doc_ids:
            title = f"SChain · {', '.join(graph.doc_ids[:3])}"
        else:
            title = "SChain knowledge graph"
    path.write_text(render_html(graph, title=title), encoding="utf-8")
    return path


def render_all_graph(
    facts: pd.DataFrame,
    output_dir: PathLike,
    *,
    also_per_document: bool = True,
) -> dict[str, Path]:
    """Write ``all.html`` and optionally one HTML file per ``doc_id``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    written["all"] = render_document_graph(
        facts, output_dir / "all.html", title="SChain · all documents"
    )
    if also_per_document and not facts.empty and "doc_id" in facts.columns:
        for doc_id in sorted(facts["doc_id"].dropna().astype(str).unique()):
            path = output_dir / f"{_safe_filename(doc_id)}.html"
            written[doc_id] = render_document_graph(
                facts, path, doc_id=doc_id, title=f"SChain · {doc_id}"
            )
    return written


def load_facts(path: PathLike) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
