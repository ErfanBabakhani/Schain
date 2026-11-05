"""Offline viewer validation and interaction regressions; no browser or network."""

from __future__ import annotations

import json

import pytest

from schain.viewer import validate_graph, write_viewer


def graph():
    return {"nodes": [{"id": "a"}, {"id": "b", "name": "B", "type": "Company"}],
            "edges": [{"id": "e", "source": "b", "target": "a", "relation": "suppliesTo"}]}


def test_missing_optional_labels_have_safe_fallbacks_without_mutating_input():
    data = graph()
    result = validate_graph(data)
    assert result["nodes"][0]["name"] == "a"
    assert result["nodes"][0]["type"] == "Unknown"
    assert "name" not in data["nodes"][0]


@pytest.mark.parametrize("field,value", [
    ("name", 123), ("aliases", {}), ("aliases", [1]),
    ("discovery_tier", "1"), ("discovery_tier", 0.5),
    ("discovery_tier", True), ("discovery_tier", -1),
])
def test_malformed_node_shapes_are_rejected(field, value):
    data = graph()
    data["nodes"][0][field] = value
    with pytest.raises(ValueError):
        validate_graph(data)


@pytest.mark.parametrize("field,value", [
    ("source", []), ("source", "missing"), ("target", "missing"),
    ("quality", []), ("quality", "checked"),
    ("evidence", {}), ("evidence", [None]), ("source_doc_ids", "doc"),
])
def test_malformed_evidence_shapes_are_rejected(field, value):
    data = graph()
    data["edges"][0][field] = value
    with pytest.raises(ValueError):
        validate_graph(data)


def test_source_text_cannot_escape_embedded_json_script(tmp_path):
    data = graph()
    attack = '</script><script>alert("source text")</script>/*__SCRIPT__*/'
    data["nodes"][0]["name"] = attack
    output = write_viewer(data, tmp_path / "viewer.html")
    html = output.read_text(encoding="utf-8")
    assert attack not in html
    assert "__GRAPH_DATA__" not in html
    assert 'src="https://' not in html
    payload = html.split('<script type="application/json" id="graphData">', 1)[1].split('</script>', 1)[0]
    assert json.loads(payload) == validate_graph(data)
    assert json.loads(payload)["nodes"][0]["name"] == attack
