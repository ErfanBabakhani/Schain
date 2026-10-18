"""Offline viewer validation and interaction regressions; no browser or network."""

from __future__ import annotations

import json
import shutil
import subprocess
from importlib.resources import files

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
    ("source", []), ("quality", []), ("quality", "checked"),
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
    html = output.read_text()
    assert attack not in html
    payload = html.split('<script type="application/json" id="graphData">', 1)[1].split('</script>', 1)[0]
    assert json.loads(payload)["nodes"][0]["name"] == attack


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed (optional viewer interaction test)")
def test_inspection_keeps_zoom_and_distinguishes_failed_grounding():
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.attrs={};this.textContent='';this.value='';this.checked=false;this.hidden=false;this.style={};this.listeners={};}
 append(...children){this.children.push(...children);if(this.tag==='select'&&!this.value&&children[0])this.value=children[0].value;}
 replaceChildren(...children){this.children=[...children];}
 setAttribute(k,v){this.attrs[k]=v;}
 addEventListener(k,v){this.listeners[k]=v;}
 getBoundingClientRect(){return {width:1000,height:600};}
}
const elements=new Map();
const get=id=>{if(!elements.has(id))elements.set(id,new Element(['seed','relation','type','tier','quality','support'].includes(id)?'select':id));return elements.get(id)};
get('graphData').textContent=JSON.stringify({nodes:[{id:'a',name:'A',type:'Company',is_seed:true,discovery_tier:0},{id:'b',name:'B',type:'Company',discovery_tier:1}],edges:[{id:'e',source:'b',target:'a',relation:'suppliesTo',source_doc_ids:['d'],evidence:[{doc_id:'d',quote:'q',evidence_quote_grounded:false}],quality:{evidence_validated:true}}]});
const context=vm.createContext({document:{getElementById:get,createElement:t=>new Element(t),createElementNS:(_,t)=>new Element(t)},URL,Blob,setTimeout,clearTimeout,console});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
vm.runInContext('zoom(0.5)',context);
const before=vm.runInContext('JSON.stringify(view)',context);
vm.runInContext('inspectEdge(bundle.edges[0])',context);
assert.equal(vm.runInContext('JSON.stringify(view)',context),before);
const gather=e=>[e.textContent,...e.children.flatMap(gather)].filter(Boolean);
assert(gather(get('details')).includes('Quote did not match source'));
vm.runInContext('inspectNode(bundle.nodes[0])',context);
assert.equal(vm.runInContext('JSON.stringify(view)',context),before);
const original=vm.runInContext('JSON.stringify(bundle)',context);
context.bad={nodes:[{id:'x',name:7,is_seed:true}],edges:[]};
assert.throws(()=>vm.runInContext('load(bad,"bad")',context),/must be a string/);
assert.equal(vm.runInContext('JSON.stringify(bundle)',context),original);
context.minimal={nodes:[{id:'x',is_seed:true},{id:'y',is_seed:true}],edges:[]};
vm.runInContext('load(minimal,"minimal")',context);
assert.equal(vm.runInContext('bundle.nodes[0].name',context),'x');
assert.equal(vm.runInContext('bundle.nodes[0].type',context),'Unknown');
console.log('Viewer interaction checks passed');
'''
    app = files("schain").joinpath("viewer_assets/app.js")
    result = subprocess.run([shutil.which("node"), "-e", script, str(app)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
