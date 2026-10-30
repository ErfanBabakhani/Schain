"""Rebuild the fictional graph and offline HTML; requires only Python's stdlib.

Run from the project root: python examples/build_demo.py
No article-derived results or LLM accuracy are represented by this fixture.
"""
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schain.viewer import write_viewer

ENTITIES = [
    ("aster", "Aster Motors", "Company", 0), ("boreal", "Boreal Mobility", "Company", 0),
    ("cell", "Cobalt Cellworks", "Company", 1), ("chip", "Helix Semiconductors", "Company", 1),
    ("drive", "Meridian Drives", "Company", 1), ("refine", "Lumen Refining", "Company", 2),
    ("metals", "Rivet Metals", "Company", 2), ("mine", "Orchard Mine", "Mine", 3),
    ("silicon", "Quartz Materials", "Company", 2), ("port", "Northport", "Location", None),
    ("cells", "Battery cells", "Product", None), ("lithium", "Lithium carbonate", "Material", None),
]
RELATIONS = [
    ("cell", "suppliesTo", "aster", "Cobalt Cellworks supplies battery cells to Aster Motors."),
    ("cell", "suppliesTo", "boreal", "Cobalt Cellworks supplies battery cells to Boreal Mobility."),
    ("chip", "suppliesTo", "aster", "Helix Semiconductors supplies chips to Aster Motors."),
    ("drive", "suppliesTo", "boreal", "Meridian Drives supplies motors to Boreal Mobility."),
    ("refine", "suppliesTo", "cell", "Lumen Refining supplies lithium carbonate to Cobalt Cellworks."),
    ("metals", "suppliesTo", "drive", "Rivet Metals supplies metal parts to Meridian Drives."),
    ("mine", "suppliesTo", "refine", "Orchard Mine supplies ore to Lumen Refining."),
    ("silicon", "suppliesTo", "chip", "Quartz Materials supplies silicon to Helix Semiconductors."),
    ("cell", "produces", "cells", "Cobalt Cellworks produces battery cells."),
    ("refine", "produces", "lithium", "Lumen Refining produces lithium carbonate."),
    ("cell", "locatedIn", "port", "Cobalt Cellworks is located in Northport."),
    ("refine", "owns", "mine", "Lumen Refining owns Orchard Mine."),
]
def canonical(label, name):
    # Demo IDs are explicit fixture IDs; real exports use schain.names.canonical_id.
    return 'demo:'+name
nodes=[{"id":canonical(t,k),"name":n,"type":t,"aliases":[n],"discovery_tier":tier,"is_seed":tier==0} for k,n,t,tier in ENTITIES]
ids={k:n["id"] for (k,*_),n in zip(ENTITIES,nodes)}
edges=[];documents=[]
for i,(source,relation,target,quote) in enumerate(RELATIONS):
    doc_id=f"synthetic-{i+1:02d}"
    evidence={"doc_id":doc_id,"fact_id":f"demo-fact-{i}","quote":quote,"document_title":"Fictional interface fixture", "source_url":None,"content_source":"synthetic_fixture","provider":"demo","chunk_index":0,"model":None,"evidence_quote_grounded":True}
    edges.append({"id":f"demo-edge-{i}","source":ids[source],"target":ids[target],"relation":relation,"source_doc_ids":[doc_id],"support_count":1,"evidence_count":1,"evidence":[evidence],"quality":{"status":"unchecked","evidence_validated":False,"evidence_quote_grounded":True,"contested":False},"expansion_relevant":relation=='suppliesTo'})
    documents.append({"doc_id":doc_id,"title":"Fictional interface fixture","text":quote})
graph={"schema_version":"1.0","metadata":{"demo":True,"generated_at":"2026-09-08","description":"Entirely synthetic. No model calls. Not measured research results."},"nodes":nodes,"edges":edges,"documents":documents}
path=ROOT/'examples/demo_graph.json';path.write_text(json.dumps(graph,indent=2)+'\n',encoding='utf-8')
write_viewer(graph,ROOT/'examples/graph-explorer.html')
print('Created examples/demo_graph.json and examples/graph-explorer.html (synthetic only)')
