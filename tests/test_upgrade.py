"""Regression checks for source grounding, exact evaluation and offline export."""
import json
import pandas as pd
import pytest
from schain.kg.validate import validate_facts_frame
from schain.validation.gold import evaluate_gold
from schain.viewer import write_viewer, validate_graph
from .test_quality import _accepted_supply


def test_hallucinated_quote_rejected_before_llm(ingest_config, ontology, stub_llm):
    facts = pd.DataFrame([_accepted_supply("CATL", "BMW", "d")])
    accepted, rejected = validate_facts_frame(facts, config=ingest_config, ontology=ontology, llm=stub_llm,
                                             document_text="CATL does not supply BMW.")
    assert accepted.empty
    assert rejected.iloc[0].rejection_reason == "evidence_not_in_source"
    assert not stub_llm.prompts


def test_literal_quote_matches_normalized_whitespace(ingest_config, ontology, stub_llm):
    facts = pd.DataFrame([_accepted_supply("CATL", "BMW", "d")])
    accepted, rejected = validate_facts_frame(facts, config=ingest_config, ontology=ontology, llm=stub_llm,
                                             document_text="CATL\n supplies   to BMW.")
    assert rejected.empty and len(accepted) == 1
    assert bool(accepted.iloc[0].evidence_quote_grounded)


def test_no_model_does_not_claim_validation(ingest_config, ontology):
    facts = pd.DataFrame([_accepted_supply("CATL", "BMW", "d")])
    accepted, _ = validate_facts_frame(facts, config=ingest_config, ontology=ontology, document_text="CATL supplies to BMW.")
    assert not bool(accepted.iloc[0].evidence_validated)
    assert accepted.iloc[0].validation_status == "unchecked"


def test_evaluation_direction_negatives_and_document_scope():
    edge = {"subject_id":"A", "relation":"suppliesTo", "object_id":"B"}
    gold = {"documents":[{"doc_id":"positive", "triples":[edge]}, {"doc_id":"negative", "triples":[]}]}
    pred = [{"doc_id":"positive", **edge}, {"doc_id":"positive", **edge},
            {"doc_id":"negative", **edge}, {"doc_id":"outside", **edge},
            {"doc_id":"positive", "subject_id":"B", "relation":"suppliesTo", "object_id":"A"}]
    r = evaluate_gold(gold, pred)
    assert r["micro"] == {"true_positive":1,"false_positive":2,"false_negative":0,"precision":1/3,"recall":1.0,"f1":.5}
    assert r["excluded_prediction_documents"] == 1
    assert evaluate_gold(gold, [])["micro"]["recall"] == 0


def test_viewer_is_self_contained_and_source_cannot_inject_script(tmp_path):
    payload = {"nodes":[{"id":"a","name":"</script><script>alert(1)</script>"}],"edges":[]}
    path = write_viewer(payload, tmp_path / "graph.html")
    html = path.read_text()
    assert "</script><script>alert(1)</script>" not in html
    assert "__GRAPH_DATA__" not in html and "/*__SCRIPT__*/" not in html
    embedded = html.split('id="graphData">')[1].split('</script>')[0]
    assert json.loads(embedded) == validate_graph(payload)
    assert json.loads(embedded)["nodes"][0]["name"] == payload["nodes"][0]["name"]
    assert 'src="https://' not in html


def test_viewer_rejects_dangling_edges():
    with pytest.raises(ValueError, match="existing node"):
        validate_graph({"nodes":[{"id":"a"}], "edges":[{"id":"e","source":"a","target":"b"}]})
