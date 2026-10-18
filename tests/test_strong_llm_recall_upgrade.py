"""Offline regression for the strong-LLM-first recall upgrade."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from schain.config import load_config
from schain.expansion.budget import entity_document_limits, remaining_target_reserve
from schain.expansion.escalation import STAGE_C, STAGE_D, STAGE_E, build_escalation_queries
from schain.expansion.loop import _admit_above_target
from schain.expansion.telemetry import premature_discovery_exhaustion_warning
from schain.kg.extract import classify_extraction_error
from schain.kg.list_extract import extract_supplier_list_facts
from schain.kg.relation_evidence import DIRECTION_AMBIGUOUS, UPSTREAM_EXPLICIT, lexical_relation_gate
from schain.kg.validate import is_generic_company_phrase, validate_facts_frame
from schain.llm.adjudication import adjudicate_supply_direction, second_pass_evidence
from schain.llm.query_planner import plan_upstream_queries
from schain.sources.base import SearchResult, _extract_pdf_text
from schain.sources.manifest import RetrievalManifest
from schain.sources.ranking import is_usable_discovery_result
from schain import state
from .conftest import StubLLM


def test_default_document_policy_v031():
    config = load_config("config/expansion.yaml")
    expansion = config["expansion"]
    assert expansion["max_documents"] == 320
    assert expansion.get("max_documents_per_run", expansion["max_documents"]) == 320
    assert config["runtime"]["max_minutes"] == 120 or load_config("config/base.yaml")["runtime"]["max_minutes"] == 120
    assert [entity_document_limits(expansion, t) for t in range(6)] == [
        (24, 36), (14, 24), (10, 18), (8, 14), (6, 10), (0, 0),
    ]
    assert expansion["max_expandable_entities_by_tier"] == {1: 10, 2: 8, 3: 6, 4: 4}
    assert expansion["retrieval"]["max_title_lookups_per_entity"] == 12
    assert expansion["retrieval"]["max_candidates_per_entity"] == 96
    assert expansion["retrieval"]["query_planner_max_categories"] == 8
    assert expansion["retrieval"]["query_planner_max_queries"] == 12
    assert expansion["max_tier"] == 5
    assert expansion["search"]["defaults"]["max_results_by_provider"] == {
        "google_news": 10, "gdelt": 16, "general_web": 18, "wikipedia": 4,
    }


def test_tier5_terminal_and_frontier_caps_do_not_limit_storage(expand_config):
    expansion = expand_config["expansion"]
    assert entity_document_limits(expansion, 5) == (0, 0)
    # Caps only limit expansion search beam — graph may store more suppliers.
    assert expansion["max_expandable_entities_by_tier"][1] == 10


def test_productive_entity_continues_above_target():
    admit, reason, rescue = _admit_above_target(
        used=14, target=14, hard=22, direction=UPSTREAM_EXPLICIT,
        existing_keys={"e1", "e2"}, zero_gain=0, patience=3, min_docs=3,
        rescue_enabled=True, connection=None, config={"expansion": {}},
        max_documents=0,
    )
    assert admit is True
    assert rescue is False
    assert reason is None


def test_zero_yield_escalation_queries_include_planner_and_official(expand_config):
    specs = build_escalation_queries(
        expand_config, "BMW", 0, llm=None,
        stages=["directional", "broad", STAGE_C, STAGE_D, STAGE_E],
    )
    stages = {s["escalation_stage"] for s in specs}
    assert STAGE_C in stages or STAGE_D in stages
    assert STAGE_E in stages
    assert any("suppliers" in s["query"].casefold() for s in specs)


def test_query_planner_schema_and_no_facts(tmp_path, expand_config):
    expand_config["storage"]["state_db"] = str(tmp_path / "state.db")
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps({
        "upstream_categories": ["battery", "semiconductor"],
        "queries": ['"BMW" battery supplier', '"BMW" semiconductor sourcing'],
        "suppliers": ["CATL"],  # must be ignored — planner never inserts facts
    }))
    plan = plan_upstream_queries(
        llm=llm, config=expand_config, entity_name="BMW", tier=0,
    )
    assert "upstream_categories" in plan and "queries" in plan
    assert "suppliers" not in plan or plan.get("source") == "llm"
    assert all("CATL supplies" not in q for q in plan["queries"])


def test_direction_adjudication_ambiguous_rejects(tmp_path, expand_config):
    expand_config["storage"]["state_db"] = str(tmp_path / "state.db")
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps({
        "supported": False, "supplier": None, "customer": None,
        "status": "ambiguous", "reason": "roles unclear",
    }))
    verdict = adjudicate_supply_direction(
        llm, expand_config, subject="A", obj="B",
        evidence="A and B announced a strategic partnership.",
    )
    assert verdict["status"] == "ambiguous"
    assert verdict["supported"] is False


def test_evidence_second_pass_yes(tmp_path, expand_config):
    expand_config["storage"]["state_db"] = str(tmp_path / "state.db")
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content='{"decision":"YES","reason":"explicit"}')
    decision = second_pass_evidence(llm, expand_config, {
        "subject_id": "company:a", "relation": "suppliesTo", "object_id": "company:b",
        "subject_name": "A", "object_name": "B", "subject_type": "Company",
        "object_type": "Company", "evidence": "A supplies batteries to B.",
    })
    assert decision == "YES"


def test_quote_grounding_still_mandatory(ontology, expand_config):
    facts = pd.DataFrame([{
        "doc_id": "d1", "chunk_index": 0,
        "subject_id": "company:catl", "subject_name": "CATL", "subject_type": "Company",
        "relation": "suppliesTo",
        "object_id": "company:bmw", "object_name": "BMW", "object_type": "Company",
        "evidence": "hallucinated quote not in source",
        "source_url": "https://example.test/a",
    }])
    accepted, rejected = validate_facts_frame(
        facts, config=expand_config, ontology=ontology, llm=None,
        document_text="CATL supplies batteries to BMW under a multi-year agreement.",
    )
    assert accepted.empty
    assert "evidence_not_in_source" in rejected.rejection_reason.tolist()[0]


def test_structured_list_extraction_and_no_invented_customer():
    text = "Key suppliers to BMW:\n- CATL\n- Samsung SDI\n- Panasonic\n"
    rows = extract_supplier_list_facts(
        text, doc_id="d1", origin_name="BMW", origin_id="company:bmw",
    )
    assert len(rows) >= 2
    assert all(r["object_name"] == "BMW" for r in rows)
    assert all(r["relation"] == "suppliesTo" for r in rows)
    # Without origin grounding in heading/window, do not invent customer.
    orphan = extract_supplier_list_facts(
        "Key suppliers:\n- MysteryCo\n", doc_id="d2", origin_name="BMW",
    )
    assert orphan == []


def test_generic_auto_supplier_rejected():
    assert is_generic_company_phrase("Auto supplier")
    assert is_generic_company_phrase("major supplier")
    assert not is_generic_company_phrase("Bosch")
    assert not is_generic_company_phrase("Supplier Holdings Inc")


def test_temporal_owns_rejected_completed_accepted():
    ok, reason = lexical_relation_gate(
        "owns", "BMW agreed to buy a stake in the battery firm next year.",
    )
    assert ok is False
    assert "insufficient" in reason
    ok2, reason2 = lexical_relation_gate(
        "owns", "CATL is a subsidiary of Tencent Holdings.",
    )
    assert ok2 is True
    assert reason2 == "positive_relation_language"


def test_gdelt_query_aligned_title_not_eliminated():
    result = SearchResult(
        provider="gdelt",
        query='"BMW" suppliers',
        title="Major battery suppliers expand EV capacity",
        url="https://news.example/a",
        snippet="Major battery suppliers expand EV capacity",
    )
    assert is_usable_discovery_result(result, "BMW") is True


def test_title_lookup_attempted_persisted(tmp_path):
    path = tmp_path / "manifest.csv"
    manifest = RetrievalManifest(path, run_id="r1")
    result = SearchResult(
        provider="google_news", query="q", title="t", url="https://news.google.com/x",
        title_lookup_attempted=True,
    )
    manifest.record_candidate("BMW", 0, "google_news", "NEWS", "q", 1, result)
    manifest.record_content(result)
    manifest.flush()
    summary = manifest.summary()
    assert summary["title_lookups_attempted"] >= 1


def test_premature_discovery_is_diagnostic_only():
    warning = premature_discovery_exhaustion_warning(
        global_cap=320, documents_processed=60, runtime_max_minutes=120,
        elapsed_seconds=17 * 60, stop_reason="frontier_empty",
    )
    assert warning is not None
    assert warning["diagnostic_only"] is True
    assert warning["code"] == "PREMATURE_DISCOVERY_EXHAUSTION"


def test_extraction_retry_categories():
    assert classify_extraction_error(ValueError("JSONDecodeError: Expecting value")) == "invalid_json"
    assert classify_extraction_error(TimeoutError("timed out")) == "timeout"
    assert classify_extraction_error(ConnectionError("server disconnected")) == "transport_error"
    assert classify_extraction_error(ValueError("schema validation error")) == "schema_error"
    assert classify_extraction_error(ValueError("empty response from model")) == "empty_output"


def test_pdf_text_helper_handles_missing_support():
    text, error = _extract_pdf_text(b"not-a-pdf")
    assert text is None
    assert error in ("PARSE_FAILED", "PDF_SUPPORT_MISSING", "EMPTY_CONTENT")


def test_one_source_policy_unchanged(expand_config):
    verification = expand_config.get("verification") or {}
    assert not verification.get("require_independent_sources")
    assert not verification.get("min_sources")


def test_run_snapshot_records_new_semantics(tmp_path, expand_config):
    from schain.run_config import ensure_run_snapshot
    expand_config["storage"]["state_db"] = str(tmp_path / "state.db")
    expand_config["storage"]["run_config"] = str(tmp_path / "run_config.json")
    expand_config["expansion"]["max_documents"] = 320
    snap = ensure_run_snapshot(expand_config)
    assert snap["global_document_hard_cap"] == 320
    assert "query_planner_policy" in snap
    assert "search_escalation_policy" in snap
    assert snap["runtime_cap_minutes"] in (120, None) or snap.get("runtime_cap_minutes") is not None


def test_deeper_tier_reservation_blocks_starvation(expand_config, connection):
    expand_config["expansion"]["max_documents"] = 320
    expand_config["expansion"]["retrieval"]["target_documents_by_tier"] = {
        0: 14, 1: 10, 2: 8, 3: 6, 4: 4, 5: 0,
    }
    expand_config["expansion"]["max_expandable_entities_by_tier"] = {1: 10, 2: 8, 3: 6, 4: 4}
    state.seed_or_ensure = getattr(state, "upsert_frontier", None)
    # Empty frontier → reserve planned targets for seeds + deeper beams.
    reserve = remaining_target_reserve(connection, expand_config)
    assert reserve > 100
