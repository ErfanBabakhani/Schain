"""Graph writes: idempotent MERGE, ontology-constrained labels, batching."""

from __future__ import annotations

import pandas as pd
import pytest

from schain.kg.facts import FACT_COLUMNS
from schain.kg.writer import (
    ensure_constraints,
    facts_to_graph_rows,
    merge_nodes,
    merge_relationships,
)


class RecordingDriver:
    """Captures the Cypher and parameters the writer would send."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute_query(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return None

    @property
    def queries(self) -> list[str]:
        return [query for query, _ in self.calls]


def node(canonical_id="company:bmw", label="Company", name="BMW", aliases=("BMW",)):
    return {"canonical_id": canonical_id, "label": label, "name": name,
            "aliases": list(aliases), "doc_id": "d1"}


def relationship(rel_type="suppliesTo"):
    return {"type": rel_type, "start_id": "company:catl", "end_id": "company:bmw",
            "evidence": "CATL supplies BMW.", "doc_id": "d1"}


def facts_frame(**overrides):
    row = {column: None for column in FACT_COLUMNS}
    row.update(
        fact_id="f1", doc_id="d1", subject_id="company:catl", subject_type="Company",
        subject_name="CATL", subject_mention="CATL", relation="suppliesTo",
        object_id="company:bmw", object_type="Company", object_name="BMW",
        object_mention="BMW AG", evidence="CATL supplies BMW.",
    )
    row.update(overrides)
    return pd.DataFrame([row])


def test_writes_use_merge_and_never_create():
    driver = RecordingDriver()
    merge_nodes(driver, "neo4j", [node()], ["Company"])
    merge_relationships(driver, "neo4j", [relationship()], ["suppliesTo"])
    assert driver.queries
    assert all("MERGE" in query for query in driver.queries)
    assert not any("CREATE (" in query for query in driver.queries)


def test_nodes_merge_on_canonical_id_with_both_labels():
    driver = RecordingDriver()
    merge_nodes(driver, "neo4j", [node()], ["Company"])
    query = driver.queries[0]
    assert "MERGE (e:Company:Entity {canonical_id: row.canonical_id})" in query
    # aliases accumulate as a set union rather than being overwritten
    assert "WHERE NOT a IN row.aliases" in query


def test_relationships_merge_and_accumulate_provenance():
    driver = RecordingDriver()
    merge_relationships(driver, "neo4j", [relationship()], ["suppliesTo"])
    query = driver.queries[0]
    assert "MERGE (a)-[r:suppliesTo]->(b)" in query
    assert "MATCH (a:Entity {canonical_id: row.start_id})" in query
    assert "r.source_doc_ids" in query and "WHERE d <> row.doc_id" in query


def test_labels_outside_the_ontology_are_refused():
    driver = RecordingDriver()
    with pytest.raises(ValueError, match="unknown entity label"):
        merge_nodes(driver, "neo4j", [node(label="Supplier")], ["Company", "Mine"])
    assert driver.calls == []  # nothing was sent


def test_relation_types_outside_the_ontology_are_refused():
    driver = RecordingDriver()
    with pytest.raises(ValueError, match="unknown relation type"):
        merge_relationships(driver, "neo4j", [relationship("SUPPLIES_TO")], ["suppliesTo"])
    assert driver.calls == []


def test_rows_are_grouped_by_label_and_type():
    driver = RecordingDriver()
    merge_nodes(
        driver, "neo4j",
        [node(), node("mine:greenbushes", "Mine", "Greenbushes", ("Greenbushes",))],
        ["Company", "Mine"],
    )
    assert len(driver.calls) == 2
    assert {"Company", "Mine"} == {
        "Company" if ":Company:" in query else "Mine" for query in driver.queries
    }


def test_large_writes_are_batched():
    driver = RecordingDriver()
    nodes = [node(f"company:c{i}", name=f"C{i}", aliases=(f"C{i}",)) for i in range(250)]
    written = merge_nodes(driver, "neo4j", nodes, ["Company"], batch_size=100)
    assert written == 250
    assert [len(kwargs["rows"]) for _, kwargs in driver.calls] == [100, 100, 50]


def test_constraint_is_created_idempotently():
    driver = RecordingDriver()
    ensure_constraints(driver, "neo4j")
    query = driver.queries[0]
    assert "IF NOT EXISTS" in query and "n.canonical_id IS UNIQUE" in query


def test_facts_become_deduplicated_node_and_relationship_rows():
    frame = pd.concat([facts_frame(), facts_frame(fact_id="f2")], ignore_index=True)
    nodes, relationships = facts_to_graph_rows(frame)
    assert len(nodes) == 2                      # CATL and BMW, once each
    assert len(relationships) == 2              # one row per fact
    aliases = {n["canonical_id"]: n["aliases"] for n in nodes}
    assert aliases["company:bmw"] == ["BMW AG"]  # the mention is kept as an alias


def test_writer_does_not_need_a_resolution_map():
    """After a restart the resolution map is gone, but the facts table is enough."""
    nodes, relationships = facts_to_graph_rows(facts_frame(), resolution=None)
    assert [n["canonical_id"] for n in nodes] == ["company:catl", "company:bmw"]
    assert relationships[0]["type"] == "suppliesTo"


# --- exactly what V1 does to the graph after extraction ------------------------------


def test_a_new_canonical_node_is_inserted_and_an_existing_one_is_reused():
    """MERGE on canonical_id: first sighting creates, later ones attach to the same node."""
    driver = RecordingDriver()
    merge_nodes(driver, "neo4j", [node()], ["Company"])
    query, params = driver.calls[0]
    assert "MERGE (e:Company:Entity {canonical_id: row.canonical_id})" in query
    assert "ON CREATE SET" in query          # first sighting only
    assert params["rows"][0]["canonical_id"] == "company:bmw"
    # the id is the whole identity rule: nothing else decides whether it is the same node
    assert "MATCH" not in query and "CREATE (" not in query


def test_a_repeated_relationship_does_not_become_a_second_edge():
    driver = RecordingDriver()
    merge_relationships(driver, "neo4j", [relationship(), relationship()], ["suppliesTo"])
    query, params = driver.calls[0]
    assert query.count("MERGE (a)-[r:suppliesTo]->(b)") == 1
    # both rows are UNWOUND through the same MERGE, so they converge on one edge
    assert len(params["rows"]) == 2
    assert "UNWIND $rows AS row" in query


def test_evidence_from_several_documents_accumulates_on_one_edge():
    """Many documents may support one relationship; that must not fan out into edges."""
    frame = pd.concat([
        facts_frame(fact_id="f1", doc_id="doc-a", evidence="A says CATL supplies BMW."),
        facts_frame(fact_id="f2", doc_id="doc-b", evidence="B says CATL supplies BMW."),
    ], ignore_index=True)
    nodes, relationships = facts_to_graph_rows(frame)

    assert len(nodes) == 2                                   # CATL and BMW, once each
    assert {(r["start_id"], r["type"], r["end_id"]) for r in relationships} == {
        ("company:catl", "suppliesTo", "company:bmw")
    }
    driver = RecordingDriver()
    merge_relationships(driver, "neo4j", relationships, ["suppliesTo"])
    query, _ = driver.calls[0]
    # provenance is a set union, so a second document adds a doc id, not an edge
    assert "r.source_doc_ids = [d IN coalesce(r.source_doc_ids, []) " in query
    assert "WHERE d <> row.doc_id] + [row.doc_id]" in query


def test_the_writer_contains_no_deletion_or_temporal_logic():
    """V1 merges structure and stores evidence. It never revises the real world."""
    from schain.config import REPO_ROOT

    source = (REPO_ROOT / "schain" / "kg" / "writer.py").read_text(encoding="utf-8")
    upper = source.upper()
    for forbidden in ("DELETE", "DETACH", "REMOVE ", "SET r = ", "SET e = "):
        assert forbidden not in upper, f"writer must not contain {forbidden!r}"
    for temporal in ("valid_from", "valid_to", "terminated", "deactivat", "superseded",
                     "is_active", "end_date"):
        assert temporal not in source.lower(), f"writer must not track {temporal!r}"


def test_no_module_deletes_nodes_or_relationships():
    """Guards the whole package against edge/node removal creeping in."""
    from schain.config import REPO_ROOT

    for path in (REPO_ROOT / "schain").rglob("*.py"):
        text = path.read_text(encoding="utf-8").upper()
        assert "DETACH DELETE" not in text, f"{path.name} deletes graph data"


def test_node_properties_are_not_blindly_overwritten():
    """A later document adds aliases; it does not reset the entity's identity."""
    driver = RecordingDriver()
    merge_nodes(driver, "neo4j", [node()], ["Company"])
    query, _ = driver.calls[0]
    assert "e.first_seen_doc_id = row.doc_id" in query      # recorded once, on create
    assert query.count("ON CREATE SET") == 1
    # aliases accumulate as a union rather than replacing the stored list
    assert "[a IN coalesce(e.aliases, []) WHERE NOT a IN row.aliases]" in query
