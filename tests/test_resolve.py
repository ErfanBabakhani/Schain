"""Prompt 2: cache, deterministic normalization, and not over-merging."""

from __future__ import annotations

import json

from neo4j_graphrag.components.types import Neo4jGraph, Neo4jNode
from neo4j_graphrag.llm import LLMResponse

from schain.kg.resolve import resolve_entities
from schain.names import mention_key


def graph(*named_nodes):
    """named_nodes: (label, name) pairs."""
    return Neo4jGraph(
        nodes=[
            Neo4jNode(id=f"chunk:{index}", label=label, properties={"name": name})
            for index, (label, name) in enumerate(named_nodes)
        ]
    )


class GroupingLLM:
    """Returns fixed groups and counts calls."""

    def __init__(self, groups, per_label=None):
        self.groups = groups
        self.per_label = per_label or {}
        self.calls = 0
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        for label, groups in self.per_label.items():
            if f"list of {label} names" in prompt:
                return LLMResponse(content=json.dumps({"groups": groups}))
        return LLMResponse(content=json.dumps({"groups": self.groups}))


def cache_from(entries):
    return {mention_key(e["label"], e["mention"]): e for e in entries}


def test_deterministic_variants_never_reach_the_llm():
    llm = GroupingLLM([["BMW AG"]])
    resolution, _ = resolve_entities(graph(("Company", "BMW AG"), ("Company", "bmw ag.")), llm=llm)
    assert len({entry["canonical_id"] for entry in resolution.values()}) == 1
    # only one distinct name was offered, so no grouping call was needed at all
    assert llm.calls == 0


def test_llm_merges_abbreviation_with_legal_name():
    llm = GroupingLLM([["CATL", "Contemporary Amperex Technology Co. Limited"], ["BMW"]])
    resolution, new = resolve_entities(
        graph(("Company", "CATL"),
              ("Company", "Contemporary Amperex Technology Co. Limited"),
              ("Company", "BMW")),
        llm=llm,
    )
    ids = {entry["mention"]: entry["canonical_id"] for entry in resolution.values()}
    assert ids["CATL"] == ids["Contemporary Amperex Technology Co. Limited"]
    assert ids["BMW"] != ids["CATL"]
    assert llm.calls == 1
    assert {entry["mention"] for entry in new} == set(ids)


def test_canonical_id_comes_from_group_membership_not_model_wording():
    """The longest mention names the group, so the id is reproducible across runs."""
    llm = GroupingLLM([["CATL", "Contemporary Amperex Technology Co. Limited"]])
    resolution, _ = resolve_entities(
        graph(("Company", "CATL"),
              ("Company", "Contemporary Amperex Technology Co. Limited")),
        llm=llm,
    )
    assert {e["canonical_id"] for e in resolution.values()} == {
        "company:contemporary-amperex-technology"
    }


def test_cached_mentions_cost_no_llm_call():
    cache = cache_from([
        {"label": "Company", "mention": "CATL", "canonical_id": "company:catl",
         "canonical_name": "CATL", "method": "llm"},
    ])
    llm = GroupingLLM([])
    resolution, new = resolve_entities(graph(("Company", "CATL")), llm=llm, cache=cache)
    assert llm.calls == 0
    assert new == []
    assert list(resolution.values())[0]["method"] == "cache"


def test_new_alias_attaches_to_an_existing_canonical_entity():
    """A later run must not create a second node for the same company."""
    cache = cache_from([
        {"label": "Company", "mention": "CATL",
         "canonical_id": "company:contemporary-amperex-technology",
         "canonical_name": "Contemporary Amperex Technology Co. Limited", "method": "llm"},
    ])
    # A real acronym may attach; a qualified subsidiary name must stay separate.
    llm = GroupingLLM([["CATL", "Contemporary Amperex Technology Co. Limited"]])
    cache = {"company:known-parent": next(iter(cache.values()))}
    resolution, _ = resolve_entities(graph(("Company", "CATL")), llm=llm, cache=cache)
    assert list(resolution.values())[0]["canonical_id"] == (
        "company:contemporary-amperex-technology"
    )


def test_entity_types_are_resolved_separately():
    llm = GroupingLLM([], per_label={"Company": [["BMW"], ["Bosch"]],
                                     "Location": [["Germany"], ["Munich"]]})
    resolve_entities(
        graph(("Company", "BMW"), ("Company", "Bosch"),
              ("Location", "Germany"), ("Location", "Munich")),
        llm=llm,
    )
    assert llm.calls == 2  # one call per entity type, never a mixed list
    assert not any("Germany" in p for p in llm.prompts if "Company names" in p)


def test_unparseable_output_keeps_entities_apart():
    """A broken Prompt 2 response must not merge unrelated companies."""

    class BrokenLLM:
        def invoke(self, prompt):
            return LLMResponse(content="not json at all")

    resolution, _ = resolve_entities(
        graph(("Company", "BMW"), ("Company", "Bosch")), llm=BrokenLLM()
    )
    assert len({entry["canonical_id"] for entry in resolution.values()}) == 2


def test_names_invented_by_the_model_are_ignored():
    llm = GroupingLLM([["BMW", "A Company That Was Not In The List"]])
    resolution, _ = resolve_entities(graph(("Company", "BMW")), llm=llm)
    assert {e["canonical_name"] for e in resolution.values()} == {"BMW"}


def test_dropped_names_are_not_lost():
    """A model that omits a name still leaves that entity resolved, on its own."""
    llm = GroupingLLM([["BMW"]])  # Bosch omitted from the response
    resolution, _ = resolve_entities(
        graph(("Company", "BMW"), ("Company", "Bosch")), llm=llm
    )
    assert {e["mention"] for e in resolution.values()} == {"BMW", "Bosch"}
    assert len({e["canonical_id"] for e in resolution.values()}) == 2


def test_without_an_llm_only_deterministic_grouping_happens():
    resolution, _ = resolve_entities(
        graph(("Company", "CATL"),
              ("Company", "Contemporary Amperex Technology Co. Limited")),
        llm=None,
    )
    assert len({e["canonical_id"] for e in resolution.values()}) == 2


def test_nodes_without_a_name_are_skipped():
    bad = Neo4jGraph(nodes=[Neo4jNode(id="c:0", label="Company", properties={})])
    resolution, new = resolve_entities(bad, llm=None)
    assert resolution == {} and new == []


# --- configured known aliases ------------------------------------------------------


ALIASES = {"Company": {"CATL": "Contemporary Amperex Technology Co. Limited",
                       "Nornickel": "Norilsk Nickel"}}


class ExplodingLLM:
    """Any call is a failure: the alias stage must be purely deterministic."""

    def invoke(self, prompt):
        raise AssertionError("Prompt 2 must not be called for a configured alias")


def test_configured_alias_collapses_without_any_llm_call():
    resolution, _ = resolve_entities(
        graph(("Company", "CATL"),
              ("Company", "Contemporary Amperex Technology Co. Limited")),
        llm=ExplodingLLM(), aliases=ALIASES,
    )
    assert len({entry["canonical_id"] for entry in resolution.values()}) == 1
    assert all(entry["method"] == "deterministic" for entry in resolution.values())


def test_alias_alone_still_resolves_to_the_configured_canonical_name():
    resolution, _ = resolve_entities(graph(("Company", "CATL")), llm=None, aliases=ALIASES)
    entry = list(resolution.values())[0]
    assert entry["canonical_name"] == "Contemporary Amperex Technology Co. Limited"
    assert entry["canonical_id"] == "company:contemporary-amperex-technology"
    assert entry["mention"] == "CATL"          # the raw mention is still recorded


def test_alias_matching_ignores_case_and_punctuation():
    resolution, _ = resolve_entities(graph(("Company", "catl.")), llm=None, aliases=ALIASES)
    assert list(resolution.values())[0]["canonical_id"] == (
        "company:contemporary-amperex-technology"
    )


def test_aliases_are_scoped_to_their_entity_type():
    """A Company alias must not rewrite a Location of the same name."""
    resolution, _ = resolve_entities(graph(("Location", "CATL")), llm=None, aliases=ALIASES)
    assert list(resolution.values())[0]["canonical_id"] == "location:catl"


def test_unaliased_names_still_reach_the_llm():
    llm = GroupingLLM([["Bosch"], ["Continental"]])
    resolve_entities(graph(("Company", "Bosch"), ("Company", "Continental")),
                     llm=llm, aliases=ALIASES)
    assert llm.calls == 1


def test_the_shipped_alias_map_is_conservative(ontology):
    """Guard against someone mapping a subsidiary onto its parent."""
    aliases = ontology.get("aliases", {}).get("Company", {})
    assert aliases, "the shipped ontology should carry some aliases"
    forbidden = {"Panasonic Energy": "Panasonic", "CATL Ningde": "CATL"}
    for alias, target in forbidden.items():
        assert aliases.get(alias) != target
