"""Tier-aware query generation."""

from __future__ import annotations

from schain.expansion.queries import generate_queries


def test_tier0_templates_produce_the_documented_queries(expand_config):
    queries = [spec["query"] for spec in generate_queries(expand_config, "BMW", 0)]
    assert queries == [
        '"BMW" suppliers',
        '"BMW" sourcing',
        '"BMW" battery suppliers',
        '"BMW" semiconductor suppliers',
    ]


def test_tier1_uses_a_different_template_set(expand_config):
    queries = [spec["query"] for spec in generate_queries(expand_config, "CATL", 1)]
    assert queries == [
        '"CATL" suppliers',
        '"CATL" component suppliers',
        '"CATL" raw material suppliers',
    ]


def test_tier2_moves_further_upstream(expand_config):
    queries = [spec["query"] for spec in generate_queries(expand_config, "Ganfeng Lithium", 2)]
    assert '"Ganfeng Lithium" mines' in queries
    assert '"Ganfeng Lithium" lithium suppliers' in queries


def test_limits_are_carried_per_query(expand_config):
    specs = {spec["query"]: spec for spec in generate_queries(expand_config, "BMW", 0)}
    assert specs['"BMW" suppliers']["max_results"] == 30
    assert specs['"BMW" semiconductor suppliers']["max_results"] == 15


def test_query_count_is_linear_in_entities(expand_config):
    """Node-centric expansion: cost is entities x templates, never entity pairs."""
    entities = ["BMW", "Toyota", "Tesla Inc.", "BYD Co., Ltd."]
    total = sum(len(generate_queries(expand_config, name, 0)) for name in entities)
    templates = len(generate_queries(expand_config, "BMW", 0))
    assert total == len(entities) * templates
    assert total < len(entities) ** 2 * templates  # nothing pairwise is generated


def test_template_without_placeholder_is_used_literally(caplog):
    config = {"expansion": {"search": {"defaults": {"max_results": 5},
                                       "tiers": {0: {"queries": [{"query": "battery news"}]}}}}}
    spec = generate_queries(config, "BMW", 0)[0]
    assert spec["query"] == "battery news"
    assert "placeholder" in caplog.text


def test_unconfigured_tier_generates_nothing(expand_config):
    assert generate_queries(expand_config, "BMW", 7) == []
