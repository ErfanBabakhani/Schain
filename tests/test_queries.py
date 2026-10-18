"""Tier-aware upstream query generation, depth and per-provider retrieval limits."""

from __future__ import annotations

import pytest

from schain.config import query_limit, resolve_tier_queries
from schain.expansion.queries import generate_queries

NEWS_PROVIDERS = ("google_news", "gdelt")

# Upstream-focused production templates plus high-yield list/report queries.
EXPECTED_TEMPLATES = {
    tier: [
        'supplies "{entity}"',
        '"supplier to" "{entity}"',
        '"{entity}" sources from supplier',
        '"{entity}" purchases from supplier',
        '"{entity}" procurement supplier',
        '"{entity}" suppliers',
        '"{entity}" key suppliers',
        '"{entity}" major suppliers',
        '"{entity}" component suppliers',
        '"{entity}" supply chain',
        '"{entity}" supplier list',
        '"{entity}" key suppliers list',
        '"{entity}" supplier award',
        '"{entity}" procurement report',
        '"{entity}" sourcing report',
        '"{entity}" sustainability report suppliers',
        '"{entity}" annual report suppliers',
        '"{entity}" supply chain report',
        '"{entity}" filetype:pdf suppliers',
    ]
    for tier in range(5)
}

EXPECTED_LIMITS = {
    tier: {"google_news": 10, "gdelt": 16, "general_web": 18, "wikipedia": 4}
    for tier in range(5)
}


def test_production_depth_is_five_tiers(expand_config):
    assert expand_config["expansion"]["max_tier"] == 5


@pytest.mark.parametrize("tier", sorted(EXPECTED_TEMPLATES))
def test_tiers_below_terminal_are_searched(tier, expand_config):
    assert generate_queries(expand_config, "BMW", tier), f"tier {tier} produced no queries"


def test_tier_five_and_beyond_are_terminal(expand_config):
    for tier in (5, 6):
        assert resolve_tier_queries(expand_config, tier) == []
        assert generate_queries(expand_config, "Some Deep Mine", tier) == []


def test_beyond_the_configured_depth_nothing_is_generated(expand_config):
    assert generate_queries(expand_config, "BMW", 7) == []


@pytest.mark.parametrize("tier,templates", sorted(EXPECTED_TEMPLATES.items()))
def test_tier_templates_match_the_configured_set(tier, templates, expand_config):
    assert [spec["template"] for spec in generate_queries(expand_config, "X", tier)] == templates


def test_entity_is_substituted_into_every_template(expand_config):
    queries = [spec["query"] for spec in generate_queries(expand_config, "BMW", 0)]
    assert queries[:3] == [
        'supplies "BMW"',
        '"supplier to" "BMW"',
        '"BMW" sources from supplier',
    ]
    assert all("{entity}" not in query for query in queries)


def test_upstream_templates_avoid_ambiguous_sourcing_primary(expand_config):
    templates = [spec["template"] for spec in generate_queries(expand_config, "X", 0)]
    assert '"{entity}" sourcing' not in templates
    assert '"{entity}" "supply agreement"' not in templates
    assert any("supplies" in t for t in templates)
    assert any("sources from" in t for t in templates)
    assert '"{entity}" suppliers' in templates
    assert '"{entity}" key suppliers' in templates
    assert '"{entity}" component suppliers' in templates


def test_deep_queries_do_not_assume_an_industrial_role(expand_config):
    shallow = [spec["template"] for spec in generate_queries(expand_config, "Acme", 0)]
    deep = [spec["template"] for spec in generate_queries(expand_config, "Acme", 4)]
    assert shallow == deep
    assert all("mines" not in template and "refining" not in template for template in deep)
    # The buyer phrase is unquoted, allowing inserted product/material words.
    assert '"{entity}" sources from supplier' in deep


def test_template_without_placeholder_is_used_literally(caplog):
    config = {"expansion": {"search": {"defaults": {"max_results": 5},
                                       "tiers": {0: {"queries": [{"query": "battery news"}]}}}}}
    spec = generate_queries(config, "BMW", 0)[0]
    assert spec["query"] == "battery news"
    assert "placeholder" in caplog.text


@pytest.mark.parametrize("tier,limits", sorted(EXPECTED_LIMITS.items()))
def test_limits_are_per_provider_per_query(tier, limits, expand_config):
    for spec in generate_queries(expand_config, "BMW", tier):
        assert {provider: query_limit(spec, provider) for provider in spec["providers"]} == limits


def test_news_providers_are_competitive_with_general_web(expand_config):
    spec = generate_queries(expand_config, "BMW", 0)[0]
    assert query_limit(spec, "general_web") >= query_limit(spec, "google_news")
    assert query_limit(spec, "gdelt") >= query_limit(spec, "google_news")
    assert query_limit(spec, "google_news") > query_limit(spec, "wikipedia")


def test_all_four_source_families_are_queried(expand_config):
    providers = generate_queries(expand_config, "BMW", 0)[0]["providers"]
    assert set(providers) == {"google_news", "gdelt", "general_web", "wikipedia"}


def test_wikipedia_stays_small_at_every_tier(expand_config):
    for tier in EXPECTED_LIMITS:
        for spec in generate_queries(expand_config, "BMW", tier):
            assert query_limit(spec, "wikipedia") == 4


def test_per_query_override_beats_the_tier_default():
    config = {"expansion": {"search": {
        "defaults": {"max_results": 20, "providers": ["gdelt"]},
        "tiers": {0: {"defaults": {"max_results": 50},
                      "queries": [{"query": "a"}, {"query": "b", "max_results": 15}]}},
    }}}
    specs = {spec["query"]: spec for spec in resolve_tier_queries(config, 0)}
    assert specs["a"]["max_results"] == 50
    assert specs["b"]["max_results"] == 15


def test_tier_without_defaults_falls_back_to_the_global_value():
    config = {"expansion": {"search": {
        "defaults": {"max_results": 20, "providers": ["gdelt"]},
        "tiers": {2: {"queries": [{"query": "a"}]}},
    }}}
    assert resolve_tier_queries(config, 2)[0]["max_results"] == 20


def test_per_provider_limits_inherit_independently():
    config = {"expansion": {"search": {
        "defaults": {"max_results": 20, "providers": ["gdelt", "wikipedia"],
                     "max_results_by_provider": {"wikipedia": 10}},
        "tiers": {1: {"defaults": {"max_results": 50}, "queries": [{"query": "a"}]}},
    }}}
    spec = resolve_tier_queries(config, 1)[0]
    assert query_limit(spec, "gdelt") == 50
    assert query_limit(spec, "wikipedia") == 10


def test_no_cumulative_target_mechanism_exists(expand_config):
    for spec in generate_queries(expand_config, "BMW", 0):
        assert "target_relevant_documents" not in spec
    assert "target_relevant_documents" not in expand_config["expansion"]


def test_search_volume_is_node_centric_not_pairwise(expand_config):
    entities = ["BMW", "Toyota", "Tesla Inc.", "BYD Co., Ltd.", "CATL", "Panasonic"]
    per_entity = len(generate_queries(expand_config, "BMW", 0))
    total = sum(len(generate_queries(expand_config, name, 0)) for name in entities)

    assert total == len(entities) * per_entity
    pairwise = len(entities) * (len(entities) - 1) * per_entity
    assert total < pairwise
    for name in entities:
        for spec in generate_queries(expand_config, name, 0):
            assert sum(other in spec["query"] for other in entities) == 1
