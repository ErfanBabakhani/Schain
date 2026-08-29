"""Tier-aware query generation, depth and per-provider retrieval limits."""

from __future__ import annotations

import pytest

from schain.config import query_limit, resolve_tier_queries
from schain.expansion.queries import generate_queries

NEWS_PROVIDERS = ("google_news", "gdelt")

# The production template set, tier by tier.
EXPECTED_TEMPLATES = {
    0: ['"{entity}" suppliers', '"{entity}" sourcing', '"{entity}" "supply agreement"',
        '"{entity}" battery suppliers', '"{entity}" semiconductor suppliers',
        '"{entity}" raw material supply'],
    1: ['"{entity}" suppliers', '"{entity}" sourcing', '"{entity}" "supply agreement"',
        '"{entity}" component suppliers', '"{entity}" battery material suppliers',
        '"{entity}" raw material suppliers'],
    2: ['"{entity}" suppliers', '"{entity}" sourcing', '"{entity}" "supply agreement"',
        '"{entity}" material suppliers', '"{entity}" precursor suppliers',
        '"{entity}" feedstock suppliers'],
    3: ['"{entity}" suppliers', '"{entity}" raw material suppliers', '"{entity}" feedstock',
        '"{entity}" precursor', '"{entity}" refining', '"{entity}" "offtake agreement"'],
    4: ['"{entity}" suppliers', '"{entity}" feedstock suppliers',
        '"{entity}" refinery suppliers', '"{entity}" concentrate suppliers',
        '"{entity}" ore suppliers', '"{entity}" "offtake agreement"'],
    5: ['"{entity}" suppliers', '"{entity}" raw material sourcing',
        '"{entity}" mining suppliers', '"{entity}" mines', '"{entity}" extraction',
        '"{entity}" "offtake agreement"'],
}

# Per-provider candidate ceilings, per query.
EXPECTED_LIMITS = {
    0: {"google_news": 50, "gdelt": 50, "general_web": 30, "wikipedia": 10},
    1: {"google_news": 50, "gdelt": 50, "general_web": 30, "wikipedia": 10},
    2: {"google_news": 50, "gdelt": 50, "general_web": 30, "wikipedia": 10},
    3: {"google_news": 40, "gdelt": 40, "general_web": 25, "wikipedia": 10},
    4: {"google_news": 30, "gdelt": 30, "general_web": 20, "wikipedia": 10},
    5: {"google_news": 30, "gdelt": 30, "general_web": 20, "wikipedia": 10},
}


# --- depth ---------------------------------------------------------------------------


def test_production_depth_is_six_tiers(expand_config):
    assert expand_config["expansion"]["max_tier"] == 6


@pytest.mark.parametrize("tier", sorted(EXPECTED_TEMPLATES))
def test_tiers_zero_to_five_are_searched(tier, expand_config):
    assert generate_queries(expand_config, "BMW", tier), f"tier {tier} produced no queries"


def test_tier_six_is_terminal_and_produces_no_queries(expand_config):
    """Tier-6 entities are stored, but nothing is ever searched for them."""
    assert resolve_tier_queries(expand_config, 6) == []
    assert generate_queries(expand_config, "Some Deep Mine", 6) == []


def test_beyond_the_configured_depth_nothing_is_generated(expand_config):
    assert generate_queries(expand_config, "BMW", 7) == []


# --- templates -----------------------------------------------------------------------


@pytest.mark.parametrize("tier,templates", sorted(EXPECTED_TEMPLATES.items()))
def test_tier_templates_match_the_configured_set(tier, templates, expand_config):
    assert [spec["template"] for spec in generate_queries(expand_config, "X", tier)] == templates


def test_entity_is_substituted_into_every_template(expand_config):
    queries = [spec["query"] for spec in generate_queries(expand_config, "BMW", 0)]
    assert queries[:3] == ['"BMW" suppliers', '"BMW" sourcing', '"BMW" "supply agreement"']
    assert all("{entity}" not in query for query in queries)


def test_upper_tiers_move_upstream_in_vocabulary(expand_config):
    tier5 = [spec["query"] for spec in generate_queries(expand_config, "Ganfeng Lithium", 5)]
    assert '"Ganfeng Lithium" mines' in tier5
    assert '"Ganfeng Lithium" "offtake agreement"' in tier5


def test_template_without_placeholder_is_used_literally(caplog):
    config = {"expansion": {"search": {"defaults": {"max_results": 5},
                                       "tiers": {0: {"queries": [{"query": "battery news"}]}}}}}
    spec = generate_queries(config, "BMW", 0)[0]
    assert spec["query"] == "battery news"
    assert "placeholder" in caplog.text


# --- limits --------------------------------------------------------------------------


@pytest.mark.parametrize("tier,limits", sorted(EXPECTED_LIMITS.items()))
def test_limits_are_per_provider_per_query(tier, limits, expand_config):
    for spec in generate_queries(expand_config, "BMW", tier):
        assert {provider: query_limit(spec, provider) for provider in spec["providers"]} == limits


def test_news_providers_get_the_largest_share(expand_config):
    spec = generate_queries(expand_config, "BMW", 0)[0]
    for provider in NEWS_PROVIDERS:
        assert query_limit(spec, provider) > query_limit(spec, "general_web")
        assert query_limit(spec, provider) > query_limit(spec, "wikipedia")


def test_all_four_source_families_are_queried(expand_config):
    providers = generate_queries(expand_config, "BMW", 0)[0]["providers"]
    assert set(providers) == {"google_news", "gdelt", "general_web", "wikipedia"}


def test_wikipedia_stays_small_at_every_tier(expand_config):
    for tier in EXPECTED_LIMITS:
        for spec in generate_queries(expand_config, "BMW", tier):
            assert query_limit(spec, "wikipedia") == 10


# --- precedence ----------------------------------------------------------------------


def test_per_query_override_beats_the_tier_default():
    config = {"expansion": {"search": {
        "defaults": {"max_results": 20, "providers": ["gdelt"]},
        "tiers": {0: {"defaults": {"max_results": 50},
                      "queries": [{"query": "a"}, {"query": "b", "max_results": 15}]}},
    }}}
    specs = {spec["query"]: spec for spec in resolve_tier_queries(config, 0)}
    assert specs["a"]["max_results"] == 50    # tier default
    assert specs["b"]["max_results"] == 15    # query override


def test_tier_without_defaults_falls_back_to_the_global_value():
    config = {"expansion": {"search": {
        "defaults": {"max_results": 20, "providers": ["gdelt"]},
        "tiers": {2: {"queries": [{"query": "a"}]}},
    }}}
    assert resolve_tier_queries(config, 2)[0]["max_results"] == 20


def test_per_provider_limits_inherit_independently():
    """A tier overriding only max_results keeps the global per-provider entries."""
    config = {"expansion": {"search": {
        "defaults": {"max_results": 20, "providers": ["gdelt", "wikipedia"],
                     "max_results_by_provider": {"wikipedia": 10}},
        "tiers": {1: {"defaults": {"max_results": 50}, "queries": [{"query": "a"}]}},
    }}}
    spec = resolve_tier_queries(config, 1)[0]
    assert query_limit(spec, "gdelt") == 50       # from the tier default
    assert query_limit(spec, "wikipedia") == 10   # inherited global per-provider entry


def test_no_cumulative_target_mechanism_exists(expand_config):
    """Limits are ceilings per query, never a total number of documents to collect."""
    for spec in generate_queries(expand_config, "BMW", 0):
        assert "target_relevant_documents" not in spec
    assert "target_relevant_documents" not in expand_config["expansion"]


# --- cost shape ----------------------------------------------------------------------


def test_search_volume_is_node_centric_not_pairwise(expand_config):
    """Cost is entities x templates; no query is ever built from a pair of entities."""
    entities = ["BMW", "Toyota", "Tesla Inc.", "BYD Co., Ltd.", "CATL", "Panasonic"]
    per_entity = len(generate_queries(expand_config, "BMW", 0))
    total = sum(len(generate_queries(expand_config, name, 0)) for name in entities)

    assert total == len(entities) * per_entity
    # what an all-pairs scheme would cost, for contrast
    pairwise = len(entities) * (len(entities) - 1) * per_entity
    assert total < pairwise
    # no generated query mentions two entities at once
    for name in entities:
        for spec in generate_queries(expand_config, name, 0):
            assert sum(other in spec["query"] for other in entities) == 1
