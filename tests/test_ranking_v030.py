"""Origin-relative retrieval priorities without network or model calls."""

from copy import deepcopy

import pytest

from schain.expansion.queries import generate_queries
from schain.kg.relation_evidence import (
    DIRECTION_AMBIGUOUS, DOWNSTREAM_EXPLICIT, UPSTREAM_EXPLICIT,
    load_relation_evidence,
)
from schain.sources.base import SearchResult
from schain.sources.origin import classify_origin_relevance
from schain.sources.ranking import (
    has_explicit_relation_language, is_usable_discovery_result,
    mentions_entity_name, rank_discovery_candidate, sort_discovery_results,
)


def candidate(title, *, provider="general_web", text=None, snippet="", **kwargs):
    return SearchResult(provider=provider, query='"BorgWarner" supplier', title=title,
                        url="https://example.test/" + str(abs(hash(title))),
                        text=text, snippet=snippet, **kwargs)


def test_direction_dominates_content_and_reputation_and_keeps_downstream():
    upstream = candidate("Acme supplies BorgWarner", provider="google_news")
    ambiguous = candidate("BorgWarner signs a supply agreement with Acme")
    downstream = candidate("BorgWarner supplies Stellantis",
                           text="BorgWarner supplies Stellantis. " * 20)
    upstream._source_effective_score = 0
    downstream._source_effective_score = 100
    assert sort_discovery_results([downstream, ambiguous, upstream], "BorgWarner") == [
        upstream, ambiguous, downstream,
    ]
    assert upstream._discovery_direction == UPSTREAM_EXPLICIT
    assert ambiguous._discovery_direction == DIRECTION_AMBIGUOUS
    assert downstream._discovery_direction == DOWNSTREAM_EXPLICIT


@pytest.mark.parametrize("headline", [
    "Acme supplies BorgWarner",
    "Acme is a supplier to BorgWarner",
    "BorgWarner sources automotive components from Acme",
    "BorgWarner purchases battery cells from Acme",
    "BorgWarner selects Acme as its component supplier",
    "Acme delivers components to BorgWarner",
])
def test_upstream_language_receives_top_priority(headline):
    result = candidate(headline)
    rank_discovery_candidate(result, "BorgWarner")
    assert result._discovery_direction == UPSTREAM_EXPLICIT


def test_mixed_document_keeps_upstream_priority():
    result = candidate("BorgWarner supplies Stellantis",
                       text="BorgWarner supplies Stellantis. Acme supplies BorgWarner.")
    rank_discovery_candidate(result, "BorgWarner")
    assert result._discovery_direction == UPSTREAM_EXPLICIT


def test_same_candidate_rank_ignores_source_reputation():
    result = candidate("Acme supplies BorgWarner")
    assert rank_discovery_candidate(result, "BorgWarner", source_effective_score=0) == (
        rank_discovery_candidate(result, "BorgWarner", source_effective_score=100)
    )


def test_full_text_footer_does_not_demote_evidence():
    clean = candidate("Acme supplies BorgWarner", text="Acme supplies BorgWarner.")
    footer = candidate("Acme supplies BorgWarner", text=(
        "Acme supplies BorgWarner. Privacy policy. Cookie policy. Terms of use."
    ))
    assert rank_discovery_candidate(clean, "BorgWarner") == rank_discovery_candidate(footer, "BorgWarner")


def test_annual_and_sustainability_reports_are_not_negative():
    report = candidate("BorgWarner sustainability report", text="Acme supplies BorgWarner.")
    ordinary = candidate("BorgWarner annual overview", text="Acme supplies BorgWarner.")
    assert rank_discovery_candidate(report, "BorgWarner") == rank_discovery_candidate(ordinary, "BorgWarner")


def test_portal_stays_below_ambiguous_supply_article():
    portal = candidate("BorgWarner supplier portal login")
    agreement = candidate("BorgWarner signs a supply agreement with Acme")
    assert rank_discovery_candidate(agreement, "BorgWarner") > rank_discovery_candidate(portal, "BorgWarner")


def test_explicit_snippet_remains_usable_when_full_text_missing():
    result = candidate("BorgWarner purchases precision automotive components from Acme",
                       provider="google_news")
    assert is_usable_discovery_result(result, "BorgWarner")


def test_configured_alias_works_in_ranking_and_usability():
    result = candidate("Acme supplies precision components to CATL for new manufacturing lines")
    long_name = "Contemporary Amperex Technology"
    assert is_usable_discovery_result(result, long_name, aliases=["CATL"])
    rank_discovery_candidate(result, long_name, aliases=["CATL"])
    assert result._discovery_direction == UPSTREAM_EXPLICIT


def test_name_substrings_are_not_entity_mentions():
    assert not mentions_entity_name("BYDauto supplier network", "BYD")
    assert not mentions_entity_name("BMWi7 supplier network", "BMW")
    assert mentions_entity_name("The BMW AG supplier network", "BMW")


def test_upstream_body_survives_other_oem_headline():
    result = candidate("Tesla and Rivian compare new electric models",
                       text="Acme supplies precision components to BorgWarner.")
    assert classify_origin_relevance(result, "BorgWarner") == (True, "ok")


def test_oem_product_comparison_without_upstream_evidence_stays_filtered():
    result = candidate("Tesla and Rivian compare new electric models",
                       text="BorgWarner is mentioned in a list of automotive brands.")
    assert classify_origin_relevance(result, "BorgWarner") == (False, "generic_result")


def test_configured_relation_cue_changes_discovery_eligibility():
    policy = deepcopy(load_relation_evidence())
    text = "BorgWarner commissions precision cells from Acme."
    base = has_explicit_relation_language(text, policy=policy)
    policy["suppliesTo"]["positive"] = ["commissions .* from"]
    assert has_explicit_relation_language(text, policy=policy)
    assert base is False


def test_nondirectional_names_wins_expand_do_not_count_as_relation_evidence():
    for text in ("BMW names a CEO", "BorgWarner wins an award", "BMW and Toyota expand offices"):
        assert not has_explicit_relation_language(text)


def test_duplicate_query_specs_do_not_repeat_searches():
    config = {"expansion": {"search": {"defaults": {"providers": ["gdelt"], "max_results": 4},
        "tiers": {0: {"queries": [
            {"query": 'supplies "{entity}"'}, {"query": 'supplies "{entity}"'},
            {"query": 'supplies "{entity}"', "max_results": 2},
        ]}}}}}
    specs = generate_queries(config, "BMW", 0)
    assert len(specs) == 2
    assert [spec["max_results"] for spec in specs] == [4, 2]


def test_two_good_documents_do_not_get_padded_with_navigation_results():
    good = [candidate("Acme supplies precision automotive components to BorgWarner"),
            candidate("BorgWarner purchases automotive battery cells from SecondCorp")]
    noise = [candidate("BorgWarner supplier portal login and vendor registration " + str(index),
                       snippet="Register as a supplier to access the BorgWarner sourcing portal.")
             for index in range(6)]
    ranked = sort_discovery_results(good + noise, "BorgWarner")
    usable = [r for r in ranked if is_usable_discovery_result(r, "BorgWarner")]
    assert usable == good


def test_generic_full_text_does_not_become_supply_evidence_from_footer():
    result = candidate("BorgWarner releases corporate quarterly update", text=(
        "BorgWarner describes electric vehicles and semiconductor market conditions. "
        "The company opened a new office and discussed team development this year.\n"
        "Supplier portal login. Vendor registration."
    ))
    assert not is_usable_discovery_result(result, "BorgWarner")


def test_bilateral_supply_agreement_remains_a_meaningful_candidate():
    result = candidate("BorgWarner signs a supply agreement with Acme", text=(
        "BorgWarner signs a supply agreement with Acme. "
        "The parties announced a multiyear automotive program scheduled for next year."
    ))
    assert is_usable_discovery_result(result, "BorgWarner")


def test_navigation_title_does_not_hide_explicit_upstream_body():
    result = candidate("BorgWarner supplier portal", text=(
        "Acme supplies precision automotive components to BorgWarner. "
        "The partners announced a contract extension for the next vehicle program."
    ))
    assert is_usable_discovery_result(result, "BorgWarner")


def test_direct_article_gets_fetch_slot_ahead_of_equivalent_wrapper_headlines():
    wrappers = [SearchResult(
        provider="google_news", query="BorgWarner supplier",
        title=f"Vendor{index} supplies precision components to BorgWarner",
        url=f"https://news.google.com/rss/articles/{index}",
    ) for index in range(24)]
    direct = SearchResult(
        provider="general_web", query="BorgWarner supplier",
        title="OtherVendor supplies precision components to BorgWarner",
        url="https://publisher.test/articles/supply-contract",
    )
    ranked = sort_discovery_results(wrappers + [direct], "BorgWarner")
    assert ranked[0] is direct
    assert direct in ranked[:24]
    # The signal concerns URL shape, never a provider quota.
    direct.provider = "google_news"
    assert sort_discovery_results(wrappers + [direct], "BorgWarner")[0] is direct
