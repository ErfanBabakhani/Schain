import pytest

from schain.config import REPO_ROOT, load_config, validate_expansion_config


def test_comparison_configs_share_candidate_and_input_caps():
    adaptive = load_config(REPO_ROOT / "config/adaptive_expand.yaml")
    legacy = load_config(REPO_ROOT / "config/legacy_comparison.yaml")
    validate_expansion_config(adaptive)
    validate_expansion_config(legacy)
    assert adaptive["expansion"]["retrieval"]["max_candidates_per_entity"] == 200
    assert legacy["expansion"]["retrieval"]["max_candidates_per_entity"] == 200
    assert adaptive["expansion"]["selection"]["total_input_tokens"] == 200000
    assert legacy["expansion"]["selection"]["total_input_tokens"] == 200000
    assert adaptive["expansion"]["selection"]["candidate_pool_policy"] == "title_snippet"
    assert legacy["expansion"]["selection"]["candidate_pool_policy"] == "title_snippet"
    assert adaptive["storage"]["state_db"] != legacy["storage"]["state_db"]
    assert adaptive["expansion"]["search"] == legacy["expansion"]["search"]
    assert adaptive["expansion"]["sources"] == legacy["expansion"]["sources"]
    assert adaptive["expansion"]["relevance"] == legacy["expansion"]["relevance"]
    assert adaptive["seeds"] == legacy["seeds"]
    assert adaptive["validation"] == legacy["validation"]
    assert adaptive["ontology_file"] == legacy["ontology_file"]
    for key in (
        "max_candidates_per_entity", "max_candidates_per_domain",
        "max_full_text_fetches_per_entity", "max_full_text_fetches_per_domain",
        "max_title_lookups_per_entity",
    ):
        assert adaptive["expansion"]["retrieval"][key] == \
            legacy["expansion"]["retrieval"][key]
    assert adaptive["expansion"]["selection"]["input_token_metric"] == \
        legacy["expansion"]["selection"]["input_token_metric"] == "tokenizer_input_tokens"
    assert adaptive["llm"]["model"] == legacy["llm"]["model"]
    priors = adaptive["expansion"]["selection"]["initial_action_prior"]
    assert priors["snippet"] == .60
    assert priors["full_text"] + priors["both"] == .10
    assert priors["skip"] == .30
    assert adaptive["expansion"]["selection"]["snippet_batch_size"] == 10
    # Adaptive selection is token-stopped; the legacy comparison intentionally retains
    # expansion.yaml's per-run/per-tier document stops for a legacy-policy reference.
    assert adaptive["expansion"]["max_documents_per_run"] is None
    assert legacy["expansion"]["max_documents_per_run"] == 320


def test_adaptive_policy_accepts_configured_candidate_limit_and_requires_budget():
    config = {"expansion": {
        "selection": {
            "policy": "adaptive_linucb",
            "total_input_tokens": 1000,
            "target_tokens_per_candidate": 100,
        },
        "retrieval": {"max_candidates_per_entity": 175},
    }}
    validate_expansion_config(config)
    assert config["expansion"]["retrieval"]["max_candidates_per_entity"] == 175

    config["expansion"]["retrieval"]["max_candidates_per_entity"] = 0
    with pytest.raises(ValueError, match="positive max_candidates_per_entity"):
        validate_expansion_config(config)

    config["expansion"]["retrieval"]["max_candidates_per_entity"] = 200
    config["expansion"]["selection"].pop("total_input_tokens")
    with pytest.raises(ValueError, match="requires expansion.selection.total_input_tokens"):
        validate_expansion_config(config)


def test_server_preset_separates_api_model_alias_from_tokenizer_model():
    config = load_config(REPO_ROOT / "config/server.yaml")

    assert config["llm"]["model"] == "schain-qwen"
    assert config["llm"]["tokenizer_model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert config["expansion"]["selection"]["official_domains"]
