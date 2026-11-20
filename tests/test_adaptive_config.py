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


def test_adaptive_policy_requires_candidate_and_budget_configuration():
    config = {"expansion": {
        "selection": {"policy": "adaptive_linucb"},
        "retrieval": {"max_candidates_per_entity": 20},
    }}
    with pytest.raises(ValueError, match="requires max_candidates_per_entity"):
        validate_expansion_config(config)
