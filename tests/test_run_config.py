"""Offline experiment identity, export provenance and configuration authority."""

import copy
import json

import pytest

from schain import state
from schain.config import (REPO_ROOT, apply_runtime_overrides, load_config,
                           log_effective_expansion_config, resolve_tier_queries,
                           seed_entities)
from schain.documents import make_document
from schain.run_config import ensure_run_snapshot, effective_export_config, snapshot_path


def test_snapshot_drives_export_semantics_and_preserves_output_paths(expand_config):
    original = ensure_run_snapshot(expand_config)
    changed = copy.deepcopy(expand_config)
    changed["seeds"] = {"different": ["BMW"]}
    changed["expansion"]["max_tier"] = 2
    changed["storage"]["graph_export"] = "/tmp/another-output.json"
    effective, snapshot = effective_export_config(changed)
    assert seed_entities(effective) == ["Toyota", "Tesla Inc.", "BMW", "BYD Co., Ltd."]
    assert effective["expansion"]["max_tier"] == 5
    assert effective["storage"]["graph_export"] == "/tmp/another-output.json"
    assert snapshot["started_at"] == original["started_at"]


@pytest.mark.parametrize("changed", ["model", "seeds", "max_tier", "validation"])
def test_resume_rejects_silent_experiment_change(expand_config, changed):
    ensure_run_snapshot(expand_config)
    if changed == "model":
        expand_config["llm"]["model"] = "different-model"
    elif changed == "seeds":
        expand_config["seeds"] = ["BMW"]
    elif changed == "max_tier":
        expand_config["expansion"]["max_tier"] = 2
    else:
        expand_config["validation"]["evidence"]["enabled"] = False
    with pytest.raises(ValueError, match="--restart"):
        ensure_run_snapshot(expand_config)


def test_operational_cap_changes_are_recorded_on_resume(expand_config):
    old = ensure_run_snapshot(expand_config)
    expand_config["expansion"]["max_documents"] = 180
    expand_config["expansion"]["max_documents_per_run"] = 180
    new = ensure_run_snapshot(expand_config)
    assert new["identity_sha256"] == old["identity_sha256"]
    assert new["global_document_hard_cap"] == 180
    assert len(new["config_revisions"]) == 1
    assert ensure_run_snapshot(expand_config) == new


def test_snapshot_does_not_store_credentials(expand_config):
    expand_config["llm"].update(api_key="secret-value", base_url="https://user:pwd@example.org/v1?token=abc")
    expand_config["expansion"]["sources"]["general_web"]["api_key"] = "hidden-key"
    ensure_run_snapshot(expand_config)
    text = snapshot_path(expand_config).read_text()
    assert all(secret not in text for secret in ("secret-value", "hidden-key", "user:pwd", "token=abc"))
    assert "https://example.org/v1" in text


def test_legacy_snapshot_does_not_invent_historical_provenance(expand_config, connection):
    state.register_document(connection, make_document("CATL supplies BMW."))
    snapshot = ensure_run_snapshot(expand_config)
    assert snapshot["legacy_artifacts_unverified"] is True


def test_restart_removes_snapshot(expand_config):
    from schain.pipeline import clear_generated_data
    ensure_run_snapshot(expand_config)
    clear_generated_data(expand_config)
    assert not snapshot_path(expand_config).exists()


def test_corrupt_snapshot_is_never_silently_replaced(expand_config):
    path = snapshot_path(expand_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("broken")
    with pytest.raises(json.JSONDecodeError):
        effective_export_config(expand_config)


def test_runtime_overlay_rejects_seed_and_depth_overwrite(tmp_path):
    for content in ("seeds: [BMW]", "expansion: {max_tier: 2}"):
        path = tmp_path / "runtime.yaml"
        path.write_text(f"runtime_overlay: true\n{content}\n")
        with pytest.raises(ValueError, match="explicit CLI"):
            load_config(path)


def test_runtime_helper_preserves_depth_unless_explicitly_unprotected(expand_config):
    overrides = {"expansion": {"max_tier": 2, "max_documents": 50}}
    protected = apply_runtime_overrides(expand_config, overrides)
    assert protected["expansion"]["max_tier"] == 5
    assert protected["expansion"]["max_documents"] == 50
    assert apply_runtime_overrides(expand_config, overrides, protect_max_tier=False)["expansion"]["max_tier"] == 2


def test_diamond_include_is_not_a_cycle(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("llm: {model: shared}\n")
    left, right = tmp_path / "left.yaml", tmp_path / "right.yaml"
    for branch in (left, right):
        branch.write_text(f"include: ['{base}']\n")
    child = tmp_path / "child.yaml"
    child.write_text(f"include: ['{left}', '{right}']\n")
    assert load_config(child)["llm"]["model"] == "shared"


def test_partial_provider_override_keeps_parent_limits():
    config = {"expansion": {"search": {
        "defaults": {"max_results": 20, "max_results_by_provider": {"gdelt": 12, "wikipedia": 5}},
        "tiers": {0: {"defaults": {"max_results_by_provider": {"wikipedia": 2}},
                       "queries": [{"query": "test"}]}},
    }}}
    assert resolve_tier_queries(config, 0)[0]["max_results_by_provider"] == {"gdelt": 12, "wikipedia": 2}


def test_colab_effective_experiment_and_safe_logging(caplog):
    config = load_config(REPO_ROOT / "config/colab_professor_expand.yaml")
    assert len(seed_entities(config)) == 4
    assert config["expansion"]["max_tier"] == 5
    assert config["runtime"]["max_minutes"] == 120
    assert config["llm"]["model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert resolve_tier_queries(config, 5) == []
    config["llm"]["api_key"] = "should-never-be-logged"
    with caplog.at_level("INFO"):
        log_effective_expansion_config(config)
    assert all(word in caplog.text for word in ("seed_count=4", "max_tier=5", "model=Qwen", "runtime_minutes=120"))
    assert "should-never-be-logged" not in caplog.text


def test_repeatable_seed_cli_preserves_commas():
    from schain.__main__ import build_parser
    args = build_parser().parse_args(["expand", "--seed", "BMW", "--seed", "BYD Co., Ltd."])
    assert args.seed == ["BMW", "BYD Co., Ltd."]
