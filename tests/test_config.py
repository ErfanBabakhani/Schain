"""Configuration loading, the include mechanism, and the search-limit hierarchy."""

from __future__ import annotations

import pytest

from schain.config import (
    REPO_ROOT,
    load_config,
    resolve_tier_queries,
    seed_entities,
    storage_path,
)


def test_include_brings_in_shared_settings():
    config = load_config(REPO_ROOT / "config/ingest.yaml")
    # llm/neo4j/storage live in config/base.yaml, not in ingest.yaml
    assert config["llm"]["model"] == "Qwen/Qwen3.8-27B"
    assert config["neo4j"]["database"] == "neo4j"
    assert config["storage"]["state_db"].endswith("state.db")
    # and the including file's own keys survive
    assert config["mode"] == "ingest"
    assert config["ingest"]["chunk_size"] == 2000


def test_own_keys_override_included_ones(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("llm:\n  model: base-model\n  temperature: 0.7\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text(f"include:\n  - {base}\nllm:\n  model: child-model\n", encoding="utf-8")

    config = load_config(child)
    assert config["llm"]["model"] == "child-model"   # overridden
    assert config["llm"]["temperature"] == 0.7       # merged, not replaced wholesale


def test_circular_include_is_reported(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(f"include:\n  - {b}\n", encoding="utf-8")
    b.write_text(f"include:\n  - {a}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="circular config include"):
        load_config(a)


def test_query_limits_follow_closest_value_wins():
    config = load_config(REPO_ROOT / "config/expansion.yaml")

    tier0 = resolve_tier_queries(config, 0)
    by_query = {spec["query"]: spec for spec in tier0}
    # tier default 30 applies...
    assert by_query['"{entity}" suppliers']["max_results"] == 30
    # ...except where a single query overrides it
    assert by_query['"{entity}" semiconductor suppliers']["max_results"] == 15

    # tier 2 declares no defaults, so the global default applies
    tier2 = {spec["query"]: spec for spec in resolve_tier_queries(config, 2)}
    assert tier2['"{entity}" material suppliers']["max_results"] == 20
    assert tier2['"{entity}" mines']["max_results"] == 40  # per-query override


def test_fields_resolve_independently(tmp_path):
    """A tier that overrides only max_results still inherits the global providers."""
    config = {
        "expansion": {
            "search": {
                "defaults": {"max_results": 20, "providers": ["wikipedia", "gdelt"]},
                "tiers": {1: {"defaults": {"max_results": 5},
                              "queries": [{"query": '"{entity}" suppliers'}]}},
            }
        }
    }
    spec = resolve_tier_queries(config, 1)[0]
    assert spec["max_results"] == 5
    assert spec["providers"] == ["wikipedia", "gdelt"]


def test_tier_without_configuration_yields_no_queries():
    config = load_config(REPO_ROOT / "config/expansion.yaml")
    assert resolve_tier_queries(config, 9) == []


def test_seeds_are_flattened_and_deduplicated():
    config = load_config(REPO_ROOT / "config/expansion.yaml")
    seeds = seed_entities(config)
    assert seeds == ["Toyota", "Tesla Inc.", "BMW", "BYD Co., Ltd."]
    assert len(seeds) == len(set(seeds))


def test_storage_path_creates_parent(ingest_config):
    path = storage_path(ingest_config, "extracted_facts")
    assert path.parent.exists()


def test_missing_storage_key_is_reported(ingest_config):
    with pytest.raises(KeyError, match="storage.nope"):
        storage_path(ingest_config, "nope")
