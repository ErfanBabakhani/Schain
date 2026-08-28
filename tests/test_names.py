"""Deterministic name normalization and stable canonical identifiers."""

from __future__ import annotations

from schain.names import canonical_id, mention_key, normalization_key, pick_canonical_name


def test_legal_suffixes_and_case_collapse():
    assert normalization_key("BMW AG") == normalization_key("bmw ag.") == normalization_key("BMW") == "bmw"
    assert normalization_key("BYD Co., Ltd.") == "byd"
    assert normalization_key("Ganfeng Lithium Co Ltd") == normalization_key("Ganfeng Lithium")


def test_accents_are_folded():
    assert normalization_key("Société Générale") == "societe generale"


def test_meaningful_words_are_not_stripped():
    """Only legal forms are removed; 'Mining' and 'Motors' are part of the name."""
    assert normalization_key("Zijin Mining Group") == "zijin mining"
    assert normalization_key("Tesla Motors") == "tesla motors"


def test_a_name_made_only_of_suffixes_survives():
    assert normalization_key("Holdings") == "holdings"


def test_canonical_ids_are_stable_and_slugged():
    assert canonical_id("Company", "BYD Co., Ltd.") == "company:byd"
    assert canonical_id("Company", "Ganfeng Lithium") == "company:ganfeng-lithium"
    assert canonical_id("Location", "Germany") == "location:germany"
    # the same entity written two ways yields one id
    assert canonical_id("Company", "BMW AG") == canonical_id("Company", "bmw")


def test_canonical_name_is_the_longest_mention():
    assert pick_canonical_name(["CATL", "Contemporary Amperex Technology Co. Limited"]) == (
        "Contemporary Amperex Technology Co. Limited"
    )
    # ties break alphabetically, so the choice is deterministic
    assert pick_canonical_name(["BBB", "AAA"]) == "AAA"


def test_mention_key_is_type_scoped():
    assert mention_key("Company", "Germany") != mention_key("Location", "Germany")
