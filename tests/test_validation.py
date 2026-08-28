"""External supplier-customer reference validation."""

from __future__ import annotations

import pandas as pd
import pytest

from schain.kg.facts import FACT_COLUMNS
from schain.validation.sectivia import (
    compare_with_facts,
    find_reference_file,
    load_reference,
)


@pytest.fixture
def reference_csv(tmp_path):
    path = tmp_path / "relations.csv"
    path.write_text(
        "supplier,customer\n"
        "CATL,BMW AG\n"
        "Ganfeng Lithium Co Ltd,CATL\n"
        "Bosch,Toyota\n",
        encoding="utf-8",
    )
    return path


def facts(*edges):
    rows = []
    for index, (subject, obj) in enumerate(edges):
        row = {column: None for column in FACT_COLUMNS}
        row.update(
            fact_id=f"f{index}", doc_id="d1", subject_id=subject, subject_type="Company",
            relation="suppliesTo", object_id=obj, object_type="Company",
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=FACT_COLUMNS)


def test_reference_is_normalized_to_our_canonical_ids(reference_csv):
    frame = load_reference(reference_csv)
    assert list(frame.columns) == ["supplier", "customer", "supplier_id", "customer_id"]
    # the same normalization rule as the graph, so the two are comparable
    assert set(frame.supplier_id) == {"company:catl", "company:ganfeng-lithium", "company:bosch"}
    assert "company:bmw" in set(frame.customer_id)


def test_alternative_column_names_are_accepted(tmp_path):
    path = tmp_path / "edges.csv"
    path.write_text("source,target\nCATL,BMW\n", encoding="utf-8")
    frame = load_reference(path)
    assert frame.iloc[0].supplier_id == "company:catl"


def test_unrecognized_columns_are_reported(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="could not find a supplier column"):
        load_reference(path)


def test_duplicate_reference_edges_are_collapsed(tmp_path):
    path = tmp_path / "dupes.csv"
    path.write_text("supplier,customer\nCATL,BMW\nCATL,BMW AG\n", encoding="utf-8")
    assert len(load_reference(path)) == 1


def test_comparison_reports_coverage(reference_csv):
    reference = load_reference(reference_csv)
    extracted = facts(
        ("company:catl", "company:bmw"),                 # in the reference
        ("company:ganfeng-lithium", "company:catl"),     # in the reference
        ("company:panasonic", "company:tesla"),          # outside its scope
    )
    report = compare_with_facts(reference, extracted)
    assert report["reference_edges"] == 3
    assert report["matched_edges"] == 2
    assert report["coverage"] == pytest.approx(2 / 3, abs=1e-4)
    assert report["extracted_not_in_reference"] == 1
    assert report["examples_missing"] == [("company:bosch", "company:toyota")]


def test_comparison_with_no_facts_is_zero_not_an_error(reference_csv):
    report = compare_with_facts(load_reference(reference_csv), pd.DataFrame())
    assert report["extracted_edges"] == 0 and report["coverage"] == 0.0


def test_only_supply_edges_are_compared(reference_csv):
    extracted = facts(("company:catl", "company:bmw"))
    extracted.loc[0, "relation"] = "owns"
    report = compare_with_facts(load_reference(reference_csv), extracted)
    assert report["matched_edges"] == 0


def test_missing_directory_yields_no_reference_file(tmp_path):
    assert find_reference_file(tmp_path / "absent") is None
