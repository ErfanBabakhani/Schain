"""Document normalization, identifiers and input discovery."""

from __future__ import annotations

import pytest

from schain.documents import (
    content_hash,
    iter_input_files,
    load_document,
    make_document,
    normalize_text,
    normalize_url,
)


def test_text_hash_ignores_formatting_noise():
    """CRLF, a BOM and trailing spaces must not produce two different documents."""
    a = normalize_text("CATL supplies\r\nBMW.  \n\n")
    b = normalize_text("﻿CATL supplies\nBMW.")
    assert a == b == "CATL supplies\nBMW."
    assert content_hash(a) == content_hash(b)


@pytest.mark.parametrize(
    "left,right",
    [
        ("https://en.wikipedia.org/wiki/CATL?a=1", "https://EN.Wikipedia.org/wiki/CATL/?a=1#history"),
        ("http://example.com/x", "http://example.com/x/?utm_source=news&utm_medium=rss"),
        ("https://site.org/", "https://site.org"),
        ("https://s.com/a?b=1&fbclid=zz", "https://s.com/a?b=1"),
    ],
)
def test_url_normalization_makes_equivalent_urls_equal(left, right):
    assert normalize_url(left) == normalize_url(right)


def test_url_normalization_keeps_meaningful_queries():
    assert normalize_url("https://s.com/search?q=bmw") == "https://s.com/search?q=bmw"
    assert normalize_url(None) is None


def test_different_urls_stay_different():
    assert normalize_url("https://s.com/a") != normalize_url("https://s.com/b")


def test_make_document_fields(document_file):
    record = make_document("CATL supplies BMW.", source_url="https://x.test/a/", title="T")
    assert record["doc_id"] == record["content_hash"][:16]
    assert record["source_url"] == "https://x.test/a"
    assert record["title"] == "T"
    assert record["char_count"] == len("CATL supplies BMW.")


def test_empty_document_is_rejected():
    with pytest.raises(ValueError, match="empty after normalization"):
        make_document("   \n\n  ")


def test_load_document_reads_a_text_file(document_file):
    record = load_document(document_file)
    assert "CATL supplies battery cells" in record["text"]
    assert record["source_path"] == str(document_file)
    assert record["title"] == document_file.stem


def test_load_document_reads_a_pdf():
    """The architecture brief in docs/ is a real PDF, read through neo4j-graphrag."""
    from schain.config import REPO_ROOT

    record = load_document(REPO_ROOT / "docs/SupplyChain_KG_V1_Architecture_Brief_Final.pdf")
    assert record["char_count"] > 1000
    assert "Supply-Chain Knowledge Graph" in record["text"]


def test_unsupported_extension_is_reported(tmp_path):
    path = tmp_path / "data.xlsx"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported input type"):
        load_document(path)


def test_iter_input_files_handles_file_and_directory(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("c", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.txt").write_text("c", encoding="utf-8")

    found = iter_input_files(tmp_path, ["*.txt", "*.md"])
    assert [p.name for p in found] == ["a.txt", "b.md", "c.txt"]
    assert iter_input_files(tmp_path / "a.txt", ["*.txt"]) == [tmp_path / "a.txt"]


def test_missing_input_path_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        iter_input_files(tmp_path / "nope", ["*.txt"])
