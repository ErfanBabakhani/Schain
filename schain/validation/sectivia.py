"""Validation against an external supplier-customer reference dataset.

The architecture brief names the Sectivia open supply-chain dataset for this role. A
public download location for it could not be located (see docs/EXTERNAL_DATASETS.md), so
nothing is downloaded automatically and no URL is guessed here. Instead this module reads
any supplier-customer table placed under ``data/external/sectivia/`` and compares it with
the extracted facts.

This is a sanity check on relationships, not a graph-analytics module: it reports how much
of the reference overlaps with what we extracted, and nothing more.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..names import canonical_id

logger = logging.getLogger(__name__)

DEFAULT_DIRECTORY = Path("data/external/sectivia")
# Column names accepted for the two endpoints, in order of preference.
SUPPLIER_COLUMNS = ("supplier", "supplier_name", "source", "from", "seller")
CUSTOMER_COLUMNS = ("customer", "customer_name", "target", "to", "buyer")


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...], role: str) -> str:
    lowered = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(
        f"could not find a {role} column in {list(frame.columns)}; "
        f"expected one of {candidates}"
    )


def load_reference(
    path: Path,
    supplier_column: Optional[str] = None,
    customer_column: Optional[str] = None,
) -> pd.DataFrame:
    """Load a supplier-customer edge list into our schema.

    Returns a frame with ``supplier``, ``customer``, ``supplier_id`` and ``customer_id``,
    where the ids use the same canonical-id rule as the graph, so the two are comparable.
    """
    path = Path(path)
    if path.suffix.lower() in (".json", ".jsonl"):
        frame = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        frame = pd.read_csv(path)

    supplier = supplier_column or _pick_column(frame, SUPPLIER_COLUMNS, "supplier")
    customer = customer_column or _pick_column(frame, CUSTOMER_COLUMNS, "customer")

    normalized = pd.DataFrame(
        {
            "supplier": frame[supplier].astype(str).str.strip(),
            "customer": frame[customer].astype(str).str.strip(),
        }
    )
    normalized["supplier_id"] = normalized.supplier.apply(lambda n: canonical_id("Company", n))
    normalized["customer_id"] = normalized.customer.apply(lambda n: canonical_id("Company", n))
    normalized = normalized.drop_duplicates(subset=["supplier_id", "customer_id"])
    logger.info("reference dataset: %d unique supplier-customer edge(s)", len(normalized))
    return normalized.reset_index(drop=True)


def compare_with_facts(
    reference: pd.DataFrame, facts: pd.DataFrame, relation: str = "suppliesTo"
) -> dict[str, Any]:
    """Compare reference edges with extracted edges.

    ``coverage`` is the share of reference edges we also extracted. Edges we extracted
    that are absent from the reference are counted but not called errors: the reference
    covers only its own 200-odd companies, so most of them are simply out of its scope.
    """
    reference_edges = set(zip(reference.supplier_id, reference.customer_id))
    if facts.empty:
        extracted_edges: set[tuple[str, str]] = set()
    else:
        supplies = facts[facts.relation == relation]
        extracted_edges = set(zip(supplies.subject_id, supplies.object_id))

    matched = reference_edges & extracted_edges
    reference_companies = set(reference.supplier_id) | set(reference.customer_id)
    extracted_companies = {company for edge in extracted_edges for company in edge}

    return {
        "reference_edges": len(reference_edges),
        "extracted_edges": len(extracted_edges),
        "matched_edges": len(matched),
        "coverage": round(len(matched) / len(reference_edges), 4) if reference_edges else 0.0,
        "reference_companies": len(reference_companies),
        "companies_in_common": len(reference_companies & extracted_companies),
        "extracted_not_in_reference": len(extracted_edges - reference_edges),
        "examples_matched": sorted(matched)[:10],
        "examples_missing": sorted(reference_edges - extracted_edges)[:10],
    }


def find_reference_file(directory: Path = DEFAULT_DIRECTORY) -> Optional[Path]:
    """First CSV/JSON file in the external dataset directory, if one was placed there."""
    directory = Path(directory)
    if not directory.exists():
        return None
    for pattern in ("*.csv", "*.json", "*.jsonl"):
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None
