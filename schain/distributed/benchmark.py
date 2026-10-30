"""Reproducible before/after metrics for exported discovery snapshots.

Precision and false-negative counts require a labelled edge set. They remain ``None``
when labels are absent; this module never fabricates expected improvements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def discovery_metrics(
    snapshot: dict[str, Any], *, labelled_edges: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    facts = snapshot.get("accepted_facts") or snapshot.get("facts") or []
    documents = snapshot.get("documents") or []
    edges = {
        (str(fact.get("subject_id")), str(fact.get("object_id")))
        for fact in facts
        if fact.get("relation") == "suppliesTo" and fact.get("subject_id") and fact.get("object_id")
    }
    suppliers = {supplier for supplier, _ in edges}
    relevant_flags = [
        bool(doc.get("origin_relevant", (doc.get("metrics") or {}).get("origin_relevant")))
        for doc in documents
        if "origin_relevant" in doc or "origin_relevant" in (doc.get("metrics") or {})
    ]
    full_text = sum(
        1 for doc in documents
        if doc.get("content_source") == "full_text"
        or (doc.get("document") or {}).get("content_source") == "full_text"
    )
    true_positives = len(edges & labelled_edges) if labelled_edges is not None else None
    return {
        "unique_verified_suppliers": len(suppliers),
        "true_upstream_edges": len(edges),
        "document_relevance": (
            sum(relevant_flags) / len(relevant_flags) if relevant_flags else None
        ),
        "full_text_ratio": full_text / len(documents) if documents else None,
        "false_negatives": (
            len(labelled_edges - edges) if labelled_edges is not None else None
        ),
        "precision": (
            true_positives / len(edges) if labelled_edges is not None and edges else None
        ),
        "runtime_seconds": snapshot.get("runtime_seconds"),
        "cost": snapshot.get("cost"),
        "documents": len(documents),
    }


def compare_snapshots(
    before: dict[str, Any], after: dict[str, Any], *,
    labelled_edges: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    old = discovery_metrics(before, labelled_edges=labelled_edges)
    new = discovery_metrics(after, labelled_edges=labelled_edges)
    delta = {
        key: (new[key] - old[key])
        for key in new
        if isinstance(new[key], (int, float)) and isinstance(old.get(key), (int, float))
    }
    return {"before": old, "after": new, "delta": delta}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args(argv)
    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))
    labels = None
    if args.labels:
        payload = json.loads(args.labels.read_text(encoding="utf-8"))
        labels = {(str(item["supplier_id"]), str(item["customer_id"])) for item in payload}
    print(json.dumps(compare_snapshots(before, after, labelled_edges=labels), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
