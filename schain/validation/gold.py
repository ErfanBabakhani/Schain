"""Exact, directed relation evaluation on a fully annotated held-out document set.

Unlike external-reference coverage, this is closed-world evaluation: all in-scope
triples in each listed document must be labeled, including zero-relation documents.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def _scores(predicted: set, gold: set) -> dict[str, Any]:
    tp = len(predicted & gold)
    fp, fn = len(predicted - gold), len(gold - predicted)
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp / len(predicted) if predicted else None,
            "recall": tp / len(gold) if gold else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None}


def evaluate_gold(gold: dict, facts: Any) -> dict:
    documents = gold.get("documents") if isinstance(gold, dict) else None
    if not isinstance(documents, list) or not documents:
        raise ValueError("Gold JSON must contain a non-empty documents array")
    gold_edges, scope = set(), set()
    for doc in documents:
        if not isinstance(doc, dict) or not isinstance(doc.get("doc_id"), str) or not doc["doc_id"] or doc["doc_id"] in scope:
            raise ValueError("Gold document IDs must be unique non-empty strings")
        if not isinstance(doc.get("triples"), list):
            raise ValueError("Each gold document needs a triples list (empty for negatives)")
        scope.add(doc["doc_id"])
        for edge in doc["triples"]:
            if not isinstance(edge, dict) or any(not isinstance(edge.get(k), str) or not edge[k] for k in ("subject_id", "relation", "object_id")):
                raise ValueError("Each gold triple needs subject_id, relation, object_id strings")
            gold_edges.add((doc["doc_id"], edge["subject_id"], edge["relation"], edge["object_id"]))
    rows = facts.to_dict(orient="records") if hasattr(facts, "to_dict") else facts
    predicted, excluded_docs = set(), set()
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k] for k in ("doc_id", "subject_id", "relation", "object_id")):
            raise ValueError("Predictions need non-empty doc_id, subject_id, relation, object_id strings")
        if row["doc_id"] not in scope:
            excluded_docs.add(row["doc_id"])
            continue
        predicted.add(tuple(row[k] for k in ("doc_id", "subject_id", "relation", "object_id")))
    relations = sorted({e[2] for e in predicted | gold_edges})
    return {"evaluation": "exact_document_level_directed_triples", "gold_documents": len(scope),
            "excluded_prediction_documents": len(excluded_docs),
            "gold_triples": len(gold_edges), "predicted_triples": len(predicted),
            "micro": _scores(predicted, gold_edges),
            "per_relation": {r: _scores({e for e in predicted if e[2] == r}, {e for e in gold_edges if e[2] == r}) for r in relations},
            "false_positives": [list(e) for e in sorted(predicted - gold_edges)],
            "false_negatives": [list(e) for e in sorted(gold_edges - predicted)],
            "notes": ["Canonical IDs and edge direction must match exactly; duplicate evidence is counted once.",
                      "All allowed relations in the listed documents must be fully annotated.",
                      "Null metrics mean an undefined denominator, not measured perfect performance.",
                      "This evaluates accepted relation extraction, not isolated-entity NER or source truthfulness."]}


def evaluate_file(gold_path: Path, facts_path: Path) -> dict:
    import pandas as pd
    if not facts_path.exists():
        raise FileNotFoundError(f"Prediction facts not found: {facts_path}")
    return evaluate_gold(json.loads(gold_path.read_text(encoding="utf-8")), pd.read_parquet(facts_path))
