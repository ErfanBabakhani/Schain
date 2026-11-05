"""Authoritative accepted-document checkpoints for the optional SQLite graph workflow.

The research Parquet layers can be repaired from these rows without another LLM call.
The graph exporter retains sole ownership of semantic edge aggregation and tiers.
"""
import json

import pandas as pd

from .checkpoint import atomic_write_parquet
from .config import storage_path
from .kg.facts import FACT_COLUMNS

SCHEMA = """
CREATE TABLE accepted_document_checkpoints(
 doc_id TEXT PRIMARY KEY REFERENCES documents(doc_id),
 document TEXT NOT NULL, resolution TEXT NOT NULL
);
CREATE TABLE accepted_facts(
 fact_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id),
 payload TEXT NOT NULL
);
CREATE INDEX accepted_facts_doc ON accepted_facts(doc_id);
"""


def commit_document(connection, document, facts, resolution):
    from .graph_export import json_safe
    from .state import RESOLVED, _now

    def encode(value):
        return json.dumps(json_safe(value), allow_nan=False, ensure_ascii=False)

    with connection:
        connection.execute(
            "INSERT INTO accepted_document_checkpoints VALUES(?,?,?) "
            "ON CONFLICT(doc_id) DO UPDATE SET document=excluded.document,resolution=excluded.resolution",
            (document["doc_id"], encode(document), encode(resolution)),
        )
        connection.executemany(
            "INSERT INTO accepted_facts VALUES(?,?,?) ON CONFLICT(fact_id) "
            "DO UPDATE SET payload=excluded.payload",
            [(row["fact_id"], document["doc_id"], encode(row)) for row in facts.to_dict(orient="records")],
        )
        connection.execute("UPDATE documents SET status=?,error=NULL,updated_at=? WHERE doc_id=?",
                           (RESOLVED, _now(), document["doc_id"]))


def load_facts(connection):
    rows = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM accepted_facts ORDER BY fact_id")]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=FACT_COLUMNS)


def load_documents(connection):
    return {row[0]: json.loads(row[1]) for row in connection.execute(
        "SELECT doc_id,document FROM accepted_document_checkpoints"
    )}


def restore_artifacts(config, connection):
    """Repair derived accepted facts and resolution before entering the existing loop."""
    atomic_write_parquet(load_facts(connection), storage_path(config, "extracted_facts"))
    rows = []
    for checkpoint in connection.execute("SELECT doc_id,resolution FROM accepted_document_checkpoints"):
        rows.extend({"doc_id": checkpoint[0], "node_id": key, **entry}
                    for key, entry in json.loads(checkpoint[1]).items())
    if rows:
        atomic_write_parquet(pd.DataFrame(rows), storage_path(config, "entity_resolution"))
