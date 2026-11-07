"""Project database migrations and the durable semantic graph read model."""
import json
import sqlite3
import uuid
from contextlib import contextmanager

from schain import state
from schain.graph_export import json_safe
from schain.viewer import validate_graph
from schain.sqlite_graph import SCHEMA as ACCEPTED_GRAPH_SCHEMA

MIGRATION_1 = """
CREATE TABLE server_schema(version INTEGER PRIMARY KEY);
INSERT INTO server_schema VALUES(1);
CREATE TABLE project(id TEXT PRIMARY KEY, settings TEXT NOT NULL, config TEXT NOT NULL);
CREATE TABLE runs(
 id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('expand','ingest')),
 status TEXT NOT NULL CHECK(status IN
 ('queued','running','pausing','paused','completed','failed')),
 input_path TEXT, created_at TEXT NOT NULL, finished_at TEXT, summary TEXT, error TEXT
);
CREATE UNIQUE INDEX one_active_run ON runs((1))
 WHERE status IN ('queued','running','pausing');
CREATE TABLE graph_nodes(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE graph_documents(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE graph_edges(
 id TEXT PRIMARY KEY, source TEXT NOT NULL REFERENCES graph_nodes(id),
 target TEXT NOT NULL REFERENCES graph_nodes(id), relation TEXT NOT NULL,
 payload TEXT NOT NULL
);
CREATE INDEX graph_edges_source ON graph_edges(source);
CREATE INDEX graph_edges_target ON graph_edges(target);
CREATE TABLE graph_evidence(
 edge_id TEXT NOT NULL REFERENCES graph_edges(id), ordinal INTEGER NOT NULL,
 payload TEXT NOT NULL, PRIMARY KEY(edge_id,ordinal)
);
CREATE TABLE graph_meta(key TEXT PRIMARY KEY, payload TEXT NOT NULL);
"""


def dumps(value):
    return json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False)


def initialize(path):
    connection = state.connect(path, journal_mode="DELETE")
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='server_schema'"
            ).fetchone()
            if not exists:
                for statement in MIGRATION_1.split(";"):
                    if statement.strip():
                        connection.execute(statement)
            version = connection.execute("SELECT MAX(version) FROM server_schema").fetchone()[0]
            if version == 1:
                for statement in ACCEPTED_GRAPH_SCHEMA.split(";"):
                    if statement.strip():
                        connection.execute(statement)
                connection.execute("INSERT INTO server_schema VALUES(2)")
                version = 2
            if version == 2:
                for statement in """
                    ALTER TABLE runs ADD COLUMN input_name TEXT;
                    CREATE TABLE live_documents(doc_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                        title TEXT, source_url TEXT, stage TEXT NOT NULL, updated_at TEXT NOT NULL);
                    CREATE TABLE live_events(id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL, doc_id TEXT, stage TEXT NOT NULL,
                        payload TEXT NOT NULL, created_at TEXT NOT NULL);
                    CREATE INDEX live_documents_run ON live_documents(run_id);
                    CREATE INDEX live_documents_active ON live_documents(updated_at DESC)
                        WHERE stage NOT IN ('resolved','failed','paused','completed','interrupted');
                    CREATE TABLE live_nodes(id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, payload TEXT NOT NULL);
                    CREATE TABLE live_edges(id TEXT PRIMARY KEY, doc_id TEXT NOT NULL,
                        source TEXT NOT NULL, target TEXT NOT NULL, payload TEXT NOT NULL);
                    CREATE INDEX live_nodes_document ON live_nodes(doc_id);
                    CREATE INDEX live_edges_document ON live_edges(doc_id);
                    INSERT INTO server_schema VALUES(3);
                """.split(";"):
                    if statement.strip():
                        connection.execute(statement)
                version = 3
            if version != 3:
                raise ValueError(f"Unsupported server schema version: {version}")
    finally:
        connection.close()


@contextmanager
def connect(path):
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA synchronous=FULL")
    try:
        yield connection
    finally:
        connection.close()


def save_graph(connection, bundle, *, resolved_document=None):
    """Publish a complete graph generation atomically, retaining canonical payloads."""
    validate_graph(bundle)
    with connection:
        if resolved_document:
            connection.execute("DELETE FROM live_edges WHERE doc_id=?", (resolved_document,))
            connection.execute("DELETE FROM live_nodes WHERE doc_id=?", (resolved_document,))
        for table in ("graph_evidence", "graph_edges", "graph_documents", "graph_nodes", "graph_meta"):
            connection.execute(f"DELETE FROM {table}")
        connection.executemany("INSERT INTO graph_nodes VALUES(?,?)", [
            (n["id"], dumps(n)) for n in bundle["nodes"]
        ])
        connection.executemany("INSERT INTO graph_documents VALUES(?,?)", [
            (d["doc_id"], dumps(d)) for d in bundle.get("documents", [])
        ])
        for edge in bundle["edges"]:
            payload = {key: value for key, value in edge.items() if key != "evidence"}
            connection.execute("INSERT INTO graph_edges VALUES(?,?,?,?,?)", (
                edge["id"], edge["source"], edge["target"], edge.get("relation", ""), dumps(payload)
            ))
            connection.executemany("INSERT INTO graph_evidence VALUES(?,?,?)", [
                (edge["id"], i, dumps(e)) for i, e in enumerate(edge.get("evidence", []))
            ])
        connection.executemany("INSERT INTO graph_meta VALUES(?,?)", [
            (key, dumps(value)) for key, value in bundle.items()
            if key not in ("nodes", "edges", "documents")
        ])
        connection.execute("INSERT OR REPLACE INTO graph_meta VALUES('revision',?)", (dumps(uuid.uuid4().hex),))


def read_graph(connection, limit=5000, offset=0, node_limit=5000, node_offset=0):
    # A single read transaction keeps endpoints and evidence in the same generation.
    connection.execute("BEGIN")
    meta = {r[0]: json.loads(r[1]) for r in connection.execute("SELECT * FROM graph_meta")}
    tables = """WITH nodes AS (
        SELECT id,payload FROM graph_nodes UNION ALL
        SELECT id,payload FROM live_nodes WHERE id NOT IN (SELECT id FROM graph_nodes)
    ), edges AS (
        SELECT id,source,target,payload FROM graph_edges UNION ALL
        SELECT id,source,target,payload FROM live_edges
    ) """
    edges = [json.loads(r[0]) for r in connection.execute(
        tables + "SELECT payload FROM edges ORDER BY id LIMIT ? OFFSET ?", (limit, offset)
    )]
    evidence = {edge["id"]: [] for edge in edges}
    for row in connection.execute(
        "SELECT edge_id,payload FROM graph_evidence "
        "WHERE edge_id IN (SELECT value FROM json_each(?)) ORDER BY edge_id,ordinal",
        (dumps(list(evidence)),)
    ):
        evidence[row[0]].append(json.loads(row[1]))
    for edge in edges:
        edge["evidence"] = evidence[edge["id"]] or edge.get("evidence", [])
    ids = {e[key] for e in edges for key in ("source", "target")}
    ids.update(row[0] for row in connection.execute(
        tables + "SELECT id FROM nodes ORDER BY id LIMIT ? OFFSET ?", (node_limit, node_offset)
    ))
    # json_each binds one parameter regardless of page size (older SQLite variable limits).
    nodes = [json.loads(r[0]) for r in connection.execute(
        tables + "SELECT payload FROM nodes WHERE id IN (SELECT value FROM json_each(?)) "
        "OR (?=0 AND json_extract(payload,'$.is_seed')=1) ORDER BY id", (dumps(sorted(ids)), offset)
    )]
    doc_ids = {d for e in edges for d in e.get("source_doc_ids", [])}
    documents = [json.loads(r[0]) for r in connection.execute(
        "SELECT payload FROM graph_documents WHERE id IN (SELECT value FROM json_each(?)) ORDER BY id",
        (dumps(sorted(doc_ids)),)
    )]
    existing_docs = {d["doc_id"] for d in documents}
    documents.extend(dict(r) for r in connection.execute(
        "SELECT doc_id,title,source_url FROM live_documents WHERE doc_id IN (SELECT value FROM json_each(?))",
        (dumps(sorted(doc_ids - existing_docs)),)
    ))
    total = connection.execute(tables + "SELECT COUNT(*) FROM edges").fetchone()[0]
    total_nodes = connection.execute(tables + "SELECT COUNT(*) FROM nodes").fetchone()[0]
    connection.rollback()
    return {**meta, "nodes": nodes, "edges": edges, "documents": documents,
            "pagination": {"offset": offset, "limit": limit, "total_edges": total,
                           "node_offset": node_offset, "node_limit": node_limit,
                           "total_nodes": total_nodes,
                           "next_edge_offset": min(offset + limit, total),
                           "next_node_offset": min(node_offset + node_limit, total_nodes)}}
