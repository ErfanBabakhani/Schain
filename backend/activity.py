"""Durable worker progress and provisional graph rows, separate from accepted facts."""
import hashlib
import json
import uuid

from schain.names import canonical_id
from schain.state import _now

from .storage import connect, dumps


def invalidate_graph(connection):
    connection.execute("INSERT OR REPLACE INTO graph_meta VALUES('revision',?)", (dumps(uuid.uuid4().hex),))


def clear_pending(connection, doc_id):
    edges = connection.execute("DELETE FROM live_edges WHERE doc_id=?", (doc_id,)).rowcount
    nodes = connection.execute("DELETE FROM live_nodes WHERE doc_id=?", (doc_id,)).rowcount
    if edges or nodes:
        invalidate_graph(connection)


def preview(connection, document, payload):
    """Preview mentions only; these rows never participate in frontier discovery."""
    doc_id = document["doc_id"]
    graph = payload["graph"]
    nodes = {}
    for node in graph.get("nodes", []):
        name = node.get("properties", {}).get("name") or node["id"]
        label = node.get("label") or "Unknown"
        nodes[node["id"]] = {"id": canonical_id(label, str(name)), "name": str(name),
                             "type": label, "validation_status": "pending"}
    chunk_id = str(payload.get("chunk_index", "document"))
    rows = []
    for index, relation in enumerate(graph.get("relationships", [])):
        source = nodes.get(relation.get("start_node_id"))
        target = nodes.get(relation.get("end_node_id"))
        if source is None or target is None:
            continue  # Dangling extraction endpoints are not invented for the preview.
        identity = hashlib.sha256(f"{doc_id}:{chunk_id}:{index}".encode()).hexdigest()
        edge = {"id": "pending:" + identity, "source": source["id"], "target": target["id"],
                "relation": relation.get("type", "relatedTo"), "validation_status": "pending",
                "source_doc_ids": [doc_id], "evidence": [], "support_count": 0}
        rows.append((edge["id"], doc_id, edge["source"], edge["target"], dumps(edge)))
    if not rows:
        return
    used = {row[key] for row in rows for key in (2, 3)}
    connection.executemany("INSERT OR REPLACE INTO live_nodes VALUES(?,?,?)", [
        (node["id"], doc_id, dumps(node)) for node in nodes.values() if node["id"] in used
    ])
    connection.executemany("INSERT OR REPLACE INTO live_edges VALUES(?,?,?,?,?)", rows)
    invalidate_graph(connection)


def record(path, run_id, stage, document=None, payload=None):
    payload = payload or {}
    document = document or {}
    doc_id = document.get("doc_id")
    # Raw document text, prompts and model responses are never progress payloads.
    detail = {key: payload[key] for key in (
        "chunk_index", "chunk_total", "cached", "error", "accepted", "rejected", "status"
    ) if key in payload}
    detail["title"] = document.get("title") or doc_id or "Run"
    with connect(path) as connection, connection:
        if doc_id:
            phase = "extraction" if stage.startswith("chunk_") else stage
            connection.execute(
                "INSERT INTO live_documents VALUES(?,?,?,?,?,?) ON CONFLICT(doc_id) DO UPDATE SET "
                "run_id=excluded.run_id,title=excluded.title,source_url=excluded.source_url,"
                "stage=excluded.stage,updated_at=excluded.updated_at",
                (doc_id, run_id, detail["title"], document.get("source_url"), phase, _now())
            )
            if stage == "chunk_extracted":
                preview(connection, document, payload)
                detail["relationships"] = len(payload["graph"].get("relationships", []))
            if stage in ("failed", "resolved"):
                clear_pending(connection, doc_id)
        connection.execute(
            "INSERT INTO live_events(run_id,doc_id,stage,payload,created_at) VALUES(?,?,?,?,?)",
            (run_id, doc_id, stage, dumps(detail), _now())
        )


def finish(path, run_id, status):
    with connect(path) as connection, connection:
        for row in connection.execute("SELECT doc_id FROM live_documents WHERE run_id=?", (run_id,)).fetchall():
            clear_pending(connection, row[0])
        connection.execute(
            "UPDATE live_documents SET stage=?,updated_at=? WHERE run_id=? "
            "AND stage NOT IN ('resolved','failed')", (status, _now(), run_id)
        )
    record(path, run_id, "run_" + status, payload={"status": status})


def read_activity(connection, after=None, limit=100):
    connection.execute("BEGIN")
    latest = connection.execute("SELECT COALESCE(MAX(id),0) FROM live_events").fetchone()[0]
    if after is None:
        rows = list(reversed(connection.execute(
            "SELECT * FROM live_events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()))
    else:
        rows = connection.execute(
            "SELECT * FROM live_events WHERE id>? ORDER BY id LIMIT ?", (after, limit)
        ).fetchall()
    events = [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]
    cursor = events[-1]["id"] if events else latest
    active = [dict(row) for row in connection.execute(
        "SELECT doc_id,title,source_url,stage,updated_at,run_id FROM live_documents "
        "WHERE stage NOT IN ('resolved','failed','paused','completed','interrupted') "
        "AND run_id IN (SELECT id FROM runs WHERE status IN ('running','pausing')) ORDER BY updated_at DESC"
    )]
    entities = [dict(row) for row in connection.execute(
        "SELECT entity_name,tier FROM frontier WHERE status='PROCESSING' "
        "AND EXISTS (SELECT 1 FROM runs WHERE status IN ('running','pausing')) ORDER BY tier,entity_key LIMIT 20"
    )]
    pending = connection.execute("SELECT COUNT(*) FROM live_edges").fetchone()[0]
    connection.rollback()
    return {"events": events, "cursor": cursor, "has_more": cursor < latest,
            "active_documents": active, "entities": entities, "pending_relationships": pending}
