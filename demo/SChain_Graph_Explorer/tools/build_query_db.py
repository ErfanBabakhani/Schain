#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "data" / "professor_supply_chain_graph.v1.json"
DEFAULT_DB = ROOT / "data" / "professor_supply_chain_graph.sqlite"


def as_integer_flag(value):
    return 1 if bool(value) else 0


def encode_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build(json_path: Path, db_path: Path) -> None:
    graph = json.loads(json_path.read_text(encoding="utf-8"))
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    documents = graph.get("documents", [])
    metadata = graph.get("metadata", {})
    statistics = graph.get("statistics", {})
    completion = graph.get("completion", metadata.get("completion", {})) or {}

    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.executescript("""
    PRAGMA journal_mode=DELETE;
    PRAGMA foreign_keys=ON;

    CREATE TABLE metadata (
      key TEXT PRIMARY KEY,
      value TEXT,
      value_json TEXT
    );

    CREATE TABLE statistics (
      key TEXT PRIMARY KEY,
      value TEXT,
      value_json TEXT
    );

    CREATE TABLE nodes (
      id TEXT PRIMARY KEY,
      type TEXT,
      name TEXT,
      discovery_tier INTEGER,
      tier_parent_id TEXT,
      discovered_from TEXT,
      is_seed INTEGER NOT NULL DEFAULT 0,
      is_expandable INTEGER NOT NULL DEFAULT 0,
      is_supply_chain_node INTEGER NOT NULL DEFAULT 0,
      in_degree INTEGER,
      out_degree INTEGER,
      edge_count INTEGER,
      support_documents INTEGER,
      aliases_json TEXT,
      tier_parent_ids_json TEXT,
      source_doc_ids_json TEXT
    );

    CREATE TABLE node_aliases (
      node_id TEXT NOT NULL,
      alias TEXT NOT NULL,
      PRIMARY KEY(node_id, alias),
      FOREIGN KEY(node_id) REFERENCES nodes(id)
    );

    CREATE TABLE node_parents (
      node_id TEXT NOT NULL,
      parent_id TEXT NOT NULL,
      PRIMARY KEY(node_id, parent_id),
      FOREIGN KEY(node_id) REFERENCES nodes(id)
    );

    CREATE TABLE node_documents (
      node_id TEXT NOT NULL,
      doc_id TEXT NOT NULL,
      PRIMARY KEY(node_id, doc_id),
      FOREIGN KEY(node_id) REFERENCES nodes(id)
    );

    CREATE TABLE edges (
      id TEXT PRIMARY KEY,
      source_id TEXT NOT NULL,
      target_id TEXT NOT NULL,
      source_name TEXT,
      target_name TEXT,
      source_type TEXT,
      target_type TEXT,
      source_tier INTEGER,
      target_tier INTEGER,
      relation TEXT NOT NULL,
      support_count INTEGER,
      evidence_count INTEGER,
      expansion_relevant INTEGER NOT NULL DEFAULT 0,
      is_supply_chain_edge INTEGER NOT NULL DEFAULT 0,
      frontier_eligible INTEGER NOT NULL DEFAULT 0,
      quality_status TEXT,
      verification_status TEXT,
      max_source_score REAL,
      average_source_score REAL,
      full_text_support_count INTEGER,
      ontology_valid INTEGER,
      evidence_validated INTEGER,
      evidence_quote_grounded INTEGER,
      direction_explicit INTEGER,
      frontier_rejection_reason TEXT,
      fact_ids_json TEXT,
      source_doc_ids_json TEXT,
      source_urls_json TEXT,
      quality_json TEXT,
      FOREIGN KEY(source_id) REFERENCES nodes(id),
      FOREIGN KEY(target_id) REFERENCES nodes(id)
    );

    CREATE TABLE edge_documents (
      edge_id TEXT NOT NULL,
      doc_id TEXT NOT NULL,
      PRIMARY KEY(edge_id, doc_id),
      FOREIGN KEY(edge_id) REFERENCES edges(id)
    );

    CREATE TABLE edge_urls (
      edge_id TEXT NOT NULL,
      source_url TEXT NOT NULL,
      PRIMARY KEY(edge_id, source_url),
      FOREIGN KEY(edge_id) REFERENCES edges(id)
    );

    CREATE TABLE evidence (
      evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
      edge_id TEXT NOT NULL,
      fact_id TEXT,
      doc_id TEXT,
      chunk_index INTEGER,
      subject_id TEXT,
      subject_name TEXT,
      relation TEXT,
      object_id TEXT,
      object_name TEXT,
      quote TEXT,
      source_url TEXT,
      retrieval_url TEXT,
      canonical_source_url TEXT,
      source_domain TEXT,
      source_class TEXT,
      source_score REAL,
      source_known INTEGER,
      source_group_id TEXT,
      document_title TEXT,
      content_source TEXT,
      provider TEXT,
      model TEXT,
      extracted_at TEXT,
      validation_status TEXT,
      ontology_valid INTEGER,
      evidence_validated INTEGER,
      evidence_quote_grounded INTEGER,
      direction_explicit INTEGER,
      origin_entity TEXT,
      expansion_relevant INTEGER,
      raw_json TEXT,
      FOREIGN KEY(edge_id) REFERENCES edges(id)
    );

    CREATE TABLE documents (
      doc_id TEXT PRIMARY KEY,
      title TEXT,
      source_url TEXT,
      retrieval_url TEXT,
      canonical_source_url TEXT,
      source_file TEXT,
      content_source TEXT,
      char_count INTEGER,
      content_hash TEXT,
      ingested_at TEXT,
      provider TEXT,
      published_at TEXT,
      source_domain TEXT,
      source_class TEXT,
      source_base_score REAL,
      source_effective_score REAL,
      source_known INTEGER,
      publisher_name TEXT,
      publisher_domain TEXT,
      origin_entity TEXT,
      query TEXT,
      tier INTEGER,
      extraction_metrics_json TEXT,
      raw_json TEXT
    );

    CREATE TABLE raw_graph (
      id INTEGER PRIMARY KEY CHECK(id=1),
      json TEXT NOT NULL
    );
    """)


    meta_items = {
        **metadata,
        "completion": completion,
        "schema_version": graph.get("schema_version"),
    }
    for key, value in meta_items.items():
        scalar = value if isinstance(value, (str, int, float, bool)) or value is None else None
        cursor.execute("INSERT OR REPLACE INTO metadata(key,value,value_json) VALUES(?,?,?)",
                    (str(key), None if scalar is None else str(scalar), encode_json(value)))
    for key, value in statistics.items():
        scalar = value if isinstance(value, (str, int, float, bool)) or value is None else None
        cursor.execute("INSERT OR REPLACE INTO statistics(key,value,value_json) VALUES(?,?,?)",
                    (str(key), None if scalar is None else str(scalar), encode_json(value)))

    node_map = {node.get("id"): node for node in nodes}
    for node in nodes:
        stats = node.get("stats", {}) or {}
        cursor.execute("""
          INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            node.get("id"), node.get("type"), node.get("name"), node.get("discovery_tier"),
            node.get("tier_parent_id"), node.get("discovered_from"), as_integer_flag(node.get("is_seed")),
            as_integer_flag(node.get("is_expandable")), as_integer_flag(node.get("is_supply_chain_node")),
            stats.get("in_degree"), stats.get("out_degree"), stats.get("edge_count"),
            stats.get("support_documents"), encode_json(node.get("aliases", [])),
            encode_json(node.get("tier_parent_ids", [])), encode_json(node.get("source_doc_ids", []))
        ))
        for alias in node.get("aliases", []) or []:
            cursor.execute("INSERT OR IGNORE INTO node_aliases(node_id,alias) VALUES(?,?)", (node.get("id"), str(alias)))
        for pid in node.get("tier_parent_ids", []) or []:
            cursor.execute("INSERT OR IGNORE INTO node_parents(node_id,parent_id) VALUES(?,?)", (node.get("id"), str(pid)))
        for did in node.get("source_doc_ids", []) or []:
            cursor.execute("INSERT OR IGNORE INTO node_documents(node_id,doc_id) VALUES(?,?)", (node.get("id"), str(did)))

    for edge in edges:
        quality = edge.get("quality", {}) or {}
        source_node, target_node = node_map.get(edge.get("source"), {}), node_map.get(edge.get("target"), {})
        cursor.execute("""
          INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            edge.get("id"), edge.get("source"), edge.get("target"), source_node.get("name"), target_node.get("name"),
            source_node.get("type"), target_node.get("type"), source_node.get("discovery_tier"), target_node.get("discovery_tier"),
            edge.get("relation"), edge.get("support_count"), edge.get("evidence_count"),
            as_integer_flag(edge.get("expansion_relevant")), as_integer_flag(edge.get("is_supply_chain_edge")), as_integer_flag(quality.get("frontier_eligible")),
            quality.get("status"), quality.get("verification_status"), quality.get("max_source_score"),
            quality.get("average_source_score"), quality.get("full_text_support_count"),
            as_integer_flag(quality.get("ontology_valid")), as_integer_flag(quality.get("evidence_validated")), as_integer_flag(quality.get("evidence_quote_grounded")),
            as_integer_flag(quality.get("direction_explicit")), quality.get("frontier_rejection_reason"),
            encode_json(edge.get("fact_ids", [])), encode_json(edge.get("source_doc_ids", [])), encode_json(edge.get("source_urls", [])), encode_json(quality)
        ))
        for did in edge.get("source_doc_ids", []) or []:
            cursor.execute("INSERT OR IGNORE INTO edge_documents(edge_id,doc_id) VALUES(?,?)", (edge.get("id"), str(did)))
        for url in edge.get("source_urls", []) or []:
            cursor.execute("INSERT OR IGNORE INTO edge_urls(edge_id,source_url) VALUES(?,?)", (edge.get("id"), str(url)))
        for evidence in edge.get("evidence", []) or []:
            cursor.execute("""
              INSERT INTO evidence(
                edge_id,fact_id,doc_id,chunk_index,subject_id,subject_name,relation,object_id,object_name,
                quote,source_url,retrieval_url,canonical_source_url,source_domain,source_class,source_score,
                source_known,source_group_id,document_title,content_source,provider,model,extracted_at,
                validation_status,ontology_valid,evidence_validated,evidence_quote_grounded,direction_explicit,
                origin_entity,expansion_relevant,raw_json
              ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                edge.get("id"), evidence.get("fact_id"), evidence.get("doc_id"), evidence.get("chunk_index"),
                edge.get("source"), source_node.get("name"), edge.get("relation"), edge.get("target"), target_node.get("name"),
                evidence.get("quote"), evidence.get("source_url"), evidence.get("retrieval_url"), evidence.get("canonical_source_url"),
                evidence.get("source_domain"), evidence.get("source_class"), evidence.get("source_score"), as_integer_flag(evidence.get("source_known")),
                evidence.get("source_group_id"), evidence.get("document_title"), evidence.get("content_source"), evidence.get("provider"),
                evidence.get("model"), evidence.get("extracted_at"), evidence.get("validation_status"), as_integer_flag(evidence.get("ontology_valid")),
                as_integer_flag(evidence.get("evidence_validated")), as_integer_flag(evidence.get("evidence_quote_grounded")), as_integer_flag(evidence.get("direction_explicit")),
                evidence.get("origin_entity"), as_integer_flag(evidence.get("expansion_relevant")), encode_json(evidence)
            ))

    for document in documents:
        cursor.execute("""
          INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            document.get("doc_id"), document.get("title"), document.get("source_url"), document.get("retrieval_url"),
            document.get("canonical_source_url"), document.get("source_file"), document.get("content_source"), document.get("char_count"),
            document.get("content_hash"), document.get("ingested_at"), document.get("provider"), document.get("published_at"),
            document.get("source_domain"), document.get("source_class"), document.get("source_base_score"),
            document.get("source_effective_score"), as_integer_flag(document.get("source_known")), document.get("publisher_name"),
            document.get("publisher_domain"), document.get("origin_entity"), document.get("query"), document.get("tier"),
            encode_json(document.get("extraction_metrics", {})), encode_json(document)
        ))

    cursor.execute("INSERT INTO raw_graph(id,json) VALUES(1,?)", (json.dumps(graph, ensure_ascii=False),))

    cursor.executescript("""
    CREATE INDEX idx_nodes_name ON nodes(name);
    CREATE INDEX idx_nodes_type ON nodes(type);
    CREATE INDEX idx_nodes_tier ON nodes(discovery_tier);
    CREATE INDEX idx_nodes_core ON nodes(is_supply_chain_node);
    CREATE INDEX idx_edges_source ON edges(source_id);
    CREATE INDEX idx_edges_target ON edges(target_id);
    CREATE INDEX idx_edges_relation ON edges(relation);
    CREATE INDEX idx_edges_core ON edges(is_supply_chain_edge);
    CREATE INDEX idx_evidence_edge ON evidence(edge_id);
    CREATE INDEX idx_evidence_doc ON evidence(doc_id);
    CREATE INDEX idx_evidence_domain ON evidence(source_domain);
    CREATE INDEX idx_documents_provider ON documents(provider);
    CREATE INDEX idx_documents_origin ON documents(origin_entity);

    CREATE VIEW supply_chain_core AS
      SELECT
        e.id AS edge_id,
        e.source_id AS supplier_id,
        e.source_name AS supplier,
        e.source_tier AS supplier_tier,
        e.target_id AS customer_id,
        e.target_name AS customer,
        e.target_tier AS customer_tier,
        e.relation,
        e.support_count,
        e.evidence_count,
        e.quality_status,
        e.max_source_score
      FROM edges e
      WHERE e.is_supply_chain_edge=1 AND e.relation='suppliesTo';

    CREATE VIEW graph_relations AS
      SELECT
        e.id AS edge_id,
        e.source_name AS subject,
        e.relation,
        e.target_name AS object,
        e.source_type,
        e.target_type,
        e.source_tier,
        e.target_tier,
        e.support_count,
        e.evidence_count,
        e.is_supply_chain_edge,
        e.expansion_relevant,
        e.quality_status
      FROM edges e;

    CREATE VIEW supply_chain_evidence AS
      SELECT
        e.id AS edge_id,
        e.source_name AS supplier,
        e.target_name AS customer,
        e.source_tier AS supplier_tier,
        e.target_tier AS customer_tier,
        ev.fact_id,
        ev.quote,
        ev.document_title,
        ev.source_domain,
        ev.source_url,
        ev.provider,
        ev.content_source,
        ev.source_score
      FROM edges e JOIN evidence ev ON ev.edge_id=e.id
      WHERE e.is_supply_chain_edge=1 AND e.relation='suppliesTo';

    CREATE VIEW context_relations AS
      SELECT * FROM graph_relations WHERE is_supply_chain_edge=0;

    CREATE VIEW tier_summary AS
      SELECT discovery_tier AS tier, COUNT(*) AS node_count
      FROM nodes WHERE discovery_tier IS NOT NULL
      GROUP BY discovery_tier ORDER BY discovery_tier;

    CREATE VIEW seed_suppliers AS
      SELECT
        e.source_name AS supplier,
        e.target_name AS seed_company,
        e.source_tier AS supplier_tier,
        e.support_count,
        e.evidence_count
      FROM edges e JOIN nodes n ON n.id=e.target_id
      WHERE e.is_supply_chain_edge=1 AND e.relation='suppliesTo' AND n.is_seed=1;

    CREATE VIEW provider_summary AS
      SELECT provider,
             COUNT(*) AS documents,
             SUM(CASE WHEN content_source='full_text' THEN 1 ELSE 0 END) AS full_text_documents
      FROM documents GROUP BY provider ORDER BY documents DESC;
    """)


    try:
        cursor.executescript("""
        CREATE VIRTUAL TABLE evidence_fts USING fts5(quote, document_title, subject_name, object_name, content='evidence', content_rowid='evidence_id');
        INSERT INTO evidence_fts(rowid,quote,document_title,subject_name,object_name)
          SELECT evidence_id,quote,document_title,subject_name,object_name FROM evidence;
        CREATE VIRTUAL TABLE documents_fts USING fts5(title, source_domain, query, content='documents', content_rowid='rowid');
        INSERT INTO documents_fts(rowid,title,source_domain,query)
          SELECT rowid,title,source_domain,query FROM documents;
        """)
    except sqlite3.OperationalError:
        pass

    connection.commit()
    cursor.execute("VACUUM")
    connection.close()
    print(f"Built {db_path} from {json_path} ({len(nodes)} nodes, {len(edges)} edges, {len(documents)} documents)")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DB
    build(src, dst)
