# Project Decisions — Supply Chain KG V1

This file summarizes the decisions currently considered frozen for V1. See `docs/SupplyChain_KG_V1_Architecture_Brief_Final.pdf` for the compact architecture brief.

## 1. Operating modes

### INGEST
Input is an existing document collection or dataset. The system normalizes the data, runs LLM extraction/disambiguation, stores intermediate outputs, and merges facts into Neo4j.

### EXPAND
Input is a configurable seed list. The system searches configured sources, retrieves relevant documents, extracts facts, and promotes newly discovered upstream companies into the next tier until the configured boundary is reached.

Existing-edge monitoring is not part of V1.

## 2. Seeds

The default seed configuration contains the 13 AlMahri-style companies:
- Toyota, Tesla Inc., BMW, BYD Co., Ltd.
- Samsung SDI, CATL, Panasonic
- Zijin Mining, Albemarle Corporation, Tianqi Lithium, Ganfeng Lithium Co Ltd, Norilsk Nickel, Jinchuan Group

Seeds are user-configurable and may be expanded.

## 3. Boundary

Tier 0 contains seed entities. Upstream discovery continues through at most Tier 3 by default. `max_tier` is configurable.

Only appropriate supply-chain relations advance the tier. Generic graph edges such as location or ownership should not be counted as supply tiers.

## 4. Ontology

V1 uses a static ontology.

Entity types:
`Company`, `Location`, `Material`, `Person`, `Product`, `Mine`

Core relations:
`suppliesTo`, `locatedIn`, `owns`, `produces`

No dynamic ontology growth in V1.

## 5. Graph expansion

Expansion is node-centric and tier-aware.

For each frontier company, query templates are selected from configuration according to its tier. The entity name is substituted into those templates. This avoids pairwise O(N^2)-style searches.

Configuration precedence:
`global defaults < tier defaults < query-specific overrides`

Each query has its own result limit.

Retrieval sources may include Wikipedia, Google News, and GDELT. Wikipedia is a configurable retrieval source; do not assume the pipeline always fetches only the entity's own Wikipedia page.

## 6. LLM stages

Prompt 1 performs schema-guided entity and relation extraction.

Prompt 2 performs entity disambiguation/canonicalization to prevent duplicate graph nodes.

Primary local model:
`Qwen/Qwen3.8-27B`, non-thinking/instruct behavior, served with vLLM.

The LLM is accessed through one provider interface so another API/local provider can be substituted without rewriting KG logic.

## 7. Data and evaluation

The exact packaged annotated AlMahri Wikipedia evaluation dataset has not been found publicly in our review.

Therefore:
- controlled existing documents are handled with INGEST,
- seed-driven discovery is handled with EXPAND.

Sectivia is selected as an external supplier-customer relationship reference dataset for validation/sanity checking, not as the main LLM text corpus.

## 8. Persistence

Resume is the default behavior.

A fresh run must be configurable.

Use persistent checkpoint/state storage so interrupted INGEST and EXPAND runs continue instead of restarting. Expansion frontier state must be persisted.

Intermediate documents and facts are stored outside Neo4j as well.

## 9. Deduplication

Prevent duplicate documents, entities, and graph writes using:
- URL normalization,
- content hashes,
- deterministic entity normalization,
- LLM disambiguation when needed,
- canonical IDs,
- Neo4j MERGE/idempotent writes.

## 10. Out of scope for V1

- existing-edge monitoring,
- relationship change/termination detection,
- pairwise verification of all existing edges,
- autonomous agent planning,
- custom visualization dashboards,
- risk analysis,
- optimization.


## 11. Final output and visualization

The primary final output of V1 is the knowledge graph stored in **Neo4j**.

Intermediate source data, extracted facts, and checkpoint state are also retained outside Neo4j so the pipeline remains reproducible and resumable.

For visualization, use **Neo4j Browser** for basic graph inspection and demonstration. No custom visualization dashboard or frontend is required in V1.
