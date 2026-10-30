# Project Decisions — Supply Chain KG V1

This file summarizes the decisions currently considered frozen for V1. See `docs/SupplyChain_KG_V1_Architecture_Brief_Final.pdf` for the compact architecture brief.

## 1. Operating modes

### INGEST
Input is an existing document collection or dataset. The system normalizes the data, runs LLM extraction/disambiguation, stores intermediate outputs, and merges facts into Neo4j.

### EXPAND
Input is a configurable seed list. The system searches configured sources, retrieves relevant documents, extracts facts, and promotes newly discovered upstream companies into the next tier until the configured boundary is reached.

Existing-edge monitoring is not part of V1.

## 2. Seeds

The current experiment enables Toyota, Tesla Inc., BMW and BYD Co., Ltd. as Tier 0.
The battery and mining seed groups remain commented examples for a separate paper
replication experiment. `config/expansion.yaml` is authoritative; changes are explicit.

## 3. Boundary

Tier 0 contains seed entities. Upstream discovery continues through at most Tier 5 by
default; Tier 5 is terminal, so its entities
and relations are stored but never searched. `max_tier` remains configurable, and smoke
tests use a much smaller depth.

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
`Qwen/Qwen3-4B-Instruct-2507`, non-thinking/instruct behavior, served with vLLM.

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

## 12. User-authorized evidence and visualization upgrade (2026-09-08)

The user requested a paper/code review, efficiency improvements, a useful graph view,
and a new code version. Release 0.2.0 adds a local offline HTML evidence explorer;
this is an explicit exception to the earlier no-custom-visualization scope.
INGEST/EXPAND, static ontology, configurable seeds and one-explicit-source acceptance
remain the operating design. Source reputation is descriptive, not a confidence score.

Newly processed evidence must occur in the source text after Unicode/whitespace
normalization. Successful extraction chunks are checkpointed; default chunk errors
raise for resumable recovery. Distinct quote/chunk witnesses are persisted separately
and aggregate into one semantic edge. New validation/gold.py adds exact held-out
relation scoring; it is separate from external-reference coverage. These changes
are engineering improvements, not measured replication or accuracy claims.

See START_HERE.md, docs/UPGRADE_REVIEW.md and docs/TEST_RESULTS.md. Completed legacy
checkpoints are reused; re-extract into separate paths/database to compare quality.

## v0.3.0 audited policy

See `docs/AUDIT_V0.3.0.md`: one grounded valid source remains sufficient; reputation is
informational; only suppliesTo advances tiers; type-preserving frontier and cumulative
document/runtime budgets are required. Legacy unknown-grounding facts remain global
but cannot establish the core. The frontend was not changed.
