# How the pipeline works

Reference for what SChain does to documents and to the graph. Setup and commands are in
the [README](../README.md); the frozen V1 scope is in
[PROJECT_DECISIONS.md](PROJECT_DECISIONS.md).

## The two modes

| Mode | Input | Path |
|---|---|---|
| `INGEST` | documents you already have | normalize → deduplicate → Prompt 1 → Prompt 2 → validation → facts → Neo4j `MERGE` (optional). No search. |
| `EXPAND` | seed companies | persistent frontier → tier-aware queries → retrieval → relevance and origin filters → Prompt 1 → Prompt 2 → ontology/evidence validation → facts → edge support → valid `suppliesTo` suppliers join the frontier at the next tier → Neo4j `MERGE` (optional), up to `max_tier`. |

Smoke, deep discovery, Colab and distributed runs are all `EXPAND` with a different
configuration; see [config/README.md](../config/README.md).

## Expansion depth

Tier is **graph distance along `supplier --suppliesTo--> customer`**, never industrial
role: a mining company selling directly to an OEM is Tier 1, not a deep tier. Only
`suppliesTo` advances depth; `locatedIn`, `owns` and `produces` describe an entity without
adding any.

The default experiment has four OEM seeds and `max_tier: 5`: Tiers 0–4 are searched and
Tier 5 is terminal (stored, never searched). Per-tier document targets and maxima, the
per-run document ceiling (`expansion.max_documents`) and the wall-clock budget
(`runtime.max_minutes`) are all in the config files, and are cumulative across resumes.

## Retrieval sources

Four providers, all feeding **one** pipeline — there is no separate handling per source:

| Family | Providers | Default ceiling per query |
|---|---|---|
| News | Google News (RSS search), GDELT (DOC 2.0 API) | 10 and 16 candidates |
| General web | `general_web` — supplier/sourcing pages, press releases, investor relations, annual and sustainability reports, filings | 18 candidates |
| Reference | Wikipedia (MediaWiki search API) | 4 candidates |

News carries new supply agreements; the general web carries the older, more stable
sourcing information. `general_web` defaults to a DuckDuckGo backend needing no
credentials, and uses the official Brave Search API instead when `backend: brave` is set
and `BRAVE_SEARCH_API_KEY` is present. A failing or disabled provider never affects the
others; its query is recorded as failed and retried on the next run.

**Wikipedia has no special role.** It is a search provider like the others: a generated
query such as `"BMW" suppliers` goes to its search endpoint and the matching articles come
back. The pipeline never assumes the entity's own page is the document to fetch.

Limits are ceilings on candidates **per provider, per query** — not a target number of
relevant documents. Every candidate goes through the same path: URL dedup → cheap
title/snippet relevance check → origin relevance → ranking → full-text fetch where
possible → content dedup → Prompt 1. Aliases recorded by entity disambiguation for the
entity being expanded count as mentions of it throughout.

**Known limitation.** Google News RSS returns redirect/wrapper links, so an article body
often cannot be extracted. The document then falls back to title + snippet rather than
being dropped, and records which it was:

```text
content_source = full_text | title_snippet | file
```

visible in `data/processed/documents.parquet` and in the search-results layer. Extraction
from a headline simply yields fewer facts. V1 does not attempt to defeat the redirects —
no scraping stack, no browser automation.

## What one document may contribute

There is no per-document cap on facts. Prompt 1 extracts everything the usable text
evidences within the static ontology, so one document may yield two entities and one
relation, or a dozen entities and several relations. Long documents are split by the
configured chunking; after chunks are combined, repeated entities and facts collapse (a
fact id is derived from document plus canonical endpoints plus relation), and evidence is
retained.

## When a fact is accepted

A fact is accepted when the retrieved text explicitly supports it: **one valid source is
enough**. SChain does not require several independent websites to agree; multi-source
verification is disabled (`verification.enabled: false`, and `schain/verification.py` is
a no-op stub kept for imports).

A `suppliesTo` edge advances the frontier when the subject is a `Company`, the ontology
pattern is valid, evidence validation supports the relation, and the subject is not a
generic category phrase such as "chip suppliers". Title/snippet evidence counts when it is
explicit.

Source reputation (`config/source_reputation.yaml`, `schain/sources/reputation.py`) is
informational. Scores appear on documents, facts, evidence and the graph export, but they
never reject a fact, block an edge, change frontier eligibility or require full text; a
Google News wrapper may score `0` and still yield an accepted edge. Each exported edge
carries a quality object:

```json
"quality": {
  "status": "accepted",
  "verification_status": "not_required",
  "max_source_score": 94,
  "average_source_score": 72,
  "ontology_valid": true,
  "evidence_validated": true,
  "frontier_eligible": true,
  "contested": false
}
```

`support_count` is the number of unique accepted documents behind the semantic edge;
`independent_support_count` is deprecated and exported as `null`. URL, content-hash,
document-id and fact-id deduplication still run.

## What V1 does to the graph, exactly

After Prompt 1 and Prompt 2, four things happen and nothing else:

1. **MERGE node** — on `canonical_id`. A new canonical entity is created; an existing one
   is reused. Stable identity is preserved and properties are not blindly overwritten;
   aliases accumulate as a union.
2. **MERGE relationship** — a new structural relationship is created; an existing one is
   reused. Several documents supporting the same relationship do **not** produce several
   edges: `source_doc_ids` accumulates on the one edge.
3. **Persist evidence** — document id, source URL, evidence text, model and retrieval
   metadata, in the facts Parquet layer.
4. **Expand** — an eligible new `Company` discovered through `suppliesTo` enters the
   persistent frontier when its tier is within `max_tier`.

**MERGE means "do not duplicate canonical nodes and edges". It does not mean "update the
real world".** V1 performs no temporal updates: no supplier-status changes, no contract
termination detection, no edge or node deletion, no relationship replacement, no periodic
monitoring. If an older document says CATL supplies BMW and a newer one reports the
agreement ended, V1 records both as evidence and changes nothing structurally.

## Resume, document by document

Every expensive stage records its completion in the state database, so nothing
expensive runs twice:

| Document state | Meaning | A resumed run does |
|---|---|---|
| `PENDING` | registered only | Prompt 1 |
| `EXTRACTED` | Prompt 1 done, output on disk | Prompt 2 onwards — Prompt 1 is **not** repeated |
| `RESOLVED` | facts written to Parquet | the Neo4j write only — no LLM call at all |
| `COMMITTED` | merged into Neo4j | skips the document |
| `FAILED` | Prompt 1 never completed | retries Prompt 1 |

The `EXPAND` frontier and every search query are persisted too, so a crash mid-run does
not re-issue searches or re-expand finished companies. An experiment's identity (seeds,
`max_tier`, model and generation settings, ingest and validation policy, ontology) is
recorded on the first run; resuming with a different identity is refused, while
operational limits may change and are recorded as config revisions.

## Retrieval audit manifest

Each `EXPAND` run writes `logs/retrieval_manifest_<run_id>.csv` — one row per candidate a
provider returned, including candidates that were deduplicated, judged irrelevant, failed
to fetch, or never reached the LLM. Columns cover the query and provider, title, URL,
domain, publication and retrieval time, snippet and content lengths, `fetch_status`,
`http_status`, `content_mode`, `dedup_status`, `relevance_status`, `llm_status` and a
short `error` code. Article text is not copied into it.

It is for audit and debugging only — SQLite and Parquet remain the source of truth, and
rows are keyed so a resumed run updates them instead of appending duplicates. A compact
summary (raw candidates, unique URLs, fetch outcomes, full-text vs snippet-only,
relevance, how many reached Prompt 1, per provider and per tier) is printed at the end of
every run.

## Optional audit trace

`SCHAIN_AUDIT_TRACE=1` writes one self-contained audit JSON beside each graph export
(`outputs/run.v1.json` gets `outputs/run.v1.audit.json`). It adds no model calls and
changes no decision. `SCHAIN_AUDIT_FULL_CONTENT=1` also stores source and chunk text, the
exact model inputs and responses, parsed graphs, evidence and resolution payloads. The
YAML equivalents are `audit.enabled` and `audit.capture_content`.

The JSON (schema version `1.0`) has sections for the run, configuration, documents,
chunks, LLM calls, extraction graphs, integrity checks, entity resolution, candidate,
accepted and rejected facts, validation decisions, repair attempts, relationship
outcomes, graph export, events, errors and a summary. The chronological `events` list is
the lossless source, and stable IDs join documents, chunks, LLM calls, facts and exported
edges. Anything the model interface does not expose, such as the raw HTTP body or token
usage, is marked unavailable rather than reconstructed. Events are journalled during the
run and the final JSON is assembled atomically; a failed run sets `run.complete: false`.

Full-content traces can contain confidential documents, prompts and responses: keep them
in the ignored `outputs/` directory and never commit them. Secrets in the configuration
are redacted.

## Evaluating extraction quality

`python -m schain evaluate` scores accepted relations against a held-out, fully labelled
gold file:

```json
{
  "documents": [
    {
      "doc_id": "actual-document-id",
      "triples": [
        {"subject_id": "actual-supplier-canonical-id", "relation": "suppliesTo", "object_id": "actual-customer-canonical-id"}
      ]
    },
    {"doc_id": "actual-negative-document-id", "triples": []}
  ]
}
```

Label every ontology-allowed relation in each included document, including documents
with none, and keep these documents out of prompt tuning. Then:

```bash
python -m schain evaluate --config config/ingest.yaml --gold path/to/gold.json --output outputs/evaluation.json
```

It reports exact directed precision, recall and F1. Duplicate evidence does not inflate
the score, and predictions in documents outside the gold set are excluded. It does not
judge isolated entities, source truth or temporal validity. `python -m schain validate`
is different: it measures overlap with an external reference dataset (see
[EXTERNAL_DATASETS.md](EXTERNAL_DATASETS.md)), not precision.

## Inspecting the graph in Neo4j

Use Neo4j Browser at http://localhost:7474.

```cypher
// the supply network
MATCH (supplier:Company)-[r:suppliesTo]->(customer:Company)
RETURN supplier.name, customer.name, r.evidence LIMIT 50;

// who supplies BMW
MATCH (s)-[:suppliesTo]->(c:Company {canonical_id: 'company:bmw'}) RETURN s.name;

// entities and their recorded aliases
MATCH (e:Entity) RETURN labels(e), e.name, e.aliases LIMIT 50;

// which documents support an edge
MATCH ()-[r:suppliesTo]->() RETURN r.evidence, r.source_doc_ids LIMIT 25;
```

## Design notes

- **Reused from neo4j-graphrag 1.19.0**: `FixedSizeSplitter`, `LLMEntityRelationExtractor`
  (Prompt 1), `GraphSchema`, `PdfLoader`, `OpenAILLM`, and the official Neo4j driver.
  `SimpleKGPipeline` is not used: it mandates an embedder and a chunk-embedding stage that
  V1 has no use for.
- **Custom Neo4j writer.** The library's `Neo4jWriter` issues `CREATE` and relies on a later
  resolver pass (`apoc.refactor.mergeNodes`) to collapse duplicates.
  [PROJECT_DECISIONS.md](PROJECT_DECISIONS.md) §9 requires the opposite: canonical ids
  plus `MERGE`, so a duplicate is never created. APOC is not required.
- **Custom Prompt 2.** The library ships exact/fuzzy/spaCy resolvers, not an LLM
  disambiguator. Four stages run cheapest-first: the persisted resolution cache,
  deterministic normalization (case, punctuation, accents, legal suffixes), the known-alias
  map in `config/ontology.yaml`, and only then one LLM call per entity type.
- **Stable canonical ids.** A group's canonical name is the longest mention in it, not
  whatever the model calls it, and a group reaching an already-known entity reuses that
  entity's id. Repeated runs therefore cannot produce duplicate nodes.
- **Two independent JSON controls.** `llm.json_mode` asks the server for guaranteed
  parseable JSON and applies to both prompts. `llm.structured_output` additionally
  constrains Prompt 1 with the full JSON schema, which needs a grammar backend on the
  server and is off by default.
- **No lexical graph in Neo4j.** `create_lexical_graph=False`, so Neo4j holds only the
  static ontology; the evidence trail lives in the facts Parquet layer.

## Code layout

```text
schain/
├── __main__.py             CLI: ingest | expand | export | view | validate | evaluate
├── config.py               config loading, include merge, search-limit precedence
├── pipeline.py             the per-document path shared by both modes
├── ingest.py               INGEST driver
├── documents.py            loading, normalization, hashing, raw/normalized layers
├── names.py                deterministic name normalization and canonical ids
├── ontology.py             static ontology -> GraphSchema, and Prompt 1
├── state.py                SQLite checkpoint/resume state
├── state_backend.py        local SQLite versus distributed PostgreSQL state
├── run_config.py           experiment identity, safe config snapshots, provenance
├── checkpoint.py           atomic writes for expensive intermediate results
├── durable.py              durable snapshots for ephemeral runtimes (Colab)
├── async_runtime.py        one long-lived event loop for async clients
├── audit_trace.py          optional run-scoped audit trace
├── graph_export.py         frontend-ready semantic graph JSON (derived artifact)
├── viewer.py               offline HTML graph explorer (assets in viewer_assets/)
├── verification.py         no-op stub (multi-source verification removed)
├── llm/                    provider factory (vLLM or a hosted OpenAI-compatible API)
├── kg/                     extraction, resolution, validation, facts, support, writer
├── sources/                retrieval providers, ranking, origin relevance, reputation
├── expansion/              frontier, tier queries, escalation, budgets, the EXPAND loop
├── distributed/            PostgreSQL leases, outbox and maintenance for many workers
└── validation/             external reference comparison and gold-set evaluation
```

Outside the package: `config/` (see [config/README.md](../config/README.md)),
`scripts/` (the GPU server runner, status and preflight), `notebooks/` (Colab),
`examples/` and `demo/` (standalone graph explorers), and `tests/`. Everything under
`data/` and `outputs/` is generated and gitignored.

Keep modules narrow: `sources/` only retrieves and scores domains, `llm/` only builds the
model client, `kg/` holds extraction through the idempotent writer, `expansion/` decides
what is processed next, `state.py` only persists, and `pipeline.py`, `ingest.py` and
`expansion/loop.py` orchestrate. The package lives in `schain/` rather than `src/` so that
`python -m schain` works as the CLI.
