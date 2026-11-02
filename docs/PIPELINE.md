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
See [the v0.3.0 audit](AUDIT_V0.3.0.md) for the reasoning behind the tier allocations.

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
