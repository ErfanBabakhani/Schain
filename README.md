> **Updated code: 0.3.0 (8 September 2026).** Start with [START_HERE.md](START_HERE.md)
> for setup and migration; read [the v0.3.0 audit](docs/AUDIT_V0.3.0.md) for corrected retrieval, tiers, budgets and tests. See
> [docs/UPGRADE_REVIEW.md](docs/UPGRADE_REVIEW.md) for the paper comparison.
> Targeted graph-integrity, evidence-recovery, event-loop, and optional audit-trace fixes
> are documented in [docs/RELIABILITY_FIXES_V030.md](docs/RELIABILITY_FIXES_V030.md).
> The downloadable end-to-end companion JSON is documented in
> [docs/AUDIT_TRACE_JSON.md](docs/AUDIT_TRACE_JSON.md).
> This dated addendum supersedes older statements below that exclude a local viewer.

# Supply Chain Knowledge Graph — V1

Builds a reproducible supply-chain knowledge graph from unstructured text: retrieval, LLM
entity/relation extraction, entity disambiguation, controlled upstream expansion, and
idempotent Neo4j storage.

Two modes, and only two:

| Mode | Input | What it does |
|---|---|---|
| `INGEST` | documents you already have | normalize → deduplicate → Prompt 1 → Prompt 2 → facts → Neo4j `MERGE`. No search. |
| `EXPAND` | seed companies | persistent frontier → tier-aware queries → retrieval → source reputation (metadata) → Prompt 1 → Prompt 2 → ontology/evidence validation → facts → edge support → frontier from valid suppliesTo → Neo4j `MERGE` (optional) → next tier, up to `max_tier`. |

Project documents: [CLAUDE.md](CLAUDE.md) (development rules),
[docs/SupplyChain_KG_V1_Architecture_Brief_Final.pdf](docs/SupplyChain_KG_V1_Architecture_Brief_Final.pdf)
(architecture), [PROJECT_DECISIONS.md](PROJECT_DECISIONS.md) (frozen V1 scope),
[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) (layout),
[docs/EXTERNAL_DATASETS.md](docs/EXTERNAL_DATASETS.md) (validation data),
[docs/SOURCE_REPUTATION_AND_VERIFICATION.md](docs/SOURCE_REPUTATION_AND_VERIFICATION.md)
(source reputation, validation statuses, frontier gate),
[docs/SERVER_HANDOFF.md](docs/SERVER_HANDOFF.md) (server checklist).

---

# Local development and testing

This is what runs on a development machine with no GPU, no model server and no database.
All of it is verified.

## Install

Python 3.10–3.14.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Run the test suite

```bash
.venv/bin/python -m pytest tests -q
```

No network, model or database needed: a stub LLM answers both prompts, `httpx.MockTransport`
stands in for the retrieval APIs, and a recording driver captures the Cypher the writer
would send. The integration tests in `tests/test_integration.py` skip cleanly unless their
service is configured.

## Run the pipeline without external services

`--no-neo4j` stops after the facts Parquet layer, which exercises everything except the
graph write. It still needs a reachable model endpoint for the two prompts.

```bash
.venv/bin/python -m schain ingest --input docs/SupplyChain_KG_V1_Architecture_Brief_Final.pdf --no-neo4j
```

Configuration, argument parsing and the data layers can be checked with no services at all:

```bash
.venv/bin/python -m schain ingest --help
.venv/bin/python -m schain expand --help
```

---

# Server deployment — to be done later

**None of the steps in this section have been executed or verified.** The development
environment used for this audit ran offline tests only. The real model, database and
provider end-to-end run remain a later runtime validation step.

`scripts/server_run.sh` performs the intended steps in one command: install
dependencies, validate configuration, start or reuse Neo4j, start or reuse vLLM, wait for
both to be healthy, then run the pipeline and shut down only what it started.

```bash
git pull
./scripts/server_run.sh --smoke-test    # tiny real run first
./scripts/server_run.sh                 # then the full EXPAND
```

| Script | Purpose |
|---|---|
| `scripts/server_run.sh` | the one command above; `--help` documents the modes and flags |
| `scripts/server_preflight.sh` | read-only environment report (secrets shown as set/not set) |
| `scripts/server_status.sh` | run status, service health, checkpoint counts, recent log |

[docs/SERVER_HANDOFF.md](docs/SERVER_HANDOFF.md) is the checklist, and
[docs/SERVER_REQUIRED_INFO.md](docs/SERVER_REQUIRED_INFO.md) lists the few values that
cannot be auto-detected.

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `NEO4J_URI` | yes | e.g. `bolt://localhost:7687` |
| `NEO4J_USERNAME` | yes | e.g. `neo4j` |
| `NEO4J_PASSWORD` | yes | not needed only when running with `--no-neo4j` |
| `LLM_API_BASE_URL` | no | overrides `llm.base_url` from the config |
| `LLM_API_KEY` | no | required only for `llm.provider: api`; vLLM ignores its value |
| `SCHAIN_AUDIT_TRACE` | no | `1` writes `<graph-stem>.audit.json` beside the graph |
| `SCHAIN_AUDIT_FULL_CONTENT` | no | `1` includes source, chunks, exact model inputs and available responses |

The retrieval providers (Wikipedia, GDELT, Google News) need no credentials.

### Neo4j

Any Neo4j 5.x. **APOC is not required**: the graph writes are plain Cypher `MERGE`. The
pipeline creates the `Entity.canonical_id` uniqueness constraint it needs on startup.

```bash
docker run -d --name schain-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/CHANGE_ME -v "$PWD/neo4j_data:/data" neo4j:5
```

### Model endpoint

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --served-model-name Qwen/Qwen3-4B-Instruct-2507 --max-model-len 16384 --port 8000
```

Add `--tensor-parallel-size N` for multiple GPUs. Instruct/non-thinking behaviour needs no
server flag: the pipeline sends `chat_template_kwargs: {"enable_thinking": false}` with
every request, controlled by `llm.enable_thinking`.

---

# Commands

```bash
.venv/bin/python -m schain ingest --config config/ingest.yaml --input path/to/docs
```

Accepts a single `.txt`, `.md` or `.pdf` file, or a directory (searched recursively using
`ingest.file_patterns`).

```bash
.venv/bin/python -m schain expand --config config/expansion.yaml --max-tier 3
```

```bash
.venv/bin/python -m schain validate --config config/ingest.yaml
```

| Flag | Effect |
|---|---|
| `--no-neo4j` | stop after the facts table |
| `--restart` | clear checkpoint state and generated data, then run from scratch |
| `--max-documents N` | (`expand`) stop after N documents |
| `--log-level DEBUG` | more detail per step |

## Resume and restart

Resume is the default. Every expensive stage records its completion in
`data/state/state.db`, so nothing expensive runs twice:

| Document state | Meaning | A resumed run does |
|---|---|---|
| `PENDING` | registered only | Prompt 1 |
| `EXTRACTED` | Prompt 1 done, output on disk | Prompt 2 onwards — Prompt 1 is **not** repeated |
| `RESOLVED` | facts written to Parquet | the Neo4j write only — no LLM call at all |
| `COMMITTED` | merged into Neo4j | skips the document |
| `FAILED` | Prompt 1 never completed | retries Prompt 1 |

The `EXPAND` frontier and every search query are persisted too, so a crash mid-run does not
re-issue searches or re-expand finished companies.

`--restart` deletes the state database and the generated data layers. It deliberately does
**not** touch Neo4j: the writes are idempotent, so a rerun converges on the same graph.

## Configuration

| File | Contents |
|---|---|
| [config/base.yaml](config/base.yaml) | LLM, Neo4j, storage paths, runtime, concurrency. Included by both modes. |
| [config/ontology.yaml](config/ontology.yaml) | entity types, relation types, allowed patterns, which relations advance a tier, known aliases |
| [config/ingest.yaml](config/ingest.yaml) | input location, chunking |
| [config/expansion.yaml](config/expansion.yaml) | seeds, `max_tier`, sources, relevance filter, per-tier query templates |
| [config/deep_discovery.yaml](config/deep_discovery.yaml) | versioned high-recall upstream retrieval budgets for the approved Tier-5 resume |
| [docs/DEEP_DISCOVERY_SAFE_RESUME.md](docs/DEEP_DISCOVERY_SAFE_RESUME.md) | safe `bmw-supply-chain-v2` plan/apply/resume procedure and audit semantics |

Search limits resolve by **closest value wins**, field by field:

```text
global defaults  <  tier defaults  <  per-query override
```

`max_results` is a limit **per query**, not a cumulative document target. Secrets never
belong in these files.

`config/expansion.yaml` ships the four OEM seeds (Toyota, Tesla, BMW, BYD) so that `EXPAND`
has suppliers to discover. The battery and mining groups from the AlMahri selected-company
set are commented out directly below them; uncomment for strict replication.

## Data layout

Neo4j is the primary output but never the only copy, so the graph can be rebuilt without
repeating LLM calls. Everything under `data/` is gitignored.

```text
data/raw/documents.jsonl                    retrieved/ingested documents as received
data/processed/documents.parquet            normalized documents
data/processed/extractions/<doc_id>.json    Prompt 1 output per document
data/processed/facts.parquet                one row per extracted fact, with evidence
data/processed/entity_resolution.parquet    Prompt 2 results
data/processed/search_results.parquet       retrieval results per query
data/state/state.db                         checkpoint/resume state
data/external/                              externally obtained datasets
outputs/supply_chain_graph.v1.json          derived frontend-ready semantic graph
```

## Frontend-ready graph export

After a successful `ingest` or `expand` run, SChain writes a **semantic** graph bundle to
`outputs/supply_chain_graph.v1.json` (configurable via `storage.graph_export`). The file is
plain UTF-8 JSON: nodes, aggregated edges, evidence, document provenance, ontology, and
statistics. It is **not** a visualization — no coordinates, colours, HTML, or layout.

Regenerate without repeating LLM calls:

```bash
.venv/bin/python -m schain export --config config/ingest.yaml
.venv/bin/python -m schain export --config config/expansion.yaml --output outputs/demo_graph.json
```

A separate frontend may consume this bundle for 2D/3D rendering, filters, and evidence
panels. The Parquet / SQLite layers remain the source of truth; the JSON is derived.

## Expansion depth

Tier is **graph distance along `supplier --suppliesTo--> customer`**, never industrial
role: a mining company selling directly to an OEM is Tier 1, not a deep tier. Only
`suppliesTo` advances depth; `locatedIn`, `owns` and `produces` describe an entity without
adding any.

The current experiment has four active OEM seeds and `max_tier: 5`. Tiers 0–4
are searched; Tier 5 is terminal. Five directional query families are shared across
searchable tiers. Runtime and document budgets are cumulative across resumes, with
150 unique selected documents and 110 minutes as the configured safety ceilings.
A quality-ranked beam and reserved deeper-tier targets control breadth. See
[the audit policy](docs/AUDIT_V0.3.0.md) for exact tier allocations and limitations.

## Retrieval sources

Four providers, all feeding **one** pipeline — there is no separate handling per source:

| Family | Providers | Per-query candidate ceiling |
|---|---|---|
| News | Google News (RSS search), GDELT (DOC 2.0 API) | 12 per provider at Tiers 0–4 |
| General web | `general_web` — company supplier/sourcing pages, press releases, investor relations, annual and sustainability reports, filings | 12 at Tiers 0–4 |
| Reference | Wikipedia (MediaWiki API) | 5 at Tiers 0–4 |

News carries new supply agreements; the general web carries the older, more stable
sourcing information. `general_web` defaults to a DuckDuckGo backend needing no
credentials, and uses the official Brave Search API instead when `BRAVE_SEARCH_API_KEY` is
set. It is optional: if it fails or is disabled, the other providers are unaffected.

**Wikipedia has no special role and no "background mode".** It is a search provider like
the others: a generated query such as `"BMW" suppliers` goes to its search endpoint and the
matching articles come back. The pipeline never assumes the entity's own page is the
document to fetch.

Limits are ceilings on candidates **per provider, per query** — not a target number of
relevant documents, and there is no cumulative target mechanism. Every candidate then goes
through the same path: URL dedup → cheap title/snippet relevance check → full-text fetch
where possible → content dedup → Prompt 1.

**Known limitation.** Google News RSS returns redirect/wrapper links, so an article body
often cannot be extracted. When that happens the document falls back to title + snippet
rather than being dropped, and each document records which it was:

```text
content_source = full_text | title_snippet | file
```

visible in `data/processed/documents.parquet` and in the search-results layer. The pipeline
is designed to work either way; extraction from a headline simply yields fewer facts. V1
does not attempt to defeat the redirects — no scraping stack, no browser automation.

## What one document may contribute

There is no per-document cap on facts. Prompt 1 extracts everything the usable text
evidences within the static ontology, so one document may yield two entities and one
relation, or a dozen entities and several relations. Long documents are split by the
existing chunking; after chunks are combined, repeated entities and repeated facts collapse
(a fact id is derived from document plus canonical endpoints plus relation), and evidence
is retained.

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
monitoring, no stale-edge checks, no existing-edge revalidation. If an older document says
CATL supplies BMW and a newer one reports the agreement ended, V1 records both as evidence
and changes nothing structurally. That belongs to a future monitoring version.

## Retrieval audit manifest

Each `EXPAND` run writes `logs/retrieval_manifest_<run_id>.csv` — one row per candidate a
provider returned, including candidates that were deduplicated, judged irrelevant, failed
to fetch, or never reached the LLM. Columns cover the query and provider that produced it,
title, URL, domain, publication and retrieval time, snippet and content lengths (characters
and words), `fetch_status`, `http_status`, `content_mode`, `dedup_status`,
`relevance_status`, `llm_status` and a short `error` code. Article text is not copied into
it.

It is for audit and debugging only — SQLite and Parquet remain the source of truth for
resume, and rows are keyed so a resumed run updates them instead of appending duplicates. A
compact summary (raw candidates, unique URLs, fetch outcomes, full-text vs snippet-only,
relevance, how many reached Prompt 1, with a per-provider and per-tier breakdown) is printed
at the end of every run and derived from those rows.

## Inspecting the graph

Use Neo4j Browser at http://localhost:7474. There is no custom dashboard in V1.

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
  `PROJECT_DECISIONS.md` §9 requires the opposite: canonical ids plus `MERGE`, so a
  duplicate is never created. One small module of plain Cypher.
- **Custom Prompt 2.** The library ships exact/fuzzy/spaCy resolvers, not an LLM
  disambiguator. Four stages run cheapest-first: the persisted resolution cache,
  deterministic normalization (case, punctuation, accents, legal suffixes), the known-alias
  map in `config/ontology.yaml`, and only then one LLM call per entity type.
- **Stable canonical ids.** A group's canonical name is the longest mention in it, not
  whatever the model calls it, and a group reaching an already-known entity reuses that
  entity's id. Repeated runs therefore cannot produce duplicate nodes.
- **Two independent JSON controls.** `llm.json_mode` asks the server for guaranteed
  parseable JSON and applies to both prompts. `llm.structured_output` additionally
  constrains Prompt 1 with the full JSON schema, which needs a grammar backend on the server
  and is off by default. Prompt 2 is a plain call, so it relies on `json_mode` regardless.
- **No lexical graph in Neo4j.** `create_lexical_graph=False`, so Neo4j holds only the
  static ontology; the evidence trail lives in the facts Parquet layer.

## Incremental Save and persistent Colab EXPAND

Neo4j persistence is explicit: pass `--save` to merge each document's accepted,
validated facts as it completes, or `--no-save`/legacy `--no-neo4j` for a completely
Neo4j-free run. Local SQLite/Parquet checkpoints and Graph JSON export remain active
in either mode.

For restart-safe Colab expansion, use
[`notebooks/SChain_Expand_Persistent_Save.ipynb`](notebooks/SChain_Expand_Persistent_Save.ipynb).
It keeps live SQLite on local Colab storage and publishes verified backup generations
to Google Drive. Configuration, security, resume, and crash-recovery details are in
[`docs/SAVE_AND_COLAB_RESUME.md`](docs/SAVE_AND_COLAB_RESUME.md).

## Distributed Colab EXPAND

Multiple independent Colabs can join one PostgreSQL-authoritative experiment, claim
fenced company leases, share durable extraction/entity checkpoints, and deliver accepted
facts through a transactional Neo4j outbox. SQLite remains the default local backend.
See [`docs/DISTRIBUTED_NEON_EXPAND.md`](docs/DISTRIBUTED_NEON_EXPAND.md) and
[`notebooks/SChain_Expand_Neon_Distributed.ipynb`](notebooks/SChain_Expand_Neon_Distributed.ipynb).
