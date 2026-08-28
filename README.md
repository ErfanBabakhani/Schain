# Supply Chain Knowledge Graph — V1

Builds a reproducible supply-chain knowledge graph from unstructured text: retrieval, LLM
entity/relation extraction, entity disambiguation, controlled upstream expansion, and
idempotent Neo4j storage.

Two modes, and only two:

| Mode | Input | What it does |
|---|---|---|
| `INGEST` | documents you already have | normalize → deduplicate → Prompt 1 → Prompt 2 → facts → Neo4j `MERGE`. No search. |
| `EXPAND` | seed companies | persistent frontier → tier-aware queries → retrieval → dedup/filter → Prompt 1 → Prompt 2 → facts → Neo4j `MERGE` → discover upstream suppliers → next tier, up to `max_tier`. |

Project documents: [CLAUDE.md](CLAUDE.md) (development rules),
[docs/SupplyChain_KG_V1_Architecture_Brief_Final.pdf](docs/SupplyChain_KG_V1_Architecture_Brief_Final.pdf)
(architecture), [PROJECT_DECISIONS.md](PROJECT_DECISIONS.md) (frozen V1 scope),
[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) (layout),
[docs/EXTERNAL_DATASETS.md](docs/EXTERNAL_DATASETS.md) (validation data),
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
laptop cannot serve Qwen3.8-27B (4 GB VRAM against roughly 54 GB needed in bf16) and has no
Neo4j instance, so the real model, the real database and any end-to-end run belong to the
server phase.

The intended future steps, in order:

1. install dependencies (as above);
2. configure `.env`;
3. start or connect Neo4j;
4. start vLLM with Qwen3.8-27B and point `llm.base_url` at it;
5. run a small real INGEST;
6. run a small real EXPAND.

[docs/SERVER_HANDOFF.md](docs/SERVER_HANDOFF.md) is the checklist for doing this, including
the exact commands, the integration-test flags, and the first verification steps.

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `NEO4J_URI` | yes | e.g. `bolt://localhost:7687` |
| `NEO4J_USERNAME` | yes | e.g. `neo4j` |
| `NEO4J_PASSWORD` | yes | not needed only when running with `--no-neo4j` |
| `LLM_API_BASE_URL` | no | overrides `llm.base_url` from the config |
| `LLM_API_KEY` | no | required only for `llm.provider: api`; vLLM ignores its value |

The retrieval providers (Wikipedia, GDELT, Google News) need no credentials.

### Neo4j

Any Neo4j 5.x. **APOC is not required**: the graph writes are plain Cypher `MERGE`. The
pipeline creates the `Entity.canonical_id` uniqueness constraint it needs on startup.

```bash
docker run -d --name schain-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/CHANGE_ME -v "$PWD/neo4j_data:/data" neo4j:5
```

### Model endpoint

```bash
vllm serve Qwen/Qwen3.8-27B --served-model-name Qwen/Qwen3.8-27B --max-model-len 16384 --port 8000
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
```

## Retrieval sources

Wikipedia (MediaWiki API), GDELT (DOC 2.0 API) and Google News (RSS search). None needs
credentials. Wikipedia is a search provider like the others: a generated query such as
`"BMW" suppliers` goes to the search endpoint and the matching articles come back — the
pipeline never assumes the entity's own page is the document to fetch.

**Known limitation.** Google News RSS returns redirect/wrapper links, so an article body
often cannot be extracted. When that happens the document falls back to title + snippet
rather than being dropped, and each document records which it was:

```text
content_source = full_text | title_snippet | file
```

visible in `data/processed/documents.parquet` and in the search-results layer. The pipeline
is designed to work either way; extraction from a headline simply yields fewer facts. V1
does not attempt to defeat the redirects — no scraping stack, no browser automation.

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
