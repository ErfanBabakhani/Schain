# Supply Chain Knowledge Graph — V1

Builds a reproducible supply-chain knowledge graph from unstructured text: retrieval,
LLM entity/relation extraction, entity disambiguation, controlled upstream expansion, and
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
[docs/EXTERNAL_DATASETS.md](docs/EXTERNAL_DATASETS.md) (validation data).

---

## Server setup

### 1. Clone and install

Python 3.10–3.14.

```bash
git clone git@github.com:ErfanBabakhani/SChain.git
cd SChain
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

### 2. Configure `.env`

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
|---|---|---|
| `NEO4J_URI` | yes | e.g. `bolt://localhost:7687` |
| `NEO4J_USERNAME` | yes | e.g. `neo4j` |
| `NEO4J_PASSWORD` | yes | not needed only when running with `--no-neo4j` |
| `LLM_API_BASE_URL` | no | overrides `llm.base_url` from the config |
| `LLM_API_KEY` | no | required only for `llm.provider: api`; vLLM ignores its value |

The retrieval providers (Wikipedia, GDELT, Google News) need no credentials.

### 3. Start Neo4j

Any Neo4j 5.x instance works. **APOC is not required**: the graph writes are plain Cypher
`MERGE` statements. Docker example:

```bash
docker run -d --name schain-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/CHANGE_ME -v "$PWD/neo4j_data:/data" neo4j:5
```

The pipeline creates the uniqueness constraint it needs (`Entity.canonical_id`) on startup.

### 4. Start the model endpoint

Any OpenAI-compatible server works. vLLM serving the configured Qwen model:

```bash
vllm serve Qwen/Qwen3.8-27B --served-model-name Qwen/Qwen3.8-27B --max-model-len 16384 --port 8000
```

Add `--tensor-parallel-size N` for multiple GPUs. Instruct/non-thinking behaviour needs no
server flag: the pipeline sends `chat_template_kwargs: {"enable_thinking": false}` with
every request, controlled by `llm.enable_thinking` in [config/base.yaml](config/base.yaml).
Point `llm.base_url` at whatever host serves it.

### 5. Run

```bash
.venv/bin/python -m schain ingest --config config/ingest.yaml --input data/corpus
```

```bash
.venv/bin/python -m schain expand --config config/expansion.yaml
```

---

## Commands

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

Shared flags:

| Flag | Effect |
|---|---|
| `--no-neo4j` | stop after the facts Parquet layer; useful before a database exists |
| `--restart` | clear checkpoint state and generated data, then run from scratch |
| `--max-documents N` | (`expand`) stop after N documents |
| `--log-level DEBUG` | more detail per step |

A first smoke test needing neither Neo4j nor a corpus of your own — the architecture brief
is itself a PDF:

```bash
.venv/bin/python -m schain ingest --input docs/SupplyChain_KG_V1_Architecture_Brief_Final.pdf --no-neo4j
```

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
| [config/ontology.yaml](config/ontology.yaml) | entity types, relation types, allowed patterns, which relations advance a tier |
| [config/ingest.yaml](config/ingest.yaml) | input location, chunking |
| [config/expansion.yaml](config/expansion.yaml) | seeds, `max_tier`, sources, relevance filter, per-tier query templates |

Search limits resolve by **closest value wins**, field by field:

```text
global defaults  <  tier defaults  <  per-query override
```

`max_results` is a limit **per query**, not a cumulative document target. Secrets never
belong in these files.

### Seeds

`config/expansion.yaml` ships the four OEM seeds (Toyota, Tesla, BMW, BYD) so that
`EXPAND` actually has suppliers to discover. The battery and mining groups from the
AlMahri selected-company set are commented out directly below them; uncomment for strict
selected-company replication.

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

## Tests

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest tests -q
```

No network, model or database is needed: a stub LLM stands in for both prompts. The
integration tests in `tests/test_integration.py` skip cleanly unless their service is
configured — set `NEO4J_PASSWORD` for the Neo4j checks, `SCHAIN_RUN_LLM_TESTS=1` for the
model checks, and `SCHAIN_RUN_NETWORK_TESTS=1` for the live retrieval checks.

## Design notes

- **Reused from neo4j-graphrag 1.19.0**: `FixedSizeSplitter`, `LLMEntityRelationExtractor`
  (Prompt 1), `GraphSchema`, `PdfLoader`, `OpenAILLM`, and the official Neo4j driver.
- **Custom Neo4j writer.** The library's `Neo4jWriter` issues `CREATE` and relies on a
  later resolver pass (`apoc.refactor.mergeNodes`) to collapse duplicates.
  `PROJECT_DECISIONS.md` §9 requires the opposite: canonical ids plus `MERGE`, so a
  duplicate is never created. The writer is one small module of plain Cypher.
- **Custom Prompt 2.** The library ships exact/fuzzy/spaCy resolvers, not an LLM
  disambiguator. Deterministic normalization runs first, the LLM only sees what is left,
  and results are cached in the state database so the same resolution is never paid for
  twice.
- **Stable canonical ids.** A group's canonical name is the longest mention in it, not
  whatever the model calls it, and a group that reaches an already-known entity reuses that
  entity's id. Repeated runs therefore cannot produce duplicate nodes.
- **No lexical graph in Neo4j.** `create_lexical_graph=False`, so Neo4j holds only the
  static ontology; the evidence trail lives in the facts Parquet layer.
