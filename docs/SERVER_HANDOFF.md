# Server handoff checklist

For moving V1 to the GPU server. **None of the steps below have been executed yet** — the
laptop cannot run Qwen3.8-27B (4 GB VRAM) and has no Neo4j instance, so everything in this
file is still to be done.

## 1. Clone

```bash
git clone git@github.com:ErfanBabakhani/SChain.git
cd SChain
```

## 2. Python environment

Python 3.10–3.14.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## 3. Environment variables

```bash
cp .env.example .env
```

| Variable | Required | Notes |
|---|---|---|
| `NEO4J_URI` | yes | e.g. `bolt://localhost:7687` |
| `NEO4J_USERNAME` | yes | e.g. `neo4j` |
| `NEO4J_PASSWORD` | yes | omit only when running with `--no-neo4j` |
| `LLM_API_BASE_URL` | no | overrides `llm.base_url` in `config/base.yaml` |
| `LLM_API_KEY` | no | only for `llm.provider: api`; vLLM ignores the value |

The retrieval providers (Wikipedia, GDELT, Google News) need no credentials.

## 4. Neo4j

Any Neo4j 5.x. **APOC is not required by the current implementation** — every write is
plain Cypher `MERGE`, one statement per ontology label and relation type. The pipeline
creates its own `Entity.canonical_id` uniqueness constraint on startup.

```bash
docker run -d --name schain-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/CHANGE_ME -v "$PWD/neo4j_data:/data" neo4j:5
```

## 5. vLLM with Qwen3.8-27B

```bash
vllm serve Qwen/Qwen3.8-27B --served-model-name Qwen/Qwen3.8-27B --max-model-len 16384 --port 8000
```

Add `--tensor-parallel-size N` for multiple GPUs. Point `llm.base_url` in
`config/base.yaml` (or `LLM_API_BASE_URL`) at the host serving it. Instruct/non-thinking
behaviour needs no server flag: the pipeline sends
`chat_template_kwargs: {"enable_thinking": false}` with every request.

## 6. Smoke tests — start very small

Do these before any large run. A first run on a big corpus or a full expansion will spend
a lot of GPU time before any mistake becomes visible.

INGEST, one small document (the architecture brief is a convenient real PDF):

```bash
.venv/bin/python -m schain ingest --config config/ingest.yaml --input docs/SupplyChain_KG_V1_Architecture_Brief_Final.pdf
```

EXPAND, deliberately tiny — one tier, a handful of documents:

```bash
.venv/bin/python -m schain expand --config config/expansion.yaml --max-tier 0 --max-documents 3
```

Only after both look right should you raise `--max-tier` toward the default of 3 and remove
`--max-documents`.

## 7. Integration tests

They skip unless their service is configured, so run them explicitly once the stack is up:

```bash
NEO4J_PASSWORD=... SCHAIN_RUN_LLM_TESTS=1 SCHAIN_RUN_NETWORK_TESTS=1 .venv/bin/python -m pytest tests/test_integration.py -v
```

| Flag / variable | Enables |
|---|---|
| `NEO4J_PASSWORD` | live `MERGE` idempotency check against the database |
| `SCHAIN_RUN_LLM_TESTS=1` | Prompt 1 and Prompt 2 against the real model |
| `SCHAIN_RUN_NETWORK_TESTS=1` | live queries to the public retrieval APIs |

## 8. First verification steps, in order

1. `.venv/bin/python -m pytest tests -q` — the offline suite should pass unchanged.
2. `python -m schain ingest --help` and `--no-neo4j` on one document — confirms config and
   the data layers before any external service is involved.
3. Bring up Neo4j, then run `tests/test_integration.py` with `NEO4J_PASSWORD` set — proves
   `MERGE` is idempotent on the real database.
4. Bring up vLLM, then re-run with `SCHAIN_RUN_LLM_TESTS=1` — proves both prompts behave
   against the real model, and that non-thinking mode is accepted.
5. Run the small INGEST smoke test end to end, and inspect the graph in Neo4j Browser:
   ```cypher
   MATCH (s:Company)-[r:suppliesTo]->(c:Company) RETURN s.name, c.name, r.evidence LIMIT 25;
   ```
6. Run the tiny EXPAND smoke test, then check the frontier advanced by tier:
   ```bash
   sqlite3 data/state/state.db "SELECT tier, status, COUNT(*) FROM frontier GROUP BY tier, status;"
   ```
7. Re-run the same command once more: it should do almost nothing (resume is the default)
   and must not create duplicate nodes.
8. Only then scale up.

## 9. Things to expect

- Google News RSS links are redirect wrappers, so some documents will be title + snippet
  rather than full article bodies. This is recorded per document as
  `content_source = title_snippet` in `data/processed/documents.parquet`, and the pipeline
  is designed to work either way.
- `llm.structured_output` is off by default. vLLM has a grammar backend, so it can be
  turned on in `config/base.yaml` if Prompt 1 output needs tighter constraint.
- Concurrency defaults (`concurrency.retrieval: 8`, `llm_max_concurrency: 4`) are untuned
  guesses; tuning them belongs on the real GPU server.
