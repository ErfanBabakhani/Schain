# SChain — Supply-Chain Knowledge Graph

SChain builds a supply-chain knowledge graph from unstructured text. Starting from seed
companies, or from documents you already have, it retrieves sources, extracts entities
and relations with an LLM, resolves name variants to canonical entities, validates every
fact against its source text, and expands upstream tier by tier. Results are kept as
Parquet/SQLite checkpoints and a graph JSON export, and optionally merged into Neo4j.

```text
 seeds ─► search ─► relevance filter ─► extract (Prompt 1) ─► resolve (Prompt 2) ─► validate ─► facts
   ▲                                                                                              │
   └──────────── suppliers found through suppliesTo join the next tier, up to max_tier ◄──────────┘
```

Every expensive step is checkpointed, so an interrupted run resumes where it stopped and
never pays for the same search or LLM call twice.

## Quick start

Python 3.10–3.14.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
cp .env.example .env
```

Run the test suite. It needs no network, model or database:

```bash
.venv/bin/python -m pytest -q
```

Extraction needs an OpenAI-compatible model endpoint. By default that is vLLM at
`http://localhost:8000/v1` serving `Qwen/Qwen3-4B-Instruct-2507`:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 16384 --port 8000
```

Then check the whole pipeline end to end in a few minutes (one seed, three documents,
results in `data/smoke/` and `outputs/smoke_graph.json`):

```bash
.venv/bin/python -m schain expand --config config/smoke.yaml
```

On a GPU server, `./scripts/server_run.sh smoke` does all of this in one command (see
[Running on a GPU server](#running-on-a-gpu-server)).

## Run modes

| Mode | Command | Use it for |
|---|---|---|
| **Smoke check** | `python -m schain expand --config config/smoke.yaml` | verifying a setup end to end in minutes |
| **Ingest** | `python -m schain ingest --input path/to/docs` | `.txt`, `.md` and `.pdf` files you already have |
| **Expand** | `python -m schain expand` | the main experiment: discover suppliers of the configured seeds |
| **Deep discovery** | `python -m schain expand --config config/deep_discovery.yaml` | high-recall expansion with larger budgets |
| **Colab** | `python -m schain expand --config config/colab_professor_expand.yaml --save` | one Colab runtime, saving to Neo4j |
| **Distributed** | `python -m schain expand --config config/distributed_neon_expand.yaml` | several workers sharing one PostgreSQL experiment |

The notebooks in [`notebooks/`](notebooks) run Ingest and distributed deep discovery on
Google Colab. Resume is the default: rerun the same command to continue.

| Flag | Effect |
|---|---|
| `--save` / `--no-save` | merge accepted facts into Neo4j as each document completes, or not (default: the config's `save.enabled`, off) |
| `--restart` | clear this experiment's checkpoints and generated data, then start fresh (Neo4j is never touched) |
| `--max-tier N` · `--max-documents N` | override depth or the per-run document ceiling (expand) |
| `--seed NAME` | explicit seed override, repeatable (expand) |
| `--log-level DEBUG` | more detail per step |

Commands that need no model:

```bash
python -m schain export                       # rebuild the graph JSON from stored facts
python -m schain view --input outputs/supply_chain_graph.v1.json --output outputs/graph.html
python -m schain validate                     # overlap with an external supplier dataset
python -m schain evaluate --gold gold.json    # precision/recall against a labelled set
```

## Running on a GPU server

[`scripts/server_run.sh`](scripts/server_run.sh) runs one mode end to end. It creates
`.venv`, installs vLLM on first use, starts the model on a GPU (through SLURM when a
partition is configured), starts Neo4j only for `--save` runs, runs the pipeline, and
then stops whatever it started.

```bash
./scripts/server_run.sh smoke              # first: a few-minute check on the real services
./scripts/server_run.sh                    # EXPAND (tmux, or a SLURM batch job)
./scripts/server_run.sh deep               # deep-discovery profile
./scripts/server_run.sh ingest data/corpus # INGEST
./scripts/server_status.sh                 # progress, service health, recent log
```

Server settings are optional `SCHAIN_*` variables in `.env`
([`.env.example`](.env.example) lists them all). The ones most often needed:

| Variable | Purpose |
|---|---|
| `SCHAIN_SLURM_PARTITION` | run the model through SLURM on this partition |
| `SCHAIN_SLURM_GRES` | GPU request (default `gpu:1`) |
| `SCHAIN_VLLM_GPU_MEMORY_UTILIZATION` | share of GPU memory vLLM may take (default 0.90; lower it on a shared GPU) |
| `SCHAIN_VLLM_MAX_MODEL_LEN` | context length (default 16384) |
| `SCHAIN_VLLM_EXTRA_ARGS` | extra `vllm serve` flags, e.g. `--dtype half` on pre-Ampere GPUs |
| `SCHAIN_VLLM_PACKAGE` | pin vLLM when the newest release needs a newer NVIDIA driver than the server has (e.g. `vllm==0.19.1` for a CUDA 12 driver) |
| `HF_HOME` | Hugging Face model cache |

If the home directory is network storage, keep the repository (and so `.venv`) on a
local disk: a vLLM environment is tens of thousands of small files. The checklist is in
[docs/SERVER_HANDOFF.md](docs/SERVER_HANDOFF.md).

## Configuration

Configs layer through `include:`, and each file states only what differs from the one it
includes:

```text
base.yaml ─┬─ ingest.yaml
           └─ expansion.yaml ─┬─ smoke.yaml
                              ├─ deep_discovery.yaml ── distributed_neon_deep_discovery.yaml
                              ├─ distributed_neon_expand.yaml
                              └─ colab_professor_expand.yaml
```

The model and storage live in `base.yaml`; seeds, depth and retrieval live in
`expansion.yaml`. [config/README.md](config/README.md) explains which file to edit and
how to write a small overlay of your own.

Secrets and machine-specific values come from `.env`, never from the config files:

| Variable | Needed for |
|---|---|
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` | `--save` runs (any Neo4j 5.x; APOC not required) |
| `LLM_API_BASE_URL`, `LLM_API_KEY` | another endpoint, or `llm.provider: api` |
| `NEON_DATABASE_URL` | distributed runs |
| `BRAVE_SEARCH_API_KEY` | optional: the Brave backend for general-web search |
| `SCHAIN_AUDIT_TRACE`, `SCHAIN_AUDIT_FULL_CONTENT` | optional: a per-run audit JSON beside the graph |

## Outputs

Everything below is gitignored. Neo4j is never the only copy: the graph JSON and Neo4j
can both be rebuilt from the facts layer without repeating any LLM call.

```text
outputs/supply_chain_graph.v1.json         the graph: nodes, aggregated edges, evidence, provenance
data/processed/facts.parquet               one row per extracted fact, with its evidence
data/processed/extractions/<doc_id>.json   Prompt 1 output per document
data/processed/entity_resolution.parquet   Prompt 2 results
data/processed/documents.parquet           normalized documents
data/processed/search_results.parquet      retrieval results per query
data/state/state.db                        checkpoint and resume state
logs/retrieval_manifest_<run_id>.csv       every retrieval candidate and what happened to it
```

To explore a graph, open [`examples/graph-explorer.html`](examples/graph-explorer.html) in
a browser and load a graph JSON, or build a standalone page with `python -m schain view`.

## Documentation

| Document | Contents |
|---|---|
| [docs/PIPELINE.md](docs/PIPELINE.md) | how the pipeline works: tiers, sources, graph semantics, resume |
| [config/README.md](config/README.md) | configuration layering and what to edit |
| [docs/PROJECT_DECISIONS.md](docs/PROJECT_DECISIONS.md) | the frozen V1 scope |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | repository layout |
| [Architecture brief (PDF)](docs/SupplyChain_KG_V1_Architecture_Brief_Final.pdf) | the V1 architecture |
| [docs/SERVER_HANDOFF.md](docs/SERVER_HANDOFF.md) · [SERVER_REQUIRED_INFO.md](docs/SERVER_REQUIRED_INFO.md) | GPU server checklist |
| [docs/SAVE_AND_COLAB_RESUME.md](docs/SAVE_AND_COLAB_RESUME.md) | incremental Neo4j save and Colab resume |
| [docs/DISTRIBUTED_NEON_EXPAND.md](docs/DISTRIBUTED_NEON_EXPAND.md) · [DEEP_DISCOVERY_SAFE_RESUME.md](docs/DEEP_DISCOVERY_SAFE_RESUME.md) | distributed runs |
| [docs/SOURCE_REPUTATION_AND_VERIFICATION.md](docs/SOURCE_REPUTATION_AND_VERIFICATION.md) | source reputation, validation statuses, frontier gate |
| [docs/EXTERNAL_DATASETS.md](docs/EXTERNAL_DATASETS.md) | external validation datasets |
| [docs/AUDIT_TRACE_JSON.md](docs/AUDIT_TRACE_JSON.md) | the optional audit trace |
| [docs/AUDIT_V0.3.0.md](docs/AUDIT_V0.3.0.md) · [UPGRADE_REVIEW.md](docs/UPGRADE_REVIEW.md) · [RELEASE_NOTES_0.3.0.md](docs/RELEASE_NOTES_0.3.0.md) | audits and release history |
