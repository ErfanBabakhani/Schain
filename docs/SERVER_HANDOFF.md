# Server handoff

For moving V1 to the GPU server. **None of this has been executed yet** — the development
laptop cannot serve Qwen3.8-27B (4 GB VRAM) and has no Neo4j, so the real model, the real
database and any end-to-end run belong to the server phase.

## First run

```bash
git clone git@github.com:ErfanBabakhani/SChain.git    # or: git pull
cd SChain
cp .env.example .env          # then set NEO4J_PASSWORD; see below for the rest
./scripts/server_run.sh --smoke-test
```

The smoke test is a tiny real run — one tier, three documents, the normal pipeline code,
the real LLM and the real Neo4j. If it passes, do the full run:

```bash
./scripts/server_run.sh
```

Everything else is automatic: the script creates `.venv` and installs the project if
needed, reads the model and endpoint from `config/base.yaml`, starts Neo4j and vLLM if
they are not already up, waits for both to be healthy, and then launches
`python -m schain expand`.

### What you must set in `.env`

Only `NEO4J_PASSWORD`, unless one of these applies:

- the cluster uses SLURM and no GPU is visible in your shell → `SCHAIN_SLURM_PARTITION`
  (and possibly `SCHAIN_SLURM_ACCOUNT`);
- the model weights are gated → `HF_TOKEN`;
- you want the weights cached on a shared filesystem → `HF_HOME`.

The script tells you exactly which value is missing and how to discover it. See
[SERVER_REQUIRED_INFO.md](SERVER_REQUIRED_INFO.md). Every other `SCHAIN_*` setting in
`.env.example` is optional and has a working default.

If `.env` already exists, leave it alone — `cp` would overwrite it.

## Normal later run

```bash
git pull
./scripts/server_run.sh
```

Resume is the default, so this continues from the SQLite checkpoint rather than starting
over. Re-running after any interruption is always safe.

## Commands

| Task | Command |
|---|---|
| Full EXPAND run | `./scripts/server_run.sh` |
| INGEST documents | `./scripts/server_run.sh ingest path/to/docs` |
| Smoke test first | `./scripts/server_run.sh --smoke-test` |
| Status | `./scripts/server_status.sh` |
| Environment report only | `./scripts/server_preflight.sh` |
| Attach to a running run | `tmux attach -t schain` |
| Detach again | `Ctrl-b` then `d` |
| Follow the log instead | `tail -f logs/expand_*.log` |
| Stop a run | `tmux kill-session -t schain` |
| See the plan without doing anything | `./scripts/server_run.sh --dry-run` |
| Deliberate fresh start | `./scripts/server_run.sh --restart` |

`--restart` clears the checkpoint state and the generated data layers under `data/`. It
does **not** touch Neo4j: the writes are idempotent `MERGE`, so a rerun converges on the
same graph. Nothing else ever deletes state.

## What survives what

| Situation | Result |
|---|---|
| SSH disconnects | The run continues: it lives in a tmux session, not your shell. Reconnect and `tmux attach -t schain`. |
| The pipeline crashes | Owned services are shut down and the checkpoint is intact. Re-run `./scripts/server_run.sh` to resume. |
| The server reboots | tmux is gone, `data/state/state.db` is not. Re-run `./scripts/server_run.sh` to resume. |
| The run finishes | Services this run started are stopped; the GPU is released. |

## What gets shut down, and what does not

The script tracks what it started:

- **Started by the run** → stopped when the workload finishes: vLLM (so a 27B model does
  not sit on the GPU), and the Neo4j container if the script created or started it. Under
  SLURM, the script exits, `srun` returns, and the allocation is released — no idle job is
  left behind.
- **Already running beforehand** → reused and never stopped. If Neo4j or an OpenAI-compatible
  endpoint is already up, the script says so and leaves it alone.
- **Never touched** → `data/state/state.db`, the Parquet layers, the Neo4j data directory,
  and the logs.

## On this cluster (Neumann)

Verified against the real scheduler. Values stay configurable in `.env`:

```
SCHAIN_SLURM_PARTITION=ampere
SCHAIN_SLURM_ACCOUNT=ampere
SCHAIN_SLURM_GRES=gpu:a100_80gb:1
```

| | |
|---|---|
| Smoke test | interactive `srun`, capped at 2h — the cluster allows this for debugging |
| Production | `sbatch` batch job, default `SCHAIN_SLURM_TIME=24:00:00` |

Production must not use an interactive allocation: those are capped at two hours here
however long `--time` asks for. `./scripts/server_run.sh` submits the batch job for you,
prints the job id and log path, and returns your terminal. No tmux is used on SLURM —
the batch job is the detached process. Stop it with `scancel <job id>`, which
`./scripts/server_status.sh` prints for you.

Neither Docker nor root is available here, so when `NEO4J_URI` is not reachable the runner
installs **Neo4j Community from the official tarball** into `.runtime/neo4j` (gitignored),
on the compute node beside the pipeline — `localhost` on the login node is not the compute
node's `localhost`. Graph data and logs live under `.runtime/neo4j/data` and `logs`, never
in `/tmp`, and survive the run. Java 21+ must be available; the runner fails clearly if not.
If `NEO4J_PASSWORD` is unset on the first local install, one is generated and written to
`.env` (chmod 600) and never printed.

vLLM is not preinstalled here. On the first run with a GPU present, the runner installs it
once into the existing `.venv`, then reuses it. The first model load also downloads
Qwen3.8-27B from Hugging Face — the log says so explicitly, so a long first start is not
mistaken for a hang. Set `HF_HOME` to cache on shared storage.

## Notes

- **APOC is not required** by the current implementation; every write is plain Cypher
  `MERGE`. The container the script starts does not install it.
- **vLLM is not installed automatically.** It is large and GPU-specific. On the server:
  `.venv/bin/python -m pip install vllm`. The script says so if it is missing.
- Logs land in `logs/`, timestamped, e.g. `logs/expand_2026-08-29_210000.log`. vLLM gets
  its own `logs/vllm_*.log`. The path is printed when the run starts.
- Each EXPAND run also writes `logs/retrieval_manifest_<run_id>.csv`: one row per retrieved
  candidate, with fetch, dedup, relevance and Prompt-1 status. Use it to see why a document
  did or did not become graph facts. A summary is printed at the end of the run.
- Production depth is `max_tier: 6`, so a full run is large. The smoke test deliberately
  uses `--max-tier 0 --max-documents 3`; raise the depth only after it passes.
- **Start small.** The smoke test exists for this. Only scale up — full `max_tier`, no
  `--max-documents` — once it has passed once.
- Some documents will be title + snippet rather than full articles, because Google News
  RSS links are redirect wrappers. Each document records which it was in
  `content_source`.

## Verification order on the first day

1. `./scripts/server_preflight.sh` — check what the machine offers before anything runs.
2. `.venv/bin/python -m pytest tests -q` — the offline suite should pass unchanged.
3. `NEO4J_PASSWORD=... .venv/bin/python -m pytest tests/test_integration.py -v` — proves
   `MERGE` is idempotent against the real database.
4. `SCHAIN_RUN_LLM_TESTS=1 .venv/bin/python -m pytest tests/test_integration.py -v` —
   proves both prompts behave against the real model.
5. `./scripts/server_run.sh --smoke-test`.
6. Inspect the graph in Neo4j Browser:
   ```cypher
   MATCH (s:Company)-[r:suppliesTo]->(c:Company) RETURN s.name, c.name, r.evidence LIMIT 25;
   ```
7. `./scripts/server_run.sh` for the full run, then `./scripts/server_status.sh` to watch it.
