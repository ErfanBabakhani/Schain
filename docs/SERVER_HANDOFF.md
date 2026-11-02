# Running on a GPU server

How to run SChain on a GPU server with `scripts/server_run.sh`. The smoke test has passed
end to end on a SLURM cluster with the real model and live retrieval (see
[Verified setup](#verified-setup)); Neo4j persistence (`--save`) and full-size runs have
not been exercised on a server yet.

## First run

```bash
git clone git@github.com:ErfanBabakhani/SChain.git    # or: git pull
cd SChain
cp .env.example .env          # see below for the few values a cluster may need
./scripts/server_run.sh smoke
```

The smoke test is a tiny real run (`config/smoke.yaml`): one seed, one searched tier,
three documents, the normal pipeline code and the real LLM, with its own data directory
so it never touches a real experiment. If it passes, do the full run:

```bash
./scripts/server_run.sh
```

Everything else is automatic: the script creates `.venv` and installs the project if
needed, reads the model and endpoint from `config/base.yaml`, starts vLLM if it is not
already up (and Neo4j too, for a run that saves to it), waits for the services to be
healthy, and then launches `python -m schain expand`.

### What you must set in `.env`

Nothing, unless one of these applies:

- the run saves to Neo4j (`--save`) → `NEO4J_PASSWORD`;
- the cluster uses SLURM → `SCHAIN_SLURM_PARTITION` (and possibly `SCHAIN_SLURM_ACCOUNT`).
  Set it even when your login shell shows GPUs: on some clusters the login node is a
  GPU node whose GPUs belong to other jobs;
- the GPU is shared with other processes → a lower `SCHAIN_VLLM_GPU_MEMORY_UTILIZATION`
  and `SCHAIN_VLLM_MAX_MODEL_LEN`;
- the GPU predates Ampere (no bfloat16) → `SCHAIN_VLLM_EXTRA_ARGS=--dtype half`;
- the NVIDIA driver is older than the newest vLLM's PyTorch needs (the vLLM log says
  "driver on your system is too old") → pin a release, e.g.
  `SCHAIN_VLLM_PACKAGE=vllm==0.19.1`, the last one on a CUDA 12 build of PyTorch;
- the model weights are gated → `HF_TOKEN`;
- you want the weights cached on a shared filesystem → `HF_HOME`.

The script fails immediately and names exactly what is missing; it never guesses SLURM
values, because partitions, accounts and GRES syntax differ per cluster. To discover them:

| Variable | How to find it |
|---|---|
| `SCHAIN_SLURM_PARTITION` | `sinfo -o '%P %G %l'`: partitions with a non-null GRES can run GPU jobs; `%l` is the time limit |
| `SCHAIN_SLURM_ACCOUNT` (only if required) | `sacctmgr show assoc where user=$USER format=account,partition,qos` |
| `SCHAIN_SLURM_GRES` (default `gpu:1`) | the `%G` column above, e.g. `gpu:a100:2` |
| `SCHAIN_SLURM_TIME` (default `24:00:00`) | must not exceed the partition's `%l` limit |

Everything else is detected: OS, Python, `.venv`, tmux, Docker, the GPUs and driver,
whether vLLM is installed, SLURM and any current allocation, disk space, the configured
model, endpoint and Neo4j database, which `.env` values are set, and whether Neo4j and the
LLM endpoint already answer. `./scripts/server_preflight.sh` prints all of it, showing
secrets only as `set` / `not set`. Every other `SCHAIN_*` setting in `.env.example` is
optional and has a working default.

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
| Smoke test first | `./scripts/server_run.sh smoke` |
| Full EXPAND run | `./scripts/server_run.sh` |
| Deep-discovery EXPAND | `./scripts/server_run.sh deep` |
| INGEST documents | `./scripts/server_run.sh ingest path/to/docs` |
| Save to Neo4j as well | add `--save` to any of the above |
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

- **Started by the run** → stopped when the workload finishes: vLLM (so an idle model does
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
the configured 4B model from Hugging Face — the log says so explicitly, so a long first start is not
mistaken for a hang. Set `HF_HOME` to cache on shared storage.

## Notes

- **APOC is not required** by the current implementation; every write is plain Cypher
  `MERGE`. The container the script starts does not install it.
- The runner can install vLLM when a GPU is present and `SCHAIN_VLLM_AUTO_INSTALL=1`
  (default). Set it to0 for a preconfigured runtime; GPU/package compatibility still
  needs runtime verification.
- Logs land in `logs/`, timestamped, e.g. `logs/expand_2026-08-29_210000.log`. vLLM gets
  its own `logs/vllm_*.log`. The path is printed when the run starts.
- Each EXPAND run also writes `logs/retrieval_manifest_<run_id>.csv`: one row per retrieved
  candidate, with fetch, dedup, relevance and Prompt-1 status. Use it to see why a document
  did or did not become graph facts. A summary is printed at the end of the run.
- Production depth is `max_tier: 5`, so a full run is large. The smoke test deliberately
  uses `--max-tier 1 --max-documents 3`; raise the depth only after it passes.
- **Start small.** The smoke test exists for this. Only scale up — configured `max_tier` and the configured
  cumulative document/runtime safety ceilings — once it has passed once.
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
5. `./scripts/server_run.sh smoke`.
6. Inspect the graph in Neo4j Browser:
   ```cypher
   MATCH (s:Company)-[r:suppliesTo]->(c:Company) RETURN s.name, c.name, r.evidence LIMIT 25;
   ```
7. `./scripts/server_run.sh` for the full run, then `./scripts/server_status.sh` to watch it.

## Verified setup

On 2 November 2026 `./scripts/server_run.sh smoke` passed as a SLURM batch job on one
NVIDIA RTX A6000 with about 13 GB free (the GPU was shared with other users), driver
535.183.01 (CUDA 12.2), Python 3.10 and vLLM 0.19.1 serving `Qwen/Qwen3-4B-Instruct-2507`,
with Neo4j saving off. The `.env` was:

```bash
SCHAIN_VLLM_MAX_MODEL_LEN=8192
SCHAIN_VLLM_GPU_MEMORY_UTILIZATION=0.25
SCHAIN_VLLM_EXTRA_ARGS=--enforce-eager
SCHAIN_VLLM_PACKAGE=vllm==0.19.1
```

From seed BMW it retrieved 18 candidates, processed 3 documents, accepted 8 facts and
exported 9 nodes and 7 edges with no tier violations, finding four Tier-1 suppliers
(Albemarle, ArcelorMittal, Lhyfe, Qualcomm Technologies). The pipeline took 7 minutes
after a one-time vLLM install and model download. GDELT answered `429 Too Many Requests`
during the run; failed queries are recorded and retried by the next run.

If the home directory is network storage (NFS), keep the repository and its `.venv` on a
local disk: a vLLM environment is tens of thousands of small files, and creating it on
NFS can stall.

The offline suite (`.venv/bin/python -m pytest -q`) passed with 723 tests and 18 skipped
integration tests that need a live model, network or Neo4j.
