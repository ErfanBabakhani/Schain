# Values you may need to provide once

Almost everything is detected automatically. This page lists only what cannot be.

## Auto-detected — nothing for you to provide

Operating system, repository root, Python version, `.venv` presence, tmux, Docker, curl,
NVIDIA GPU presence, GPU model/count/memory, driver version, whether `vllm` is installed,
whether SLURM exists, whether you are already inside an allocation, GPU partitions
advertised by `sinfo`, disk space, the configured model, LLM base URL, Neo4j URI and
database, the checkpoint path, which `.env` variables are set, and whether Neo4j and the
LLM endpoint are already reachable.

Run `./scripts/server_preflight.sh` to see all of it. Secrets are shown as `set` /
`not set`, never printed.

## You must provide once — only in these cases

### Always (unless you run everything with `--no-neo4j`)

| Variable | Why |
|---|---|
| `NEO4J_PASSWORD` | Your choice. Also used as the container password if the script starts Neo4j. |

### Only if the server uses SLURM *and* no GPU is visible *and* no LLM endpoint is running

The script cannot invent these: partition names, accounts and GRES syntax differ per
cluster, and guessing them produces jobs that are rejected or silently mis-scheduled. It
fails immediately and names exactly what is missing.

| Variable | How to discover it on the server |
|---|---|
| `SCHAIN_SLURM_PARTITION` | `sinfo -o '%P %G %l'` — partitions with a non-null GRES can run GPU jobs; `%l` is the time limit |
| `SCHAIN_SLURM_ACCOUNT` (only if required) | `sacctmgr show assoc where user=$USER format=account,partition,qos` |
| `SCHAIN_SLURM_GRES` (default `gpu:1`) | the `%G` column above, e.g. `gpu:a100:2` |
| `SCHAIN_SLURM_TIME` (default `08:00:00`) | must not exceed the partition limit from `%l` |

Put them in `.env`. Nothing is hard-coded in the scripts.

### Only if the model needs authentication or a shared cache

| Variable | When |
|---|---|
| `HF_TOKEN` | the weights are gated on Hugging Face |
| `HF_HOME` | you want the cache on a large shared filesystem rather than `~/.cache` |

If none of the SLURM or Hugging Face cases apply, `NEO4J_PASSWORD` is the only value you
need to set.
