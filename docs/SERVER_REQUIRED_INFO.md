# Legacy server scripts

These settings apply to `scripts/server_run.sh`, which manages the CLI, vLLM and Neo4j. For the SQLite API and dedicated worker, use the root [README](../README.md).

```bash
./scripts/server_preflight.sh
./scripts/server_run.sh --help
./scripts/server_run.sh --smoke-test
./scripts/server_status.sh
```

Set credentials in an untracked `.env` file:

- `NEO4J_PASSWORD`: required for Neo4j writes; omit with `--no-neo4j`.
- `SCHAIN_SLURM_PARTITION`: required when the script needs a GPU allocation. Find partitions with `sinfo -o '%P %G %l'`.
- `SCHAIN_SLURM_ACCOUNT`: set if the cluster requires an account.
- `SCHAIN_SLURM_GRES`: requested GPU resource, such as `gpu:1`.
- `SCHAIN_SLURM_TIME`: batch time limit; default `24:00:00`.
- `HF_TOKEN`: needed only for gated model weights.
- `HF_HOME`: optional model cache directory.

Use only allocations and persistent services permitted by the institution.
