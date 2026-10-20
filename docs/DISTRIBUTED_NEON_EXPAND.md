# Distributed Expand with Neon PostgreSQL

SChain can run one expansion experiment from several independent Colab runtimes. Neon
PostgreSQL is the authoritative queue and checkpoint store; Neo4j remains the optional
accepted knowledge graph. The existing SQLite workflow remains the default for local
and offline runs.

## Architecture and guarantees

Each worker joins the same `state.experiment_id`, claims one company task with
`FOR UPDATE SKIP LOCKED`, commits, and performs retrieval and LLM work without holding a
database transaction. A background thread renews the bounded lease using another pooled
connection. Each claim increments a fencing token. Every checkpoint and completion
compares task ID, worker ID, token, status, and lease time, so a worker returning after
lease loss cannot overwrite its replacement.

PostgreSQL stores experiments, workers, shared frontier/tiers, tasks, retrieval results,
source documents, extraction/resolution checkpoints, accepted/rejected facts, entity
aliases, graph outbox intents, and bounded events. Local files are disposable caches and
are reconstructed from PostgreSQL. Prompt-1 output is uploaded immediately after
extraction and Prompt-2 facts immediately after validation. An interrupted request before
its checkpoint may run again—the unavoidable at-least-once window—but durable effects
are idempotent and fenced.

Save uses a transactional outbox. Facts and graph intent commit with task completion.
Neo4j retains the existing canonical IDs and `MERGE` behavior. A crash after Neo4j but
before PostgreSQL acknowledgement safely replays the write. A later tier is blocked while
lower-tier required outbox work remains. Save OFF never constructs or contacts a Neo4j
driver and creates no outbox rows.

## Install and configure

```bash
python -m pip install -e ".[postgres]"
```

Use `config/distributed_neon_expand.yaml` as a template. Secrets belong in environment
variables, never YAML:

```text
NEON_DATABASE_URL
NEO4J_URI                 # Save ON only
NEO4J_USERNAME            # Save ON only
NEO4J_PASSWORD            # Save ON only
```

The Neon URL should require TLS (`sslmode=require`). It is never logged. Prefer a
dedicated database role. Core configuration is:

```yaml
state:
  backend: postgres
  experiment_id: bmw-supply-chain-v1
  database_url_env: NEON_DATABASE_URL

distributed:
  enabled: true
  worker_concurrency: 1
  task_lease_seconds: 600
  heartbeat_seconds: 30
  max_task_attempts: 3
  poll_seconds: 3.0
  connection_pool_min: 1
  connection_pool_max: 4

save:
  enabled: true
  backend: neo4j
  mode: incremental
```

The heartbeat must be less than half the lease.

## Colab and command-line workflow

Open `notebooks/SChain_Expand_Neon_Distributed.ipynb` and create the named Colab
Secrets. It is entirely in English, does not mount Drive, and never uses
`session.lock.json`.

1. Choose one `EXPERIMENT_ID` and identical critical settings in all Colabs.
2. Start one local Qwen/vLLM server per runtime.
3. Run the worker-session cell in every Colab.
4. Run the same cell again for another bounded set of tasks.
5. Use the final cell for shared progress, Graph JSON, and Audit JSON.

The first worker creates the experiment and seeds under an advisory transaction lock.
Others join. A join checks seeds, maximum tier, ontology and prompt hashes, model and
extraction settings, validation, Save policy, and schema version. The actual Git commit
is recorded on the worker but is not a hardcoded required ancestor. Completed experiments
are reported, not silently restarted.

```bash
python -m schain expand \
  --config config/distributed_neon_expand.yaml \
  --experiment-id bmw-supply-chain-v1 \
  --tasks-this-session 10 \
  --save
```

Use `--no-save` for no Neo4j access. `--restart` is rejected for shared PostgreSQL state;
choose a new experiment ID instead.

## Schema and migrations

Additive migrations run transactionally under a PostgreSQL advisory lock. Current schema
version: **1**. No SQLite data is automatically changed or imported.

| Table | Durable responsibility |
| --- | --- |
| `schain_experiments` | Manifest, schema, policy, lifecycle |
| `schain_workers` | Session heartbeat and bounded statistics |
| `schain_frontier` | Canonical companies, tiers, provenance |
| `schain_tasks` | Queue, attempts, owner, expiry, fencing token |
| `schain_retrieval_checkpoints` | Ranked retrieval results |
| `schain_documents` | Source text and stage checkpoints |
| `schain_facts` | Accepted/rejected facts and provenance |
| `schain_entity_aliases` | Shared mention-to-canonical decisions |
| `schain_graph_outbox` | Neo4j intent, retry lease, acknowledgement |
| `schain_events` | Distributed lifecycle/recovery events |

Queue selection stays at the minimum active tier. Pending, retrying, leased, and required
outbox work prevents premature advancement. Experiment completion requires no active task
or outbox item; exhausted retries mark failure. Frontier uniqueness is experiment plus
canonical company ID. Duplicate discoveries append provenance but schedule one task.

Entity aliases are locked transactionally. A different canonical ID or entity type for
the same normalized mention is rejected; similar strings alone are never merged.

## Crash and recovery

- **Before PostgreSQL commit:** rollback leaves no partial facts, aliases, discoveries, or
  outbox intent.
- **After a stage checkpoint:** another worker materializes stored document, extraction,
  resolution, and facts and resumes the next stage.
- **Worker disappears:** expiry permits a new fenced claim; the old completion is rejected.
- **After task commit, before Neo4j:** the outbox remains pending.
- **After Neo4j, before acknowledgement:** stable `MERGE` identities make replay safe.
- **Neo4j unavailable:** facts remain in PostgreSQL and graph work stays retryable.
- **All Colabs stop:** a later worker resumes without local or Drive artifacts.

`PostgresRepository.progress()` reports tier and experiment status, discovered/completed
companies, pending/running/failed tasks, active workers, facts, graph commits, outbox, and
per-worker counters. Events cover start/join, claim/renew/complete/fail/reclaim, fencing,
frontier duplicates, checkpoints, identity conflicts, outbox lifecycle, and completion.

The notebook exports `<experiment>.graph.json` and `<experiment>.audit.json`. These are
derived snapshots, not resume state. Telemetry retention defaults to 30 days; required
facts/checkpoints are never pruned with events.

## Free-plan operation

- Keep each worker pool at 1–4 connections and concurrency at one per GPU.
- Keep polling bounded (three seconds by default).
- Transactions surround only claims/persistence, never inference.
- Essential text and structured checkpoints are stored; binary pages and unbounded Audit
  blobs are not.
- Monitor `database_size_bytes` and provider quotas. Capacity errors are raised; facts are
  not silently discarded.
- The same design works with one worker; multiple free Colab GPUs are not assumed.

## Backup, restoration, and old snapshots

Use Neon recovery/branches where available or `pg_dump` with the provider TLS URL. Restore
into an empty database, verify `schain_schema_migrations`, then join using the original
manifest. Back up Aura with its supported export/backup facility. Restored pending outbox
items can safely replay into the matching graph. Do not copy only task rows: documents,
facts, aliases, frontier, and outbox jointly form resume state.

The older Drive snapshot workflow remains available for SQLite mode. It is not
automatically treated as a Neon experiment. A safe import would need to validate SQLite,
Parquet, hashes, ontology, manifest, document identity, and provenance as one generation;
until a dedicated importer does that, keep it separate rather than silently discarding or
mislabeling checkpoints.

## Troubleshooting and verification

- **Manifest mismatch:** restore the first worker's critical settings or use a new ID.
- **No task claimed:** another worker may own the tier, retry delay may be active, or a
  lower-tier outbox write may be pending.
- **Repeated lease loss:** increase the lease and check Neon connectivity.
- **Neo4j retries:** verify Aura capacity/credentials; PostgreSQL facts remain safe.
- **Pool exhaustion:** reduce per-worker pools or the number of Colabs.
- **vLLM failure:** lower model length/memory utilization and inspect the notebook's unique
  local log.

Real PostgreSQL reliability tests (not SQLite proxies) run with:

```bash
SCHAIN_TEST_POSTGRES_URL=postgresql://... \
python -m pytest -q tests/test_distributed_postgres.py
```

They cover concurrent claims and discovery, expiry/fencing, rollback, reconnect/resume,
checkpoint survival, tier/outbox barriers, Save OFF, and crash-after-Neo4j replay. Live
Neon, AuraDB, vLLM, and Colab checks require their actual credentials/runtime.
