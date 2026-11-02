# Colab, Neo4j saving and distributed runs

Three capabilities beyond a plain local run, each off by default:

- **Saving to Neo4j**: accepted facts are merged into Neo4j as each document completes.
- **Durable checkpoints**: on an ephemeral runtime such as Colab, the local SQLite and
  Parquet state is snapshotted to Google Drive and restored by the next runtime.
- **Distributed EXPAND**: several runtimes work on one experiment through Neon
  PostgreSQL, which then holds the task queue and every checkpoint.

The notebooks: [`SChain_Ingest.ipynb`](../notebooks/SChain_Ingest.ipynb) runs INGEST
(`config/ingest.yaml`); [`SChain_Expand.ipynb`](../notebooks/SChain_Expand.ipynb) runs a
distributed deep-discovery worker (`config/distributed_neon_deep_discovery.yaml`).

## Saving to Neo4j

SChain keeps two independent layers:

- **The graph**: accepted, validated canonical entities and relations, merged into Neo4j
  incrementally. Raw extraction output and rejected facts never go there.
- **The execution checkpoint**: frontier, tiers, document stages, searches, retry state,
  resolution cache, facts, evidence, manifests and audit files, in local SQLite/Parquet
  (or PostgreSQL in distributed mode).

The graph JSON (`storage.graph_export`) is a derived export. It is neither the resume
checkpoint nor a substitute for Neo4j.

Save is off in `config/base.yaml`:

```yaml
save:
  enabled: false
  backend: neo4j
  mode: incremental
  max_attempts: 3
  retry_backoff_seconds: 1.0
```

`--save` turns it on and `--no-save` off (`--no-neo4j` is a legacy alias; contradictory
combinations are rejected). With Save off, SChain never constructs a Neo4j driver or
contacts the database. Credentials come from `NEO4J_URI`, `NEO4J_USERNAME` and
`NEO4J_PASSWORD`, never from YAML.

Each validated document is written as soon as its accepted facts reach the Parquet layer,
with idempotent `MERGE` and set-union updates keyed on canonical node IDs, semantic edge
identity and fact IDs. A `neo4j_commits` ledger and `neo4j_save_*` audit events record
each attempt, and transient connection errors get bounded exponential-backoff retries. A
document reaches `COMMITTED` only after its whole save succeeds. If Neo4j commits and the
process dies before SQLite is updated, resume sees the document as `RESOLVED` and replays
the merge, which changes nothing: that replay is the recovery protocol for the
non-atomic Neo4j/SQLite boundary.

## Durable checkpoints on Colab (SQLite mode)

```yaml
checkpoint_store:
  enabled: true
  experiment_id: schain-expand-2026-10-19
  directory: /content/drive/MyDrive/SChain/expand_runs
  snapshot_every_documents: 1
  keep_snapshots: 3
  lock_stale_seconds: 21600
```

`config/colab_professor_expand.yaml` is meant to receive this block, with local storage
paths, from an untracked runtime overlay written by the notebook.

`--new-run` starts a new experiment ID and `--resume` (the default) continues one. A new
run refuses to overwrite an existing experiment and resume refuses to invent a missing
one. The stored metadata fingerprints seeds, expansion policy and tier limit, model
settings, ingest policy, validation, ontology, prompt sources, Neo4j database and Save
policy; a mismatch stops before anything is restored.

SQLite itself stays on local disk (`/content`); its WAL is never operated on Drive. A
snapshot uses `sqlite3.Connection.backup`, copies the essential artifacts into a new
generation, hashes every file, renames the finished generation and only then publishes
`LATEST.json`, so an incomplete generation is ignored. Restore verifies every hash before
replacing local files.

`ACTIVE.lock` stops two live sessions from sharing an experiment ID. A lock older than
`lock_stale_seconds` is stale, and `--force-checkpoint-unlock` releases one you have
verified is abandoned. Drive is not a consensus system, so never start two notebooks on
one ID.

A crash loses at most the work since the last snapshot plus the document in flight.
Changing a critical setting requires a new experiment ID.

## Distributed EXPAND with Neon PostgreSQL

### How it works

Each worker joins the same `state.experiment_id`, claims one company task with
`FOR UPDATE SKIP LOCKED`, commits, and does retrieval and LLM work without holding a
database transaction. A background thread renews the task lease. Each claim increments a
fencing token, and every checkpoint and completion compares task, worker, token, status
and lease time, so a worker that lost its lease cannot overwrite its replacement.

PostgreSQL stores experiments, workers, the shared frontier and tiers, tasks, retrieval
results, documents, extraction and resolution checkpoints, accepted and rejected facts,
entity aliases, graph outbox intents and bounded events. Local files are disposable
caches. Prompt 1 output is uploaded right after extraction and Prompt 2 facts right after
validation; a request interrupted before its checkpoint may run again, but every durable
effect is idempotent and fenced.

Save uses a transactional outbox: facts and graph intent commit together with the task.
A crash after Neo4j but before the acknowledgement replays safely, and a later tier waits
until lower-tier outbox work is done. With Save off no outbox rows are created.

### Configure and run

```bash
python -m pip install -e ".[postgres]"
```

Secrets are environment variables: `NEON_DATABASE_URL` (with `sslmode=require`; it is
never logged), plus `NEO4J_URI`, `NEO4J_USERNAME` and `NEO4J_PASSWORD` when saving. The
templates are `config/distributed_neon_expand.yaml` and
`config/distributed_neon_deep_discovery.yaml`; the core settings are:

```yaml
state:
  backend: postgres
  experiment_id: bmw-supply-chain-v1
  database_url_env: NEON_DATABASE_URL

distributed:
  enabled: true
  worker_concurrency: 1
  task_lease_seconds: 600
  heartbeat_seconds: 30        # must be less than half the lease
  max_task_attempts: 3
  poll_seconds: 3.0
  connection_pool_min: 1
  connection_pool_max: 4
```

```bash
python -m schain expand --config config/distributed_neon_expand.yaml \
  --experiment-id bmw-supply-chain-v1 --tasks-this-session 10 --save
```

In Colab: choose one `EXPERIMENT_ID` with identical critical settings everywhere, start
one vLLM server per runtime, run the worker cell in every runtime (again for another
bounded batch), and use the final cell for shared progress, graph JSON and audit JSON.

The first worker creates the experiment and its seeds under an advisory lock; the others
join. Joining checks seeds, maximum tier, ontology and prompt hashes, model and extraction
settings, validation, Save policy and schema version. `--restart` is rejected for shared
state: use a new experiment ID instead.

### Schema

Migrations are additive and run under an advisory lock (`SCHEMA_VERSION` in
`schain/distributed/schema.py`, currently 2). SQLite data is never changed or imported
automatically.

| Table | Holds |
|---|---|
| `schain_experiments` | manifest, schema, policy, lifecycle |
| `schain_workers` | session heartbeat and statistics |
| `schain_frontier` | canonical companies, tiers, provenance |
| `schain_tasks` | queue, attempts, owner, expiry, fencing token |
| `schain_retrieval_checkpoints` | ranked retrieval results |
| `schain_documents` | source text and stage checkpoints |
| `schain_facts` | accepted and rejected facts with provenance |
| `schain_entity_aliases` | shared mention-to-canonical decisions |
| `schain_graph_outbox` | Neo4j intent, retry lease, acknowledgement |
| `schain_events` | lifecycle and recovery events |
| `schain_discovery_paths` | every accepted `suppliesTo` path and its outcome |
| `schain_maintenance_runs` | applied maintenance operations, such as an extension |

Work is taken from the lowest active tier, and pending or leased tasks and outbox items
block advancement. A frontier entry is unique per experiment and canonical company;
duplicate discoveries add provenance but schedule one task. A mention already mapped to a
different canonical ID or entity type is rejected; similar strings are never merged.

### Crash and recovery

- **Before a PostgreSQL commit**: rollback leaves no partial facts, aliases or outbox rows.
- **After a stage checkpoint**: another worker picks up the stored stage and continues.
- **A worker disappears**: its lease expires, a new fenced claim follows, and the old
  completion is rejected.
- **After the task commit, before Neo4j**: the outbox item stays pending.
- **After Neo4j, before the acknowledgement**: stable `MERGE` identities make replay safe.
- **All runtimes stop**: a later worker resumes from PostgreSQL alone.

### Operating on a free plan

Keep each worker's pool at 1–4 connections and one worker per GPU. Transactions wrap only
claims and persistence, never inference. Watch the database size and provider quotas:
capacity errors are raised, never silently dropped. One worker is enough; several free
Colab GPUs are not assumed.

Back up with Neon branches or `pg_dump`, and restore into an empty database before joining
with the original manifest. Documents, facts, aliases, frontier and outbox together form
the resume state, so never copy only the task rows.

### Troubleshooting

- **Manifest mismatch**: restore the first worker's critical settings or use a new ID.
- **No task claimed**: another worker owns the tier, a retry delay is active, or
  lower-tier outbox work is pending.
- **Repeated lease loss**: increase the lease and check Neon connectivity.
- **Pool exhaustion**: shrink the per-worker pools or run fewer runtimes.
- **vLLM failure**: lower the model length or memory utilization.

PostgreSQL reliability tests need a disposable database, never the production one:

```bash
SCHAIN_TEST_POSTGRES_URL=postgresql://... python -m pytest -q tests/test_distributed_postgres.py
```

## Extending an experiment with deep discovery

The `deep_discovery_v1` profile (`config/deep_discovery.yaml`) extends the existing
`bmw-supply-chain-v2` experiment in place: nothing is created, cloned, reset or deleted.
The model stays `Qwen/Qwen3-4B-Instruct-2507` and Save stays off.

| Tier | Document target | Document maximum |
|---:|---:|---:|
| 0 | 80 | 200 |
| 1 | 80 | 160 |
| 2 | 60 | 120 |
| 3 | 50 | 100 |
| 4 | 40 | 80 |
| 5 | terminal | 0 |

It allows 20,000 documents per experiment, 250 full-text fetches, 100 title lookups and
600 ranked candidates per company. It starts with 28 upstream-oriented queries and can add
component, LLM-planned and official/PDF query families in up to four bounded rounds.
Stopping for diminishing returns requires coverage across source domains and query
families; a company with zero yield goes through the bounded recovery ladder instead.

To apply it once:

1. Back up the Neon database.
2. Stop every worker and wait for task and outbox leases to expire.
3. Check the status and read the plan, which changes nothing:

   ```bash
   python -m schain.distributed.maintenance status --experiment-id bmw-supply-chain-v2 --json
   python -m schain.distributed.maintenance extend --experiment-id bmw-supply-chain-v2 --new-max-tier 5 --topup-tier 1 --plan
   ```

4. Confirm the plan has no active workers or leases and no manifest mismatch, then apply:

   ```bash
   python -m schain.distributed.maintenance extend --experiment-id bmw-supply-chain-v2 --new-max-tier 5 --topup-tier 1 --apply --confirm-id bmw-supply-chain-v2
   ```

The apply step takes the experiment lock, repeats the worker and lease checks, archives
the old manifest, records the new profile and limits, adds top-up tasks and reopens
terminal nodes. Completed tasks, documents, facts, aliases and evidence are never updated
or deleted, and repeating the apply returns the stored result. Afterwards run workers from
`SChain_Expand.ipynb`, which resumes existing experiments only.

Every accepted `suppliesTo` fact is recorded in `schain_discovery_paths` with its depth
and outcome (new supplier, new path to a known entity, shorter path, another customer,
self-edge, or omitted from the frontier). `schain_frontier.tier` stays the minimum
observed distance, so peer and cyclic discoveries never push a company deeper.

To compare two exported snapshots:

```bash
python -m schain.distributed.benchmark before.json after.json --labels labelled_edges.json
```

The report covers unique verified suppliers, upstream edges, document relevance,
full-text ratio, runtime and cost; precision and false negatives are `null` without
labels.

Known risks: web providers can rate-limit or block full-text retrieval; tasks already
leased can finish after another task reaches the document ceiling; and
`company:toyota` and `company:toyota-motor` are deliberately not merged unless an
evidenced alias links them.
