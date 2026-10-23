# Incremental Neo4j Save and persistent Colab resume

## What is persisted where

SChain has two independent persistence layers:

- **Final/online graph:** accepted, validated canonical entities and relations are
  merged incrementally into the configured Neo4j database. Save mode never sends
  raw extraction output or rejected facts to Neo4j.
- **Execution checkpoint:** frontier, tiers, document stages, searches, retry state,
  resolution cache, facts, evidence, manifests, graph export, and audit files live
  in local SQLite/Parquet/files. In Colab, immutable verified snapshots of those
  artifacts live under `MyDrive/SChain/expand_runs/<EXPERIMENT_ID>/`.

The Graph JSON is a derived portable export in `storage.graph_export`. It is not the
resume checkpoint and it is not a substitute for Neo4j.

## Save configuration and CLI

Save is deliberately off in the base configuration:

```yaml
save:
  enabled: false
  backend: neo4j
  mode: incremental
  max_attempts: 3
  retry_backoff_seconds: 1.0
```

Use `--save` to enable it or `--no-save` to disable it. `--no-neo4j` remains a
legacy alias for `--no-save`; argparse rejects combinations such as
`--save --no-neo4j`. With Save disabled, SChain never constructs a Neo4j driver,
verifies connectivity, creates a constraint, reads Neo4j, or writes Neo4j.

Each validated document is written immediately after its accepted facts reach the
Parquet layer. Canonical node IDs, semantic edge identity, fact IDs, aliases, source
URLs, evidence records, and document provenance use idempotent `MERGE`/set-union
updates. A `neo4j_commits` SQLite ledger and `neo4j_save_*` audit events record each
logical attempt. A document reaches `COMMITTED` only after the full save succeeds.
Transient Neo4j connectivity errors receive bounded exponential-backoff retries.

If Neo4j commits and the runtime dies before SQLite is updated or snapshotted, resume
sees the document as `RESOLVED` and safely replays the idempotent merge. This is the
intentional recovery protocol for the non-atomic Neo4j/SQLite boundary.

## Durable checkpoint configuration

```yaml
checkpoint_store:
  enabled: true
  experiment_id: schain-expand-2026-10-19
  directory: /content/drive/MyDrive/SChain/expand_runs
  snapshot_every_documents: 1
  keep_snapshots: 3
  lock_stale_seconds: 21600
```

Use `--new-run` once for a new ID and `--resume` thereafter. Resume is also the
normal default. A new run refuses to overwrite an existing experiment; resume
refuses to invent a missing one. The durable metadata fingerprints seeds, expansion
policy and tier limit, model/generation settings, ingestion policy, validation,
ontology content, prompt-source files, Neo4j database, and Save policy. A mismatch
stops before restoration or model/database work.

SQLite remains on `/content`. Snapshot creation uses `sqlite3.Connection.backup`,
copies essential artifacts into a new generation, hashes each file, renames the
finished generation, and only then publishes `LATEST.json`. Restore verifies every
hash before replacing local working files. The live SQLite WAL is never operated
directly on Google Drive.

`ACTIVE.lock` prevents two live sessions from using one experiment ID. A lock older
than `lock_stale_seconds` is considered stale. `--force-checkpoint-unlock` exists for
an explicitly verified abandoned session. Google Drive mount visibility is not a
distributed consensus system, so do not deliberately launch two notebooks for the
same ID even with this guard.

Credentials are read from `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` and
are excluded from snapshots. Never add them to YAML.

## Colab procedure

Open [SChain_Expand_Persistent_Save.ipynb](../notebooks/SChain_Expand.ipynb)
in a GPU Colab runtime.

1. Add the three Neo4j values to Colab Secrets and grant notebook access.
2. Set one stable `EXPERIMENT_ID`.
3. Set `RUN_MODE = "NEW_RUN"` only for its first execution.
4. Run the cells. They mount Drive, clone/fast-forward `mirza`, print the exact
   commit, install SChain/vLLM, generate an untracked runtime overlay, verify Neo4j,
   start Qwen, and invoke the real `schain expand --save` command.
5. For a later runtime, reuse the same ID and leave `RUN_MODE = "RESUME"`. The CLI
   restores the latest verified snapshot before opening SQLite.

The notebook defaults to Qwen/Qwen3-4B-Instruct-2507 and a 16,384-token model
context. Reduce it to 8,192 if the assigned GPU cannot serve that context.

The progress cell reports frontier state, tier counts, document checkpoint states,
accepted facts, logical Neo4j commits, and the current durable generation. The final
cell packages local Graph JSON, Audit JSON, metrics, and artifacts for download.

## Crash/recovery demonstration

The automated regression simulates loss after Neo4j has accepted all MERGE
statements but before the document checkpoint changes to `COMMITTED`:

1. First execution leaves the document `RESOLVED` while the Save ledger records the
   completed logical write.
2. Resume loads persisted facts without any LLM call.
3. The same node/edge/evidence rows are replayed.
4. Stable MERGE keys and evidence set-union leave one copy of every fact/evidence.
5. The document then reaches `COMMITTED`.

Durable restore tests separately delete the local checkpoint and recover frontier
tiers, elapsed state, entity-resolution cache, facts, and SQLite state from
Drive-like snapshot storage.

## Operational limitations

- A crash can lose work since the last completed snapshot, at most the configured
  interval plus an in-flight document. Default Colab configuration snapshots each
  completed or failed document and again at clean shutdown.
- Full Google Drive availability and mount consistency are controlled by Colab and
  Google. An incomplete generation is ignored because `LATEST.json` is published
  last.
- Neo4j and checkpoint publication are not a distributed atomic transaction.
  Idempotent replay is the recovery mechanism.
- Changing a critical experiment setting requires a new experiment ID. The durable
  compatibility policy intentionally errs on the conservative side.
