# Configuration

Every runnable config `include:`s the file above it and states **only what differs**, so
each setting has exactly one home. Dictionaries merge key by key; lists and plain values
replace.

```text
base.yaml                                   shared: model, storage, validation, runtime
├── ingest.yaml                             INGEST: documents you already have
└── expansion.yaml                          EXPAND: seeds, tiers, retrieval, query templates
    ├── smoke.yaml                          few-minute end-to-end check (own data dir)
    ├── colab_professor_expand.yaml         single Colab runtime, saving to Neo4j
    ├── distributed_neon_expand.yaml        many workers, one PostgreSQL experiment
    └── deep_discovery.yaml                 high-recall profile (versioned)
        └── distributed_neon_deep_discovery.yaml
```

Read by every mode: `ontology.yaml` (entity and relation types, allowed patterns, known
aliases), `relation_evidence.yaml` (what counts as evidence for each relation) and
`source_reputation.yaml` (informational domain scores).

## Which file to edit

| To change | Edit |
|---|---|
| model, endpoint, generation settings | `base.yaml` → `llm` |
| where data, checkpoints and the graph JSON go | `base.yaml` → `storage` |
| run time budget, resume behaviour | `base.yaml` → `runtime` |
| fact validation policy | `base.yaml` → `validation` |
| seed companies, depth (`max_tier`), documents per run | `expansion.yaml` |
| search providers, query templates, per-tier budgets | `expansion.yaml` → `expansion` |
| chunking and input location for INGEST | `ingest.yaml` |
| known company aliases | `ontology.yaml` → `aliases` |

For a one-off change, prefer a CLI flag (`--max-tier`, `--max-documents`, `--seed`,
`--save`) or your own small overlay over editing a shipped file:

```yaml
# my_run.yaml
include:
  - config/expansion.yaml
expansion:
  max_documents: 50
```

```bash
python -m schain expand --config my_run.yaml
```

## Rules worth knowing

- **Secrets never go in these files.** Neo4j credentials, API keys and server-specific
  settings come from `.env` (see `.env.example`).
- **An experiment's identity is fixed once it starts:** seeds, `max_tier`, model and
  generation settings, ingest and validation policy, and the ontology. Resuming with a
  different identity is refused; use `--restart` or a different `storage` directory.
  Operational limits (budgets, concurrency, timeouts) may change between runs.
- **Runtime overlays** (`runtime_overlay: true`, e.g. the Colab file) may change how a run
  executes but never its seeds or `max_tier`; the loader rejects those keys there.
- **Search limits resolve field by field, closest value wins:**
  `search.defaults` < tier `defaults` < per-query override. `max_results` is a ceiling on
  candidates per provider per query, not a document target.
- **`deep_discovery.yaml` is a versioned profile.** Distributed workers check its budgets
  against the approved values, so change them only together with a new profile version.
