# Project Structure

Keep the repository small. Do not create files until they are actually needed.

```text
SChain/
├── README.md
├── pyproject.toml
├── .env.example                # secrets and server settings template (copy to .env)
├── config/                     # see config/README.md for how the files layer
│   ├── base.yaml               # shared: LLM, storage, validation, runtime, backends
│   ├── ingest.yaml             # INGEST mode
│   ├── expansion.yaml          # EXPAND mode: seeds, tiers, retrieval, query templates
│   ├── smoke.yaml              # few-minute end-to-end check, isolated data directory
│   ├── deep_discovery.yaml     # high-recall EXPAND profile (versioned)
│   ├── colab_professor_expand.yaml          # Colab runtime overlay
│   ├── distributed_neon_expand.yaml         # PostgreSQL-coordinated workers
│   ├── distributed_neon_deep_discovery.yaml # the same, deep-discovery profile
│   ├── ontology.yaml           # the static V1 ontology and known aliases
│   ├── relation_evidence.yaml  # relation-specific evidence policy
│   └── source_reputation.yaml  # domain score priors (informational metadata)
├── schain/
│   ├── __main__.py             # CLI: ingest | expand | export | view | validate | evaluate
│   ├── config.py               # config loading, include merge, search-limit precedence
│   ├── pipeline.py             # the per-document path shared by both modes
│   ├── ingest.py               # INGEST driver
│   ├── documents.py            # loading, normalization, hashing, raw/normalized layers
│   ├── names.py                # deterministic name normalization and canonical ids
│   ├── ontology.py             # static ontology -> GraphSchema, and Prompt 1
│   ├── state.py                # SQLite checkpoint/resume state
│   ├── state_backend.py        # local SQLite versus distributed PostgreSQL state
│   ├── run_config.py           # experiment identity, safe config snapshots, provenance
│   ├── checkpoint.py           # atomic writes for expensive intermediate results
│   ├── durable.py              # durable snapshots for ephemeral runtimes (Colab)
│   ├── async_runtime.py        # one long-lived event loop for async clients
│   ├── audit_trace.py          # optional run-scoped audit trace (audit.py: old imports)
│   ├── graph_export.py         # frontend-ready semantic graph JSON (derived artifact)
│   ├── viewer.py, viewer_assets/  # offline HTML graph explorer
│   ├── verification.py         # no-op stub (multi-source verification removed)
│   ├── llm/                    # provider factory (vLLM or hosted OpenAI-compatible API)
│   ├── kg/                     # extraction, resolution, validation, facts, support, writer
│   ├── sources/                # retrieval providers, ranking, origin relevance, reputation
│   ├── expansion/              # frontier, tier queries, escalation, budgets, EXPAND loop
│   ├── distributed/            # PostgreSQL leases, outbox and maintenance for many workers
│   └── validation/             # external reference comparison and gold-set evaluation
├── scripts/                    # server_run.sh (one-command GPU server run), status, preflight
├── notebooks/                  # Colab INGEST and EXPAND notebooks
├── examples/                   # fictional demo graph and standalone explorer
├── demo/                       # standalone graph explorer with the professor dataset
├── docs/                       # design, decisions, audits and operating guides
├── data/                       # all gitignored: raw/, processed/, state/, external/, smoke/
├── outputs/                    # derived graph JSON (gitignored)
└── tests/
```

## Keep modules narrow

- `sources/`: retrieve documents; score domains for display metadata; no KG extraction.
- `llm/`: build the model client; no graph logic.
- `kg/`: extraction, entity resolution, candidate validation, facts table, edge support,
  and the idempotent writer.
- `verification.py`: disabled stub; second-source searches are not run.
- `expansion/`: decide which frontier entity is processed next and which queries apply;
  enqueue suppliers from ontology+evidence-valid suppliesTo edges.
- `state.py`: persistence and checkpointing only.
- `pipeline.py` / `ingest.py` / `expansion/loop.py`: orchestration.

Do not add architecture layers unless a concrete requirement needs them.

The package lives in `schain/` rather than `src/`, so that `python -m schain ...` works
as the documented CLI.
