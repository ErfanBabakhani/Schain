# Project Structure

Keep the repository small. Do not create files until they are actually needed.

```text
SChain/
├── CLAUDE.md
├── README.md
├── GIT_SETUP.md
├── PROJECT_DECISIONS.md
├── PROJECT_STRUCTURE.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── config/
│   ├── base.yaml          # shared settings: llm, neo4j, storage, runtime, concurrency
│   ├── ontology.yaml      # the static V1 ontology
│   ├── ingest.yaml         # INGEST mode
│   └── expansion.yaml      # EXPAND mode: seeds, tiers, query templates, sources
├── docs/
│   ├── SupplyChain_KG_V1_Architecture_Brief_Final.pdf
│   ├── SupplyChain_KG_V1_Architecture_Brief_Final.tex
│   └── EXTERNAL_DATASETS.md
├── schain/
│   ├── __main__.py        # CLI: ingest | expand | validate | export
│   ├── config.py          # config loading, include merge, search-limit precedence
│   ├── names.py           # deterministic name normalization and canonical ids
│   ├── documents.py       # loading, normalization, hashing, raw/normalized layers
│   ├── ontology.py        # static ontology -> GraphSchema, and Prompt 1
│   ├── state.py           # SQLite checkpoint/resume state
│   ├── pipeline.py        # the per-document path shared by both modes
│   ├── ingest.py          # INGEST driver
│   ├── graph_export.py    # frontend-ready semantic graph JSON (derived artifact)
│   ├── llm/               # provider factory (vLLM or hosted OpenAI-compatible API)
│   ├── kg/                # extraction, entity resolution, facts table, Neo4j writer
│   ├── sources/           # retrieval: wikipedia, gdelt, google_news, orchestration
│   ├── expansion/         # tier query generation, persistent frontier, EXPAND loop
│   └── validation/        # external supplier-customer reference comparison
├── data/                  # all gitignored
│   ├── raw/               # documents.jsonl
│   ├── processed/         # documents / facts / search_results / entity_resolution parquet
│   ├── external/          # externally obtained datasets
│   └── state/             # state.db
├── outputs/               # derived graph JSON (gitignored)
└── tests/
```

## Note on the package directory

The earlier draft of this file proposed `src/`. The code lives in `schain/` instead, so
that `python -m schain ingest ...` works as the documented CLI. Module responsibilities
are unchanged from that draft.

## Keep modules narrow

- `sources/`: retrieve documents; no KG extraction.
- `llm/`: build the model client; no graph logic.
- `kg/`: wrap and reuse the Neo4j GraphRAG extraction components, plus entity resolution,
  the facts table and the idempotent writer.
- `expansion/`: decide which frontier entity is processed next and which queries apply.
- `state.py`: persistence and checkpointing only.
- `pipeline.py` / `ingest.py` / `expansion/loop.py`: orchestration.

Do not add architecture layers unless a concrete requirement needs them.
