# Project Structure

Keep the repository small. Do not create files until they are actually needed.

```text
SChain/
├── README.md
├── PROJECT_DECISIONS.md
├── PROJECT_STRUCTURE.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── config/
│   ├── base.yaml          # shared settings: llm, neo4j, storage, runtime, validation
│   ├── ontology.yaml      # the static V1 ontology
│   ├── source_reputation.yaml  # domain score priors (informational metadata)
│   ├── ingest.yaml         # INGEST mode
│   └── expansion.yaml      # EXPAND mode: seeds, tiers, query templates, sources
├── docs/
│   ├── SupplyChain_KG_V1_Architecture_Brief_Final.pdf
│   ├── SupplyChain_KG_V1_Architecture_Brief_Final.tex
│   ├── EXTERNAL_DATASETS.md
│   └── SOURCE_REPUTATION_AND_VERIFICATION.md
├── schain/
│   ├── __main__.py        # CLI: ingest | expand | validate | export
│   ├── config.py          # config loading, include merge, search-limit precedence
│   ├── names.py           # deterministic name normalization and canonical ids
│   ├── documents.py       # loading, normalization, hashing, raw/normalized layers
│   ├── ontology.py        # static ontology -> GraphSchema, and Prompt 1
│   ├── state.py           # SQLite checkpoint/resume state
│   ├── pipeline.py        # the per-document path shared by both modes
│   ├── ingest.py          # INGEST driver
│   ├── verification.py    # no-op stub (multi-source verification removed)
│   ├── graph_export.py    # frontend-ready semantic graph JSON (derived artifact)
│   ├── llm/               # provider factory (vLLM or hosted OpenAI-compatible API)
│   ├── kg/                # extraction, resolve, validate, facts, support, Neo4j writer
│   ├── sources/           # retrieval + reputation + ranking
│   ├── expansion/         # tier query generation, persistent frontier, EXPAND loop
│   └── validation/        # external supplier-customer reference comparison
├── data/                  # all gitignored
│   ├── raw/               # documents.jsonl
│   ├── processed/         # documents / facts / rejected_facts / edge_support / …
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

## 0.2.0 additions

- `schain/checkpoint.py`: atomic extraction JSON checkpoints.
- `schain/viewer.py`, `schain/viewer_assets/`: offline graph explorer generator/assets.
- `schain/validation/gold.py`: exact document-level directed triple evaluation.
- `examples/`: explicitly fictional demo graph, standalone HTML and rebuild script.
- `START_HERE.md`, `docs/UPGRADE_REVIEW.md`, `docs/literature/`: use and review notes.

CLI now also includes `view` and `evaluate`; neither calls an LLM or Neo4j.

## 0.3.0 audit additions

- `schain/run_config.py`: immutable experiment identity, safe config snapshots, export provenance.
- `schain/expansion/budget.py`: tier allocations and deeper-tier target reserves.
- `schain/kg/entity_types.py`: conservative model/facility type hints.
- `docs/AUDIT_V0.3.0.md`: complete audit, policy, validation and remaining risks.
- SQLite additive origin/usage/run tables and per-document metrics preserve compatibility.
