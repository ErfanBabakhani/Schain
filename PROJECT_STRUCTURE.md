# Proposed Project Structure

Keep the repository small. Do not create all files until they are actually needed.

```text
SChain/
├── CLAUDE.md
├── README.md
├── GIT_SETUP.md
├── .gitignore
├── pyproject.toml
├── .env.example
├── config/
│   └── config.example.yaml
├── docs/
│   └── SupplyChain_KG_V1_Architecture_Brief_Final.pdf
├── src/
│   ├── sources/          # retrieval adapters: Wikipedia, GDELT, Google News
│   ├── llm/              # provider interface + Qwen/API adapters
│   ├── kg/               # Neo4j GraphRAG pipeline glue + entity resolution
│   ├── expansion/        # persistent frontier + tier/query logic
│   ├── state.py          # simple SQLite checkpoint/resume logic
│   └── main.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── external/
│   └── state/
└── tests/
```

## Keep modules narrow

- `sources/`: retrieve documents; do not perform KG extraction.
- `llm/`: normalize model calls; do not contain graph logic.
- `kg/`: wrap/reuse Neo4j GraphRAG extraction/writing components.
- `expansion/`: decide what frontier entity is processed next and which configured queries apply.
- `state.py`: persistence/checkpointing only.
- `main.py`: simple orchestration.

Do not create extra architecture layers unless a concrete project requirement genuinely needs them.
