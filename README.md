# Supply Chain KG V1 — Claude Starter Pack

This package contains the current implementation brief and development rules for the Supply Chain Knowledge Graph V1 project.

## Start here

Claude should read in this order:

1. `CLAUDE.md` — mandatory development workflow and engineering rules.
2. `docs/SupplyChain_KG_V1_Architecture_Brief_Final.pdf` — compact project architecture.
3. `PROJECT_DECISIONS.md` — frozen V1 decisions and scope.
4. `PROJECT_STRUCTURE.md` — proposed simple repository structure.
5. `config/config.example.yaml` — example configuration shape.

The project is intentionally designed to be simple, traceable, and based on official/common packages instead of custom reimplementation.

Work autonomously within the frozen V1 decisions in `CLAUDE.md`. Keep each change simple, traceable, based on common/official packages where possible, and avoid unrelated or cosmetic additions.

## Running V1

Install into a virtual environment (Python 3.10-3.14):

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -e .
```

Copy `.env.example` to `.env` and fill in the Neo4j credentials. Behaviour is read from
`config/config.yaml`; if that file is absent, `config/config.example.yaml` is used.

INGEST one document (`.txt`, `.md` or `.pdf`):

```bash
.venv/bin/python src/main.py ingest --input path/to/document.txt
```

Add `--no-neo4j` to stop after the facts table, which is useful before a graph instance
exists. Run the offline tests, which need neither a model nor a database, with:

```bash
.venv/bin/python -m pytest tests -q
```

The pipeline expects an OpenAI-compatible model endpoint at `llm.base_url` (vLLM serving
the configured Qwen model) and, unless `--no-neo4j` is passed, a Neo4j 5.x instance. The
graph writes are plain Cypher `MERGE` statements, so APOC is not required; it would only
become relevant if the neo4j-graphrag entity resolvers were adopted later.

## Repository location

This starter pack is intended to be extracted **directly into the root of the private `SChain` repository**:

`git@github.com:ErfanBabakhani/SChain.git`

After extraction, `CLAUDE.md`, `README.md`, `config/`, `docs/`, and future `src/` should all be directly under the `SChain` root. Do not keep this starter pack inside an extra nested folder.

For Git setup and private-repository authentication, see `GIT_SETUP.md`.
