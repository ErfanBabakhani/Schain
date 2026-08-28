# CLAUDE.md — Supply Chain Knowledge Graph V1

## Read this first

Before proposing or writing code:

1. Read `docs/SupplyChain_KG_V1_Architecture_Brief_Final.pdf`.
2. Read `PROJECT_DECISIONS.md`.
3. Read `PROJECT_STRUCTURE.md`.
4. Treat the decisions in those files as the current V1 scope unless the user explicitly changes them.

This project must be developed **step by step**. Do not silently redesign, extend, or "improve" the project.

---

## Repository root assumption

The current working directory is the **`SChain` repository root**.

Do not create another nested project directory such as `supply-chain-kg/`, `SChain/`, or `app/` around the project. All project files, source code, configuration, tests, and documentation should be created directly relative to this repository root.

Expected remote repository:

- SSH: `git@github.com:ErfanBabakhani/SChain.git`
- HTTPS: `https://github.com/ErfanBabakhani/SChain.git`

The repository is private. Never place credentials, tokens, SSH keys, API keys, Neo4j passwords, or downloaded model weights in Git.

---

## Mandatory working style

These rules apply to **every development task** in this project.

### Work autonomously, but keep the logic visible

The user does not have time to supervise every small implementation step. Do **not** stop for approval after each design decision.

For each task:

1. Briefly identify the problem being solved and the part of the pipeline it affects.
2. Use the simplest conventional solution that fits the existing V1 architecture.
3. Prefer official or widely used packages before writing custom infrastructure.
4. For any non-trivial change, keep the logic easy to trace: state the main data flow, functions/modules involved, and important configuration.
5. Implement only what is required for the task. Do not add unrelated improvements, speculative features, or cosmetic engineering.
6. If a necessary technical choice is new or non-obvious, explain it briefly in plain language rather than hiding it.
7. If an ambiguity can be resolved from the frozen project decisions or existing configuration, use those decisions and continue. Ask only when a missing decision would make the implementation unsafe, incompatible, or materially different.

The goal is to move the project forward without turning it into a black box.

---

## Git hygiene

Keep the repository clean from the beginning.

Required rules:

- Keep `.gitignore` at the repository root and respect it.
- Never commit `.env`, credentials, private keys, API tokens, Neo4j passwords, local databases, runtime checkpoints, model weights, caches, downloaded articles, or generated raw/processed datasets.
- Keep `.env.example` committed with variable names only and no secrets.
- Do not commit large external datasets unless there is a deliberate project decision to version them.
- Do not rewrite Git history or force-push as part of normal development.
- Make small, meaningful commits that correspond to coherent implementation steps.
- Do not create extra branches, GitHub Actions, release automation, badges, hooks, or repository cosmetics unless they become necessary for the actual project.
- Before committing, check `git status` and avoid generated or local-only files.

If Git is already initialized in `SChain`, use the existing repository. Do not run `git init` again or create a nested repository.

---

## Engineering principles

### Keep it simple and human-traceable

The code should be easy for another researcher or engineer to follow line by line.

Prefer:
- small functions with clear names,
- plain Python data structures,
- explicit data flow,
- configuration files for behavior,
- readable intermediate files,
- simple logs that explain what the pipeline is doing.

Avoid:
- unnecessary class hierarchies,
- deep inheritance,
- custom frameworks,
- "manager/service/repository/orchestrator" layers unless genuinely needed,
- hidden state,
- magic behavior,
- unnecessary metaprogramming,
- autonomous agent frameworks,
- complicated design patterns.

### Reuse before building

Before implementing a conventional function, first check whether an **official or widely used package** already provides it.

This is especially important for:
- Neo4j KG construction/writing,
- schema-guided entity/relation extraction,
- text splitting,
- LLM serving,
- standard data formats,
- HTTP/search clients,
- retries and persistence where a standard approach exists.

Prefer official code and official documentation over custom reimplementation.

Current preferred foundation:
- `neo4j-graphrag` / official Neo4j GraphRAG KG Builder components for the KG pipeline.
- Neo4j for graph storage.
- vLLM for local Qwen serving.
- standard Python packages such as `pandas`, `pyarrow`, `sqlite3`, and normal HTTP/async libraries where appropriate.

Avoid uncommon or large dependencies unless they are genuinely needed. If one is necessary, explain why it is needed and keep its use isolated and simple.

### No cosmetic engineering

Do **not** add work that only makes the project look more sophisticated.

Do not add, unless explicitly requested:
- dashboards,
- web UI,
- custom graph visualization frontend,
- decorative reports,
- agent interfaces,
- unnecessary abstractions,
- complex plugin systems,
- speculative future features,
- premature optimization.

Neo4j Browser is sufficient for basic graph inspection in V1.

### Do not hide new concepts

If you introduce a new library, data structure, algorithm, or design idea, explain it briefly in plain language. If the user needs more detail, they will ask.

Never hide complexity behind a black box just to make the answer shorter.

### No silent scope changes

If you see a potentially useful improvement:
1. mention it,
2. explain why it may help,
3. ask whether to include it.

Do not implement it automatically.

---

## V1 scope

V1 has two operating modes only:

### `INGEST`
The user already has documents/data. The system processes them and builds or extends the KG.

No automatic search or graph expansion is performed.

### `EXPAND`
The user provides seed entities. The system searches for relevant documents, extracts supply-chain facts, adds new upstream companies, and continues to the configured tier boundary.

V1 explicitly excludes:
- existing-edge monitoring,
- pairwise relationship update searches,
- relationship change detection,
- terminated-agreement monitoring,
- downstream risk/optimization analysis.

These may be considered in a later version only.

---

## Current LLM plan

Primary local model:
- `Qwen/Qwen3.8-27B`
- instruct / non-thinking behavior
- `enable_thinking = false`
- served with vLLM
- configured behind a small provider interface so the KG pipeline does not depend directly on one model.

An API model such as GPT can be used as an optional comparison/provider, but the core code must use one common LLM interface.

Do not scatter provider-specific calls through the KG pipeline.

---

## Static ontology for V1

Use a **static ontology**, following the AlMahri-style controlled schema for the first version.

Entity types:
- `Company`
- `Location`
- `Material`
- `Person`
- `Product`
- `Mine`

Core relations:
- `suppliesTo`
- `locatedIn`
- `owns`
- `produces`

Do not dynamically add new entity or relation types in V1.

The IOFoundry Supply Chain Reference Ontology (SCRO) is a useful reference/alignment layer, but do not import its full complexity into the extraction prompt for V1.

---

## LLM extraction logic

Use two clear conceptual prompts/stages.

### Prompt 1 — Entity and relation extraction

Purpose:
- identify entities allowed by the static ontology,
- identify supported relations between those entities,
- return structured output suitable for validation and storage.

Conceptually:

`document text -> entities + relations + evidence`

NER and relation extraction are conceptually separate tasks, but the implementation may use one schema-guided extraction call if the chosen Neo4j GraphRAG component supports both together. Do not force extra LLM calls only for cosmetic architectural separation.

The output should be structured and traceable, including source/evidence metadata.

### Prompt 2 — Entity disambiguation

Purpose:
- detect mentions that refer to the same real-world entity,
- group aliases/variants,
- prevent duplicate KG nodes.

Example:

`CATL` and `Contemporary Amperex Technology Co. Limited` should resolve to the same canonical company when supported.

Run disambiguation by entity type where practical, following the AlMahri idea. Store the resolution result so a resumed run does not repeat an already completed LLM call.

Deterministic normalization should happen before LLM disambiguation for trivial cases such as casing, whitespace, or punctuation.

---

## Seeds and datasets

Default AlMahri-style seed set:

EV manufacturers:
- Toyota
- Tesla Inc.
- BMW
- BYD Co., Ltd.

Battery suppliers:
- Samsung SDI
- CATL
- Panasonic

Mining/raw-material companies:
- Zijin Mining
- Albemarle Corporation
- Tianqi Lithium
- Ganfeng Lithium Co Ltd
- Norilsk Nickel
- Jinchuan Group

The seed list is configurable and is not limited to these companies.

### AlMahri data limitation

AlMahri reports a manually annotated Wikipedia-based evaluation corpus, but the exact packaged annotated dataset / complete page list has not been found as a public downloadable repository in our review.

Therefore V1 supports:
- `INGEST` for any controlled document dataset we already have,
- `EXPAND` for discovery from configured seed entities.

### External validation dataset

Use the **Sectivia supply-chain dataset** as a small external reference/validation dataset because it contains explicit supplier-customer relationships.

Do not treat Sectivia as the main text corpus for LLM extraction. Its main role is graph/relationship validation and sanity checking.

Normalize external datasets into our project data schema before use.

---

## Storage, checkpointing, and resume

Checkpoint/resume is a hard requirement.

Default behavior:
- `resume: true`

The user must also be able to request a fresh run:
- `restart_from_scratch: true`
- or an equivalent CLI flag such as `--restart`.

Use simple persistent state, preferably SQLite (`state.db`), so the pipeline can continue after interruption.

Track states such as:
- pending,
- processing,
- extracted,
- resolved,
- committed,
- failed.

In `EXPAND`, the frontier queue must also be persistent.

Do not keep the only copy of important intermediate state in RAM.

Suggested data layers:
- raw documents: JSONL,
- normalized documents: Parquet,
- extracted facts: Parquet,
- search results: Parquet or SQLite,
- runtime/checkpoint state: SQLite,
- graph: Neo4j.

Neo4j must not be the only copy of extracted facts.

---

## Deduplication and idempotency

Duplicates must be prevented at multiple levels.

Documents:
- normalized URL,
- content hash.

Entities:
- deterministic normalization,
- Prompt 2 entity disambiguation,
- canonical entity identifiers.

Graph writes:
- use idempotent Neo4j `MERGE` logic rather than blind repeated `CREATE`.

A resumed or repeated run must not create duplicate documents, nodes, or relations.

---

## Graph expansion design

Graph expansion is **node-centric and tier-aware**, not pairwise.

Do not generate queries for every pair of graph nodes.

For each expandable frontier company:
1. read its tier,
2. load the query templates configured for that tier,
3. substitute the entity name into each template,
4. retrieve up to the configured result limit for that query,
5. deduplicate candidate documents,
6. filter for relevance,
7. process relevant documents with Prompt 1,
8. resolve entities with Prompt 2,
9. merge validated facts,
10. add newly discovered upstream companies to the next frontier,
11. continue until the configured tier boundary.

Only supply-chain-relevant company nodes should normally become expansion targets.

### Tier boundary

Default:
- Tier 0 = seeds,
- Tier 1–3 = discovered upstream supply-chain tiers,
- maximum upstream depth = 3.

Tier depth must remain configurable.

### Search sources

Search/retrieval may use configured sources such as:
- Wikipedia,
- Google News,
- GDELT.

Important: **do not hard-code the logic as "always fetch the Wikipedia page of the entity."**

Wikipedia is one retrieval source among others. The system should retrieve relevant Wikipedia content through the configured retrieval/search strategy, just as other sources provide relevant documents.

### Tier-specific query templates

Each tier can have its own query templates/keywords.

Example:

Tier 0 intents:
- `"{entity}" suppliers`
- `"{entity}" sourcing`
- `"{entity}" battery suppliers`
- `"{entity}" semiconductor suppliers`

For `BMW`, these become:
- `"BMW" suppliers`
- `"BMW" sourcing`
- `"BMW" battery suppliers`
- `"BMW" semiconductor suppliers`

If CATL is discovered as an upstream supplier and enters Tier 1, Tier 1 templates may be different, for example:
- `"{entity}" suppliers`
- `"{entity}" component suppliers`
- `"{entity}" raw material suppliers`

For CATL:
- `"CATL" suppliers`
- `"CATL" component suppliers`
- `"CATL" raw material suppliers`

This keeps expansion approximately proportional to the number of expanded nodes times a small number of configured templates, rather than all node pairs.

### Query result limits and configuration precedence

Each query has a maximum number of results.

Configuration follows nearest-value-wins precedence:

`global defaults < tier defaults < query-specific overrides`

Resolve fields independently.

Example:
- global `max_results = 20`,
- Tier 0 default `max_results = 30`,
- one specific Tier 0 query `max_results = 15`.

That specific query uses 15, other Tier 0 queries use 30, and tiers without an override use 20.

Keep this hierarchy simple and explicit.

---

## Final outputs and visualization

The **primary final output** of V1 is the supply-chain knowledge graph stored in **Neo4j**.

Neo4j is not the only persisted representation. Keep the traceable intermediate data as well:
- raw/retrieved documents,
- normalized documents,
- extracted facts,
- entity-resolution results,
- checkpoint/runtime state.

This allows the graph to be rebuilt or inspected without repeating expensive LLM calls.

### Visualization

V1 includes basic graph visualization through **Neo4j Browser** using Cypher queries for inspection, debugging, and demonstrations.

Do not build a custom visualization frontend or dashboard in V1. Visualization is a lightweight way to inspect the resulting KG, not a separate research module.

---

## Preferred implementation order

Do not implement everything at once.

A safe order is:

1. Confirm package/API choices and exact current Neo4j GraphRAG interfaces.
2. Build one-document `INGEST` path end-to-end.
3. Add persistent state, resume/restart, and deduplication.
4. Extend `INGEST` to multiple documents/datasets.
5. Add the persistent `EXPAND` frontier and source retrieval.
6. Add tier-specific query generation and limits.
7. Add validation/evaluation with the selected reference dataset.

At each numbered stage, keep the implementation explicit and traceable, use the existing architecture and official/common packages where possible, and do not add unrelated features.

Do not jump ahead just because a later feature seems easy.

---

## Final rule

The goal is not to make the project look advanced.

The goal is to make it:
- correct,
- reproducible,
- easy to explain,
- easy to resume,
- easy to debug,
- easy for a human to trace.

Use the simplest conventional solution that satisfies the requested V1 requirement.
