# SChain: paper review and implementation upgrade

**8 September 2026 · Updated code version 0.2.0**

Reviewed inputs: `Schain-mirza1(1).zip` and all 12 distinct supplied papers. The separately
uploaded main article and its project-source copy are byte-identical. The ZIP contains
source code and tests, but no completed extraction dataset to assess as a real supply graph.

## Recommendation

Improve the reliability of each edge before trying to grow a larger graph. Your code
already implements much of the useful architecture: a static ontology, deterministic
normalization, LLM entity resolution, node-centric upstream expansion, persistent state,
source metadata, evidence checking and idempotent graph writes. These foundations should
be retained. The strongest additions are stricter source grounding, preserved evidence,
reliable chunk-level resume, correct graph tiers and an inspectable graph view.

This release implements those changes and adds an exact relation evaluator. It does not
claim a new research method or measured extraction-accuracy improvement. That claim needs
a controlled held-out comparison using your actual model and source corpus.

## What differs from the main article?

The main article is AlMahri, Xu and Brintrup, *Enhancing supply chain visibility with
knowledge graphs and large language models* (online 2025),
[DOI](https://doi.org/10.1080/00207543.2025.2575841). Page numbers below refer to the supplied
PDF including its publisher cover.

| Point | Paper | Supplied code / retained choice |
|---|---|---|
| Ontology | Six entity types and four relations in quantitative evaluation | Same Company, Location, Material, Person, Product, Mine; suppliesTo, locatedIn, owns, produces |
| Extraction model | GPT-4o evaluation | Configurable local/API model; inherited Qwen setting is not a verified equivalent |
| Prompting | Described as zero-shot; definitions and entity examples | Includes a worked input/output demonstration, so not an exact zero-shot replication |
| Entity resolution | Second prompt groups equivalent names by type | Deterministic normalization, aliases, cache, then optional LLM grouping |
| Final case-study graph | At least three distinct credible sources per triple | One explicit validated source can create an edge; second-source search intentionally disabled |
| Source reputation | Credibility screening in case-study pipeline | Descriptive metadata; not an edge-acceptance threshold or confidence probability |
| Scope | Static explicit graph; case study additionally includes contains | Static four-relation ontology; configurable tier-limited upstream search |
| Focal company | Multiple EV firms in the case study | Configurable seeds; BMW is one possible seed, not the universal target |

The paper's numerical entity-disambiguation `score` is a cluster identifier, **not a
confidence score** (Figure 4, PDF p. 12). The existing single-source policy is a deliberate
coverage-versus-corroboration choice, and this upgrade preserves it.

The paper reports all-relation F1 **0.328 ± 0.026** and suppliesTo precision
**0.326 ± 0.170**, with suppliesTo recall **0.629 ± 0.183** (Table 1, PDF p. 16).
These results make false supplier edges a serious concern: one mistaken edge can expand
an entire irrelevant upstream branch. The reported entity-disambiguation precision@1
of 0.973 is a separate task and cannot establish graph correctness (Table 2, PDF p. 17).

## Ideas implemented in the updated code

| Rank | Concrete problem found | Implemented change | Expected benefit / boundary |
|---|---|---|---|
| 1 | `document_text` was passed into validation but not used; invented quotes could pass an LLM judge | Require the quoted evidence to occur in the actual document after Unicode/whitespace normalization | Rejects invented source wording before another model call; does not prove the quoted statement is true |
| 2 | Validation with no LLM manufactured a supports result | Keep deterministic-only evidence explicitly unchecked | Stops false claims of model validation; such evidence cannot satisfy the existing validated frontier gate |
| 3 | A retry could repeat all successful chunks | Atomic per-chunk checkpoints, stable chunk IDs, bounded transient retries and prompt/model-sensitive partial caches | Avoids repeating completed work after partial failure; real runtime gain depends on failure rate and document size |
| 4 | IGNORE could silently mark a partially extracted document complete | Default RAISE in both operating modes | Failure remains visible/retryable; explicit IGNORE remains an opt-in partial-result policy |
| 5 | Same-document triples collapsed different quotes/chunks; Neo4j replaced the latest quote | Evidence-specific fact IDs and additive quote/document/fact records on one semantic edge | Preserves every retained witness without inflating supplier-edge or document counts |
| 6 | Export quality flags were hard-coded true; cached support could be stale | Derive nullable validation flags and counts from the actual fact rows | Legacy unknowns remain unknown; source count and evidence count stay distinct |
| 7 | Stale frontier pointers and alias variants could produce bad tiers or duplicate seeds | Alias-aware seed matching and indexed, bounded upstream BFS with supported parent paths | Correct shortest observed tiers, terminal boundary, and no phantom supplier paths |
| 8 | Resumed writes lost aliases; interrupted source persistence could leave missing provenance | Restore persisted entity resolution and source records; atomic JSON stage checkpoints | More faithful recovery without repeating successful extraction |
| 9 | Fresh-checkout service checks guessed endpoints or treated TCP as Neo4j | Read actual config with available Python; require Neo4j protocol/driver health | More reliable Linux server startup and ownership checks |
| 10 | JSON output was difficult to inspect and demonstrate | Self-contained offline HTML explorer and SVG export | Useful overview plus exact source inspection; no extra frontend package or CDN |
| 11 | External dataset overlap cannot measure extraction precision | Exact document-level directed triple evaluator, including negative documents | Enables a valid held-out quality comparison when independently annotated labels are supplied |

The existing static ontology, INGEST/EXPAND modes, one-source policy, retrieval providers,
LLM provider abstraction, seeds and Neo4j canonical graph remain. The new viewer is the
explicit scope extension authorized by your request for a useful graph representation.

Key files: `schain/kg/validate.py`, `schain/kg/extract.py`, `schain/kg/facts.py`,
`schain/kg/writer.py`, `schain/pipeline.py`, `schain/graph_export.py`,
`schain/checkpoint.py`, `schain/viewer.py`, `schain/viewer_assets/`,
`schain/validation/gold.py`, `schain/__main__.py`, and the existing server scripts.

The use of RAISE follows the documented chunk error behavior of the
[official Neo4j KG builder](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html).
The writer retains indexed uniqueness and idempotent MERGE, consistent with the
[official Cypher MERGE guidance](https://neo4j.com/docs/cypher-manual/current/clauses/merge/).
The installed API used for testing was neo4j-graphrag 1.19.0; no untested major API migration
was introduced.

## How the graph should be interpreted

The overview shows supplier-to-customer arrows and the shortest observed supplier tier
relative to the configured seed set. Ownership, location and product relationships remain
available as contextual layers, but do not create supply tiers. A seed need not be an OEM:
all configured seed companies start at tier 0. The viewer does not reinterpret that tier
as a business classification.

Select a relationship to read its exact quotes and source metadata. A source-matched quote
and a model-supported relation are separate properties. Several documents can repeat one
syndicated claim; `support_count` is not independent confirmation. Source reputation is not
a calibrated probability. No observed path means only no path in the available graph,
not absence of the real dependency. Extraction timestamps do not establish contractual
start/end dates or whether an older relationship is still active.

The viewer has search/alias matching, per-seed upstream filtering, relation/type/tier/evidence
filters, neighborhood focus, pan/zoom, a paginated relationship table, SVG download and
filtered JSON export. It reads local files and needs no internet or model. Its bounded
SVG drawing intentionally displays at most 300 nodes and 1,000 edges, with a visible limit
message; larger filtered datasets remain available in the table/JSON. The supplied demo
uses fictional companies and manually authored source sentences, not extraction results.

## What each supplied paper contributes

These are design transfers, not claims that this release reproduces each paper's system.
The detailed reviews under `docs/literature/` include full bibliographic identities,
PDF page references, method summaries, results and limitations.

| Source | Useful transfer to this implementation | Boundary |
|---|---|---|
| P01 AlMahri et al. (2025), Enhancing supply chain visibility… | Controlled schema, typed entity resolution, evidence and supplier direction (PDF pp. 8–17) | Model, prompting and three-source case-study gate differ |
| P02 Huang et al. (2025), ORLM, arXiv:2405.17743v5 | Separate executable/schema validity from semantic correctness; hard negatives (pp. 15–19, 30) | Optimization-model training results are not KG extraction results |
| P03 Callefi et al. (2026), Building a pathway… | Establish structured data and semantic capabilities before higher autonomy (pp. 9–12, 16–19) | Capability hierarchy is not a measured pipeline speedup |
| P04 Song et al. (2026), LLMs in SCM: systematic review… | Source-linked supplier knowledge, provenance and contradiction awareness (pp. 21–23) | Conceptual review, not a new validated extraction algorithm |
| P05 Congès et al., R-IO SUITE, journal 2026 / HAL 2025 | Deterministic metamodel checks and separation of persistent facts from proposals (pp. 22, 26–28) | Do not instantiate imagined roles as real supplier edges |
| P06 AlMahri et al. (2026), Automating Supply Chain Disruption Monitoring, arXiv:2601.09680 | Deterministic graph traversal, JSON contracts, cached tools and tier views (pp. 9–18) | Assumes an existing KG; risk formulas are inconsistent across its body/case study |
| P07 Sukharamwala & Won (2026), LLM4SCM | Structured backend calculations remain authoritative; inspectable outputs (pp. 9, 14–16) | Solver/data integration requires quantities absent from this graph |
| P08 Quan et al. (2025), Risk Assessment in Hyperconnected Logistic Hub Network Deployment, arXiv:2503.21115 | Bounded tool workflows and reuse of retrieved information (pp. 3–5) | Logistics-hub proof of concept, not supplier extraction benchmark |
| P09 Quan et al. (2026), multi-agent risk assessment / MARS | Source-constrained feedback, explicit rubrics and bounded revision (pp. 8–14) | Weather severity results do not validate supplier relations |
| P10 Trifone et al. (2026), Drivers and constraints of agentic AI… | Prioritize data quality, analyst oversight and understandable evidence (pp. 16–19, 32–34) | Nineteen-interview exploratory study, not performance measurements |
| P11 Baharmand & Behdani (2026), eco-friendly humanitarian supply chains | Trace numerical inputs to sources and keep scenario/proxy data separate (pp. 9–15) | Do not invent costs/capacities/emissions for a supplier graph |
| P12 Jannelli et al. (2025), Agentic LLMs…consensus-seeking | Tool/model combinations need ablation; extra coordination is not automatically better (pp. 11–17) | Inventory simulation requires demand, lead times and stock states absent here |

## Research ideas to consider next — not implemented

1. **Selective semantic review.** Recheck only ambiguous directional evidence and disputed
   entity merges. Compare one validator with one bounded independent revision. Do not assume
   adding agents improves quality enough to justify its latency and cost.
2. **Publisher/content-aware corroboration as a separate view.** Restore independent-source
   analysis only if you explicitly choose to change the current acceptance policy. First
   deduplicate syndicated text and distinguish primary announcements from repeated reporting.
3. **Corpus-level efficiency.** Benchmark Parquet reads/writes and global edge-support refresh
   before replacing them with partitioned tables or incremental updates. The current pipeline
   still rewrites shared Parquet artifacts and is intended for one writer; chunk caching does
   not solve that scaling limit.
4. **Temporal validity as a later schema/version.** Distinguish announced, planned, active and
   terminated contracts only with explicit text/time evidence and dedicated evaluation. This
   release remains static; file dates are not relationship validity intervals.
5. **Grounded question answering after graph quality is established.** Return deterministic
   paths and exact source excerpts for questions such as who supplies BMW within two tiers.
   Any prose explanation must reference those paths and must not invent numbers or certainty.

I do not recommend adding autonomous procurement, inventory negotiation, supplier replacement,
or numerical risk/optimization scores now. The supplied papers require operational variables
that this code does not contain. A large or visually dense graph is not evidence of a better
supply-chain model.

## How to demonstrate an actual improvement

Use the same held-out documents, model settings, retrieval budget, ontology and seeds for the
original and upgraded pipelines. Independently label the correct canonical directed triples,
including documents with no relation, reversed roles, subsidiaries, vague partnerships,
planned deals, repeated news and multi-chunk articles. Keep prompt examples separate.

Report directed relation precision/recall/F1 overall and per relation; supplier-edge precision
is especially important because expansion amplifies mistakes. Separately record false entity
merges, source-quote grounding failures, accepted/rejected facts, supported reachable company
count, wall-clock time, actual model calls/tokens, retries and interrupted-run recovery. An
ablation should isolate grounding, caching and any later review stage. Higher source/edge counts
alone are not a success metric. The exact evaluator provided here scores accepted triples;
NER and entity-resolution quality need their own labels and metrics.

## Validation and practical limits

The final test suite completed with **421 passed, 4 skipped** (86.39 seconds; six existing
pandas FutureWarnings). See `TEST_RESULTS.md` for the environment. Checks exercise parsing,
source grounding, negative documents, directed edge scoring, cache failure/recovery, metadata,
Neo4j query parameters, tiers, seeds and shell probes. Four existing integration tests require
external services and are skipped when unavailable. Python tests use model/database stubs;
JavaScript syntax and offline packaging are checked. The actual SVG canvas was also rendered
and visually inspected with Node/CairoSVG; the synthetic default view has nine entities and eight supply edges. No live browser interaction, model endpoint,
retrieval experiment, GPU inference, or Neo4j server integration was run.

Legacy document checkpoints are intentionally reused. New source validation is not retroactive.
Use separate storage and a separate Neo4j database/instance for a clean comparison. The
`--restart` flag deletes configured local generated artifacts but leaves Neo4j intact; it cannot
remove stale database edges. New IDs preserve quote/chunk evidence going forward; previously
lost quotes need fresh extraction. A crash after a model response but before its checkpoint may
still repeat that call. JSON checkpoints are atomic, while existing Parquet storage remains
single-writer and is not transactional across all layers.

Follow `START_HERE.md` for installation and commands. This release makes the implementation
more auditable and resilient; measured accuracy and performance gains remain to be established.
