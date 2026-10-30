# Source review: baseline extraction, ORLM, R-IO SUITE, and disruption monitoring

Prepared from the supplied full-text PDFs. Page pointers below use **PDF page numbers**, starting at the first page displayed by a PDF viewer. The main article has a publisher cover, so its printed article page is one lower than the PDF page. Bibliographic years and all results below describe the supplied versions, not a new benchmark of Schain.

## 1. AlMahri, Xu and Brintrup: the main article

**Sara AlMahri, Liming Xu and Alexandra Brintrup. _Enhancing supply chain visibility with knowledge graphs and large language models_. International Journal of Production Research, published online 5 November 2025. DOI: https://doi.org/10.1080/00207543.2025.2575841.** Supplied source: `01-Enhancing-supply-chain-visibility-with-knowledge-graphs-and-large-language-models.pdf` (byte-identical to the separately uploaded `(2)` version).

### What the method actually does

- Identifies and screens public web sources; extracts entities and relations using GPT-4o; disambiguates entities in a second prompt; stores and visualizes a knowledge graph in Neo4j. New documents can be processed in incremental ingestion batches instead of rerunning the whole corpus (PDF pp. 7–9).
- The framework is **static and explicit**: temporal changes and prediction of missing or hidden relationships are explicitly outside its scope (PDF p. 8; pp. 22–23). A static graph is distinct from a fixed ontology: both are adopted here, but they are different design dimensions.
- Prompt 1 uses predefined entity and relation categories, their definitions and examples. The authors call this zero-shot because they do not provide paired document-to-correct-output demonstrations. Examples of entity names or paraphrases of a relation do not, on their own, mean task training or fine-tuning occurred (PDF p. 9).
- The screenshots in Figures 2–3 show the illustrative three-type ontology Company/Location/Product, and produces/locatedIn/suppliesTo. However, the **quantitative evaluation** uses six entity types—Company, Location, Material, Person, Product, Mine—and four relations—locatedIn, suppliesTo, owns, produces (PDF pp. 10–11, 13–14). The case study adds `contains` (PDF p. 19). These are successive schema choices, not automatic ontology discovery.
- Prompt 2 groups nodes by entity type and gives the same numerical identifier to names referring to one entity. Its field called **`score` is an identity-cluster number, not confidence or probability**. This was checked visually in Figure 4 (PDF p. 12); it matters when interpreting inherited code.
- The real-world case study uses **at least three distinct credible sources per triple**. The authors explicitly call that a configurable, conservative design choice rather than an empirically validated threshold (PDF pp. 8, 19). Technical extraction evaluation is a separate exercise and does not evaluate source truthfulness (PDF pp. 13–14).

### The actual quantitative baseline

The manually annotated Wikipedia evaluation contains 1,277 sentences, 1,158 entities and 641 relationships; extraction is repeated ten times (PDF pp. 14–17).

| Task | Precision, mean ± SD | Recall, mean ± SD | F1, mean ± SD | Source |
|---|---:|---:|---:|---|
| All entities | 0.729 ± 0.020 | 0.405 ± 0.023 | 0.520 ± 0.020 | Table 1, PDF p. 16 |
| All relations | 0.449 ± 0.033 | 0.259 ± 0.025 | 0.328 ± 0.026 | Table 1, PDF p. 16 |
| suppliesTo | 0.326 ± 0.170 | 0.629 ± 0.183 | 0.420 ± 0.172 | Table 1, PDF p. 16 |
| Product entities | 0.521 ± 0.057 | 0.177 ± 0.018 | 0.264 ± 0.024 | Table 1, PDF p. 16 |

These values do **not** justify describing extracted supply relationships as highly accurate. In particular, suppliesTo precision is the immediate reason to strengthen evidence and direction checks before upstream graph expansion.

Entity disambiguation changes 629 nodes to 554 and reports micro-precision@1 of 0.973, versus 0.894 for string matching and 0.939 for a TF-IDF/cosine baseline (Table 2, PDF p. 17). This metric is not a calibrated confidence value and does not establish whole-graph correctness. The consistency analysis mostly reports counts, which can remain stable while the actual identities of the extracted edges change. A stronger local evaluation should compare exact canonical entity sets and directed typed triple sets across runs as well as counts.

The case study maps electric-vehicle critical mineral supply chains; BMW is one of multiple OEMs, not the inherent universal target of the method. It also discusses Toyota, Tesla, BYD and graph examples centered on Audi (PDF pp. 19–21). The code's chosen seed controls the focal firm.

The paper's small resource example processes 17 articles, reports 76,731 input and 13,519 output tokens, 281 triples, and 249.3 seconds (PDF p. 21). Treat these as historical reported measurements for that run. Do not copy the model endpoint/prices as present-day defaults, promise that Schain matches its runtime, or accept its Θ(T+E) assertion as a general complexity proof: LLM inference, entity matching, database indexes and graph traversal have additional costs. The paper itself says large-scale/dynamic maintenance remains unevaluated (PDF pp. 22–23).

### Comparison with the supplied Schain project

Checked the current `config/ontology.yaml`, README and `docs/SOURCE_REPUTATION_AND_VERIFICATION.md` in addition to the paper. Implementation-level findings are documented in ../UPGRADE_REVIEW.md.

| Design | Main article | Current documented Schain choice | Interpretation |
|---|---|---|---|
| Ontology | Predefined types; six entities/four relations in evaluation | Same six/four, frozen configuration | Faithful to quantitative schema; schema generation would be an unnecessary scope change |
| Temporal facts | Static, explicit facts | Static V1 | Aligned |
| Focal company | Several OEM case studies | Seed-driven EXPAND, with BMW aliases | Valid specialization; avoid presenting all unrelated article mentions as BMW's supply chain |
| Three-source corroboration | Case-study final graph gate | Explicitly disabled; one valid explicit source accepted | Deliberate coverage/verification tradeoff; document it rather than reintroducing it silently |
| Reputation | Screening/gating in case study | Informational metadata | An accepted fact is text-supported, not externally verified |
| Source independence | Distinct credible sources required in case study | Deduplicated document support count | Multiple documents do not prove independent corroboration; syndicated text may be one claim |
| Entity resolution | LLM within each type | Aliases and deterministic normalization before LLM | Sensible efficiency extension, provided subsidiaries are not merged with parents |
| Tier movement | Supplier networks in examples | Only suppliesTo advances tiers | Correct semantic restriction; ownership/location do not establish a supply tier |
| Additional relation | Case study includes contains | Four relations only | Explicit scope choice; do not invent contains/usedIn/material-flow edges from co-occurrence |

### Highest-value upgrades supported by this baseline

1. Require relation evidence that supports **both endpoints, the relation and its direction**. A document that merely mentions a company and product is insufficient to assert a particular customer shipment. Do not accept inferred links through a common location or product.
2. Store and display exact evidence, document ID/URL, extraction version, entity aliases and acceptance reason. Label results `accepted` or `text-supported`; keep reputation and support count as separate descriptive fields.
3. Compute upstream tiers using incoming suppliesTo edges toward the seed. Keep canonical entities unique; a company can participate at different depths in different paths. Show the shortest observed supply distance as a node's display tier and retain path-specific depths separately.
4. Use a supplier-only layer for the overview. Make locations, materials, ownership and evidence available on demand. This preserves the multi-relational graph while preventing geographic hubs and product categories from overwhelming the supply topology.
5. Cache extraction by normalized content plus the configuration/prompt/model identity that changes its meaning; checkpoint successful stages and preserve provenance during deduplication. Report actual LLM calls, token usage when provided, retries, stage timings, rejected-fact reasons and graph coverage. These are proposed engineering changes, not measured paper-derived speedups.
6. For a future research comparison, use a manually labeled, held-out corpus with difficult negatives: joint venture versus supply, planned versus completed partnership, acquisitions, subsidiaries, generic products, reversed buyer/supplier wording, snippets and repeated syndicated stories. Compare directed RE F1 and supported reachable subgraph quality, not graph size alone.

## 2. Huang et al.: ORLM

**Chenyu Huang, Zhengyang Tang, Shixi Hu, Ruoqing Jiang, Xin Zheng, Dongdong Ge, Benyou Wang and Zizhuo Wang. _ORLM: A Customizable Framework in Training Large Models for Automated Optimization Modeling_. arXiv:2405.17743v5, 4 April 2025.** Supplied PDF: `02-2405.17743v5-1-1-.pdf`.

This paper addresses translation of a natural-language optimization problem into a mathematical model and executable solver code. It is **not a knowledge graph extraction or supply relationship verification method**. OR-Instruct starts with 686 industry/textbook seed cases; combines scenario expansion, altered objectives/constraints, paraphrasing and alternate modeling techniques; filters unexecutable and duplicate cases; and obtains 32,481 training examples for supervised instruction tuning (PDF pp. 10–18). It uses COPT as the solver, with other solvers conceptually possible.

The authors estimate correctness around 70% for expansion data and 75% for augmentation data; about 39% of generated examples are removed each iteration (PDF pp. 15–16). Executable code is therefore not equivalent to a correct model. On IndustryOR, ORLM-LLaMA-3 reports 38.0% execution accuracy against 28.0% for its GPT-4 standard-prompt baseline, with other benchmark/model outcomes varying considerably (Table 2, PDF p. 19). These are optimization benchmark outcomes and cannot be used as anticipated Schain accuracy gains.

The paper explicitly separates code errors from model errors. In the analysis of modeling mistakes, 56.30% were low model completeness, 30.30% objective/constraint translation errors, and 13.40% semantic misunderstanding (Table 11, PDF p. 30). Correctness is checked by executing programs and matching objective values, which still cannot prove every constraint is correctly represented in all cases. Pass@8 results are an oracle-style existence check among candidates, not evidence that the system can select the correct candidate in deployment (PDF pp. 28–29).

Transfer to Schain: distinguish parse/schema validity from semantic entailment; introduce targeted hard examples and paraphrase invariance checks; keep validation data separate from prompt examples/training; compare alternatives on a fixed labeled set. A future procurement optimizer could consume verified graph facts plus real capacities, prices and constraints, but it belongs after data quality is established. Training ORLM, adding autonomous solver execution, or inventing missing quantitative inputs is unjustified for this static extraction release.

Source cautions: the text calls IndustryOR a 13-industry benchmark in the introduction but an eight-industry, 100-problem benchmark in Section 4.3 (PDF pp. 5 and 18); avoid claiming a precise industry count without reconciliation. Figure 3's textual problem and code use ship cost 130, while the displayed mathematical objective uses 80 (PDF p. 11). Do not copy example formulations without checking them.

## 3. Congès et al.: R-IO SUITE

**Aurélie Congès, Audrey Fertier, Nicolas Salatgé, Sébastien Rebière and Frederick Benaben. _R-IO SUITE: Integration of LLM-based AI into a knowledge management and model-driven based platform dedicated to crisis management_. DOI: https://doi.org/10.1007/s10270-024-01237-2.** The supplied cover lists _Software and Systems Modeling_ **2026**, vol. 25, pp. 25–50, while the HAL manuscript was deposited 6 January 2025. The filename's `2024` is not sufficient to override that bibliographic information. Source: `05-2024-SoSym-R-IO-AI.pdf`.

R-IO combines a predefined COSIMMA metamodel and model-driven reasoning with an LLM. It moves through data → information → decision → action; deterministic complex-event processing handles structured sensor inputs, while LLMs help interpret unstructured text/images and instantiate missing roles, functions, risks and objectives (PDF pp. 10–18). The model is **instantiated**, rather than autonomously replaced with an unconstrained generated ontology.

The useful architecture has two separations: deterministic processing for inputs/rules already understood, and LLM use for ambiguous unstructured inputs; persistent reference knowledge versus case-specific generated knowledge. For example, its invented roles/functions are not silently added to the permanent knowledge base for all future situations (PDF p. 22 and pp. 37–39). Humans validate generated instances and resolve conflicts before they enter the situation model (PDF pp. 26, 28). The prototype case study concerns an attempted intrusion into a research facility and gas-line damage (PDF pp. 28–36).

This is a qualitative demonstration, **not a quantitative extraction-accuracy or scalability benchmark**; the authors explicitly say the demonstration does not seek accuracy or technical scalability (PDF p. 28). LLM-supplied certainty/probability numbers in example JSON are not externally calibrated. The paper's role invention is inappropriate for Schain's factual supply graph: a plausible company capability must not become an asserted real supplier relationship.

Transfer to Schain: deterministic validation first; versioned frozen schema; separate durable accepted facts from suggestions/unresolved evidence; visible provenance; optional analyst review of uncertain merges. Keep speculative recommendations outside the authoritative factual layer.

## 4. AlMahri, Xu and Brintrup: direct agentic sequel

**Sara AlMahri, Liming Xu and Alexandra Brintrup. _Automating Supply Chain Disruption Monitoring via an Agentic AI Approach_. arXiv:2601.09680v1, 14 January 2026; manuscript compiled 15 January 2026.** Source: `06-2601.09680v1.pdf`.

The sequel assumes an **existing supply chain knowledge graph**. It does not solve extraction quality by adding seven agents. The seven roles are disruption monitoring, KG querying, product search, network visualization, risk management, CSCO reporting and alternative sourcing. They exchange structured JSON; LLMs interpret requests and orchestrate tools, while entity resolution, BFS, risk arithmetic and visualization use deterministic tools (PDF pp. 9–18).

Its evaluation graph has 6,596 nodes (6,495 companies and 101 countries) and 23,888 edges (17,341 suppliesTo and 6,547 locatedIn), with industries stored as company properties (PDF p. 18). Thirty **manually synthesized** scenarios cover Tesla, Mercedes-Benz and BMW, 10 each, with five disruption classes. Twenty-three scenarios have relevant paths and seven do not; the text calls these latter scenarios “false positive scenarios,” but they are negative test cases, not necessarily model errors (PDF p. 19).

Only four core agents receive quantitative evaluation; product search, visualization and alternative sourcing do not (PDF p. 19). Table 5 (PDF p. 21) reports mean F1 of 0.991 for disruption entity extraction, 0.980 for KG Query, 0.962 for Risk Manager and **0.899 for CSCO**. These are different tasks and datasets from the main extraction paper; they must not be compared directly with its NER/RE F1 as proof of improvement.

The narrative results are revealing: CSCO network-impact analysis has weighted score 0.486 ± 0.172, with **accuracy only 0.317 ± 0.251** (Table 7, PDF p. 24). Deterministic correct paths did not guarantee a correct prose interpretation. Schain should directly compute/display counts and paths and use source-linked statements rather than asking an LLM to invent summary numbers.

Methodological limits matter. Path evaluation uses order-insensitive node sets with Jaccard ≥ 0.9, even though supply paths are directed and ordered; a stronger implementation test must preserve edge order and direction. Risk scores are accepted within 0.1 tolerance (PDF pp. 20–21). The body uses five heuristic weights (35% breadth, 25% dependency ratio, 20% downstream criticality, 10% centrality and 10% depth) and configurable decision thresholds (PDF pp. 15, 20). The case-study figure instead uses **70% centrality + 30% inverse tier**, and shows 0.52/0.50 as high risk despite the body defining HIGH as ≥0.6 (Figure 13, PDF p. 34). This inconsistency prevents treating the manuscript as one unambiguous validated risk specification.

The reported 3.83-minute mean and $0.0836 per scenario are historical measurements, not Schain results; the multi-day human-response comparison is an external survey comparison, not a controlled like-for-like trial. Live news ingestion, temporal graph updates, concurrent stress tests and ablations of agentic orchestration versus deterministic tools are future work (PDF pp. 25–27, 36–38). The appendix supplies prompts only through Agent 4 in this version (PDF pp. 43–48), despite a broader statement about all prompts/tool definitions being included.

Practical transfers:

- Use deterministic, bounded, cycle-safe upstream BFS and tier layouts. Preserve complete ordered paths and identify the focal firm explicitly.
- Avoid pruning traversal to only affected-country nodes: a valid path often passes through unaffected countries before reaching an affected upstream company. The appendix's wording on filtering at each tier is ambiguous; follow the actual network question instead of copying that phrase.
- Cache shared lookups, batch graph traversals and expose truncation/coverage limits. Keep numeric summaries derived from the exact displayed graph.
- Product annotations require evidence for the **specific supplier–customer flow**. A company catalog does not establish which of its products it sells to every customer.
- A missing path means “no observed path in this graph up to this depth,” not proof that a supplier is unexposed or safe. The alternative-sourcing example overreaches when it treats absent Russian dependencies up to Tier 3 as a verified low-risk substitute (PDF pp. 35–36).
- Treat centrality and dependency counts as structural indicators, not calibrated failure probabilities. Supplier replacement requires commercial/operational information absent from this static graph. Automated replacement is outside this release.

## Synthesis for implementation priorities

The strongest near-term contribution is a more faithful and auditable extraction-and-visualization system: trustworthy direction/type/evidence checks, conservative canonical identity, deterministic upstream paths, bounded retrieval, resumable cached stages and an interface that lets users inspect why each edge exists. These changes can be implemented and regression-tested without claiming a new research method. The research claim, if desired later, needs a labeled comparison showing better supported RE precision/recall and useful supply-chain coverage at a measured token/runtime cost. More agents, more edges and more colorful graphs are not standalone evidence of improvement.
