# Framework and decision-support papers: source-grounded review

This review covers project sources 03, 04, 07 and 11. Page references below are **1-based PDF pages**, including publisher cover pages where present. The papers' proposed architectures and reported experiments are distinguished from our implementation proposals. The current project's README and `docs/SOURCE_REPUTATION_AND_VERIFICATION.md` were also read to avoid recommending features as new when the code already advertises them.

## 04 — Large language models in supply chain management: a systematic literature review and application framework

**Bibliographic identity.** Zhe Song, Ying Xie, Lichao Yang and Yifan Zhao (2026), *International Journal of Production Research*, published online 13 March 2026, DOI `10.1080/00207543.2026.2641103`. Supplied PDF has 42 pages including the publisher cover. Full extracted text, references and appendices read.

**What the paper actually does.** It is a systematic literature review and conceptual application framework, not a new extraction algorithm or a benchmark of supply-chain graph construction. Its framework combines Dynamic Capabilities Theory—sensing, seizing, transforming—with Context–Mechanism–Outcome reasoning and the five SCOR processes used in the paper: Plan, Source, Make, Deliver, Return (PDF pp. 2–3, 6–7, 16–18). The five model families are explicitly described as conceptual categories rather than a definitive architecture taxonomy (Table 1, PDF p. 5). The search was performed in April 2025 across Scopus and Web of Science; the narrative reports 36 database-selected studies and another 25 from snowballing (PDF pp. 7–8). Thematic analysis groups applications into forecasting/simulation, semantic automation, and cognitive coordination (PDF pp. 10–11).

**Relevant supported findings and recommendations from the authors.**

- Sourcing is directly relevant to the current project: supplier discovery, multilingual documents, upstream traceability, fragmented supplier information and semantic matching are discussed under the Source process (PDF pp. 14–15, 21).
- The authors explicitly identify future traceable supplier-intelligence pipelines containing **sources, timestamps, confidence and contradiction handling** (PDF p. 21). This is a research direction, not a demonstrated implementation in this paper.
- Explainability and provenance are recurring requirements. RAG and graph integration are proposed to ground outputs in verifiable sources; domain-specific knowledge graphs and evaluation benchmarks are future research directions (PDF pp. 20–23).
- The review warns that many applications assume idealized or high-quality data, inadequately quantify uncertainty, lack robust comparisons with conventional/hybrid methods, and remain proof-of-concept systems (PDF pp. 14–16). It calls for contextual evaluation and integration with existing systems rather than only demonstrations of technical feasibility.
- The practical pathway is problem first, capability/mechanism second, outcome third (PDF pp. 18–20). The illustrated warehouse-simulation case is borrowed from another study rather than a new empirical validation of the review's framework.

**Limitations and review observations.**

1. The framework does not prove that adding agents, RAG, confidence scoring, or a graph increases the accuracy or efficiency of this repository. Its thematic conclusions supply design motivation.
2. The review covers mixed technologies, including non-language-model Transformers and conventional NLP. Appendix E itself lists ViT, Swin, Transformer forecasting and LSTM-based work (PDF pp. 41–42). Results from that heterogeneous literature should not be treated as a measured benefit of generative LLM extraction.
3. The screening narrative contains an arithmetic inconsistency: after 262 retrieved full texts and 187 exclusions, it states that 93 remained, although 262 − 187 = 75 (PDF p. 8). The table of methodological categories totals 61, consistent with the stated 36 plus 25 final sources (PDF p. 10), but the flow cannot be reproduced from its prose alone. Avoid using the inconsistent intermediate numbers as evidence.
4. Appendix D provides many model-specific numeric and capability assertions without a clearly specified common benchmark for each cell (PDF pp. 36–40). It is not a sound basis for choosing deployment model versions, context lengths or hardware requirements. Verify those against provider/model documentation when deploying.

**Our implementation proposals motivated by this paper.** These are our engineering adaptations, not claimed paper contributions:

- Improve the existing evidence record so an exported edge can be inspected against its exact document ID, URL, evidence text, evidence location where available, extraction/run identity, and document publication/retrieval timestamps. Keep absent dates unknown. Distinguish publication/retrieval dates from a relation's effective date; one does not establish the other.
- Preserve contradictory or qualified claims as evidence metadata when the extraction supports them; do not silently convert a future agreement, rumor, or terminated relationship into an unconditional present supply relation.
- Keep the repository's existing policy that one explicit supporting document can be sufficient. The paper does **not** justify reinstating an independent-source count gate or rejecting evidence purely because of publisher reputation. Provenance and confidence annotations should help inspection without becoming invented factual probabilities.
- Present a supplier-oriented graph with a clear focal company, upstream levels, relation-type filters and a selected-edge evidence panel. This is a UI proposal; the paper gives traceability motivation but does not evaluate a particular graph layout.
- Evaluate relation precision, direction accuracy, entity merges, evidence anchoring and upstream expansion on an annotated sample. Compare baseline versus each added component, and report runtime/calls/tokens separately. Neither a dense graph nor a passing unit-test suite establishes extraction quality.

## 07 — LLM4SCM: a framework for intelligent supply chain decisions

**Bibliographic identity.** Urvi Chintukumar Sukharamwala and Daehan Won (2026), *International Journal of Production Research*, published online 5 June 2026, DOI `10.1080/00207543.2026.2682360`. Supplied PDF has 20 pages including the cover. Full extracted text, references and appendix read.

**What the paper actually does.** It couples an LLM interface with a mathematical optimization backend. The three stages are natural-language query processing, translation/integration into a solver-executable model, and solver-grounded explanation/refinement (PDF pp. 2–3, 5–10). The prototype uses Python/Streamlit/AutoGen, an OpenAI API model, and Gurobi 11.0.0 (PDF p. 10). The experimental problem is a stylized supply-network mixed-integer program with two factories, six candidate depots and six customers; its data and model are drawn from Williams' textbook example (PDF pp. 17, 19–20).

**Important distinction between conceptual method and implemented prototype.** The paper presents attention/NER/CRF equations and conceptual prompt selection/reasoning operators, but explicitly says its actual query-to-structure step uses **prompted structured output from a pretrained LLM**. It does not train a standalone Transformer–CRF tagger (PDF pp. 6–8). Similarly, natural-language reasoning does not perform the authoritative calculation: execution, checking and structured artifact extraction are performed by the optimization backend (PDF p. 9).

**Relevant supported findings.**

- The system keeps solver status, objective value, decision-variable assignments and before/after differences as structured artifacts for explanation (PDF p. 9). This separation is the most relevant transferable architecture for our KG.
- Its reported benchmark contains 30 what-if scenarios. Evaluation includes semantic similarity of explanations, feasibility classification, numeric objective error, decision-variable precision/recall, field coverage and latency (PDF pp. 10–13). Those results concern this particular optimization interface, not knowledge-graph construction.
- The authors report that even when the backend solved correctly, generated prose could give the wrong numeric result. One demonstration reports 187,500 in LLM prose versus the solver-grounded 208,500 (Table 6, PDF p. 14; discussion PDF pp. 15–16).
- Semantically similar queries could produce different behavior: replacing “consequences” with “implications” in a depot-closure question changed an error into a successful response (PDF p. 15). Controlled schemas and query normalization are therefore more defensible than relying on fluent output alone.
- The authors explicitly identify unsafe code parsing, unbounded-model handling, occasional session restarts, hallucinations, general-purpose model dependence and the narrow benchmark as limitations (PDF pp. 15–16). The 30-scenario evaluation data are available from the authors upon request, rather than included as a complete public benchmark in the supplied PDF (PDF p. 17).

**Metrics should be treated carefully.** The paper's best reported feasibility accuracy is 92%, with objective relative error 7.2%, variable precision 0.84 and recall 0.89 (Table 4, PDF p. 12). These are the authors' reported values, not reproducible findings from our repository. As a review observation, several percentages in that table cannot result from a single integer correct-count divided by 30 under the metric definition on PDF p. 11; an averaging or repetition scheme is not specified there. Table 3 and the accompanying prose differ on the leading BERTScore (84.84 vs 84.94, PDF p. 11), and Table 5's reported end-to-end means are smaller than the LLM-time means (PDF p. 13) without a clear timing explanation. Do not use these figures to predict gains in the KG or as target acceptance thresholds.

**Our implementation proposals motivated by this paper.**

- Treat validated facts and the graph backend as authoritative. Compute node count, edge count, tier distance, relation direction, document support count and path results deterministically. The frontend should display those values directly, without asking an LLM to reproduce them.
- Keep LLM-generated labels/descriptions separate from identifiers and validated graph properties. Human-readable prose should never overwrite the supplier/customer endpoint or transform graph topology.
- For future graph Q&A, allow typed query intents backed by fixed parameterized queries or graph algorithms: “direct suppliers”, “upstream suppliers within depth k”, “evidence for this edge”, “shared upstream supplier”. Return edge/document IDs with the answer. Unrestricted LLM-generated Cypher or Python is unnecessary for these operations.
- If simulation or sourcing optimization is added later, keep scenario edges and decisions in a separate derived layer. A news-derived relation does not provide capacities, reliable costs, lead times or demand. Only run optimization once those inputs are supplied with units and provenance; solver mathematics cannot repair missing business data.
- Add a reproducible benchmark manifest and machine-readable quality report. Measure correctness of factual fields independently of natural-language semantic similarity. High text similarity can coexist with a wrong cost or wrong supply direction, as the paper's own examples demonstrate.

## Practical synthesis for the current repository

The existing project already has retrieval, extraction, resolution, ontology/evidence gates, durable checkpoints, upstream expansion, Neo4j storage and a semantic graph export. The reviewed framework literature supports **strengthening these boundaries**, rather than replacing the pipeline with a large multi-agent architecture.

| Priority | Proposed engineering direction | What supports it | What remains to be demonstrated |
|---|---|---|---|
| Immediate | Preserve exact evidence/provenance through aggregation and expose it on edge selection | Song et al., PDF pp. 21–23 | That anchoring/aggregation fixes retain evidence and improve analyst inspection |
| Immediate | Compute topology, tiers and graph statistics deterministically | LLM4SCM's backend/prose separation, PDF pp. 9, 15–16 | Correctness on the repository's graph fixtures and representative real outputs |
| Immediate | Evaluate false entity merges, wrong direction, unsupported edges and repeated work separately | Song et al.'s evaluation gaps, PDF pp. 14–16; LLM4SCM's separate metric dimensions, PDF pp. 11–13 | Gold-label extraction quality and measured resource consumption |
| Near term | Read-only typed graph queries with explicit evidence IDs | LLM4SCM's constrained structured-update principle, PDF pp. 6–9 | Reliable intent mapping and answer grounding on a held-out question set |
| Future | Cost/capacity planning or risk simulation connected to the KG | LLM4SCM and humanitarian optimization are complementary decision layers | Appropriate verified numeric inputs, formulation and domain-specific validation |

The selected papers do not establish that duplicate publication counts are independent corroboration, that source reputation is a factual probability, or that higher node count means better visibility. They also do not justify using the supplier tier as an industry-role label. These choices should remain governed by the project's semantic definition and tested against explicit evidence.

## Source 03 — capability sequencing rather than a new KG algorithm

**Citation.** Mario Henrique Callefi, Elza Magalhães, Léony Luis Lopes Negrão, Matthias Thürer, Janaina Siegler, Ivan Tomašević and Moacir Godinho Filho (2026), “Building a pathway of GenAI-enabled capabilities for supply chain management,” *International Journal of Production Research*. Published online 25 February 2026. DOI: https://doi.org/10.1080/00207543.2026.2636664. Supplied file: `03-Building-a-pathway-of-GenAI-enabled-capabilities-for-supply-chain-management.pdf`.

### What the authors actually did

- They combine an SLR, Fuzzy Delphi, Interpretive Structural Modeling (ISM), Fuzzy MICMAC, expert validation, and secondary case analysis. The SLR searched Engineering Village, Web of Science and Scopus, screened 1,092 unique records, and reports 55 included English-language articles (PDF p.6).
- The Fuzzy Delphi stage retained 20 qualified experts from 26 responses; the subsequent ISM panel had 26 experts. Participants had relevant supply-chain experience and reported practical or academic GenAI exposure (PDF pp.7–8).
- Fourteen initial capabilities passed the quantitative relevance threshold. Qualitative feedback removed two overlapping constructs—intelligent resource reconfiguration and proactive risk management—leaving 12 capabilities for the structural analysis. Their removal was for conceptual redundancy, not because the concepts lacked importance (PDF p.11).
- ISM establishes an expert-judged influence hierarchy. Fuzzy MICMAC adds strength/dependence analysis based on expert agreement; it is not a fitted causal model from observed operational outcomes (PDF pp.8–10).
- The resulting four-stage pathway is: **data consolidation → operational intelligence → adaptability/resilience → differentiation/customer value**. Data integration and interoperability form the foundation; document analysis, visibility and knowledge retention link the foundation to more dependent capabilities (PDF pp.11–17).
- The paper maps publicly documented DHL Supply Chain and Walmart initiatives onto that pathway. For DHL, the account includes converting heterogeneous RFQs into an internal logistics ontology before downstream solution design. These are secondary company/trade-press case accounts, not a controlled implementation experiment by the authors (PDF pp.10, 19–22).

### Useful evidence for this project

1. **Prioritise a coherent semantic data foundation.** The foundational capability C4 means connecting, translating and harmonising heterogeneous data. C8 covers document interpretation/extraction and C11 covers organised, reusable knowledge (Table 6, PDF pp.12–13; stage discussion, PDF pp.14–16). This directly supports improving the KG schema, company identity consistency, provenance and repeatable ingestion before adding more autonomous decision functions.
2. **Use human effort for validation and exceptions.** Proposition 2 explicitly describes moving synthesis to GenAI while retaining human verification, boundary checks and accountable decisions (PDF p.22). A review queue for uncertain aliases or unsupported supplier relations is a defensible adaptation.
3. **Treat scenarios as a later capability.** The paper discusses counterfactual scenario simulation as a possible route to anticipatory resilience (PDF pp.22–23). This supports a later scenario layer that is clearly distinguished from extracted observations; it does not justify merging imagined edges into the factual graph.
4. **Evaluate capabilities before claiming operational benefits.** The paper recommends early attention to interoperability and data governance before advanced analytics/simulation (PDF pp.24–25). For the current implementation, directly measurable ingestion and graph-quality outcomes are more appropriate than claims of business resilience improvement.

### Limits and critical reading

- The authors explicitly acknowledge that relationships are based on expert judgment rather than quantitative performance measurement, and call for longitudinal and cross-sector empirical tests of their propositions (PDF p.25). The stage order is a research-informed roadmap, not a universal proven causal law.
- Reported company productivity/scale examples are secondary-source performance proxies (PDF pp.21–22). They must not be reported as benchmarks for Schain or as evidence that a particular caching or graph algorithm produces the same gains.
- The paper does not supply an entity-resolution algorithm, a supplier-link extraction benchmark, an extraction prompt, a Neo4j ingestion implementation or a graph layout benchmark. Those are engineering design choices to evaluate separately.
- Small internal presentation inconsistencies argue against mechanically copying its mathematical details: the displayed Delphi aggregation/defuzzification equations and reported table scores should be checked against the cited method before anyone reimplements that analysis (PDF pp.7, 13). This does not prevent using its high-level capability sequencing.
- **Additional appendix reproducibility observation:** the body describes iterative max–min stabilisation to convergence (PDF pp.9–10), but the provided Appendix B2 MATLAB program performs a single `C = maxMinCompose(A, B)` between the binary and fuzzy direct matrices and prints the result. It provides no convergence loop or subsequent driving/dependence computation (PDF pp.32–33). Therefore the appendix does not fully implement the procedure described in the body. Appendix A supplies its ISM matrices and a worked direct/transitive-link example (PDF pp.31–32); Appendix C supplies selected company/trade-source excerpts mapped to capabilities (PDF pp.33–34), reinforcing that these cases are secondary interpretation rather than independently measured causal results. Expert-profile tables are in the main body, PDF pp.7–8, and were reviewed.

### Proposed implementation adaptations — our design, not paper results

| Priority | Adaptation for Schain | Why it fits this paper | Scope boundary |
|---|---|---|---|
| Now | One canonical entity identity and explicit supplier-to-customer direction | Makes the semantic foundation coherent | Preserve aliases and ambiguous candidates instead of assuming every similar name is identical |
| Now | Save each extracted relationship's document ID, source URL, evidence span and extraction configuration | Makes organisational knowledge inspectable and reusable | An evidence span is provenance, not automatic proof of correctness |
| Now | Deterministic validation and an exceptions report before accepted-edge insertion | Matches the paper's supervisory validation model | Do not equate a model-generated confidence score with calibrated probability |
| Now | Incremental ingestion, content hashes and cache keys containing model/prompt/schema versions | Reduces repeated data consolidation and extraction work | Engineering proposal; no efficiency multiplier is established by this paper |
| Now | Focal-company view with explicit tier, source panel and relationship filters | Converts a coherent data foundation into usable visibility | Display observed connectivity; do not imply completeness of the real supply network |
| Later | Read-only scenario subgraphs and structured questions over validated graph evidence | A staged move toward decision support/scenario capability | Synthetic edges remain distinct from extracted facts; predictive claims need validation |

## Source 11 — evidence-grounded parameter extraction plus optimisation

**Citation.** Hossein Baharmand and Behzad Behdani (2026), “Generative artificial intelligence and mathematical modelling for eco-friendly humanitarian supply chains,” *International Journal of Production Research*. Published online 26 March 2026. DOI: https://doi.org/10.1080/00207543.2026.2646335. Supplied file: `11-Generative-artificial-intelligence-and-mathematical-modelling-for-eco-friendly-humanitarian-supply-chains-1-.pdf`.

### What the authors actually did

- The study develops a humanitarian relief-flow decision support model combining qualitative evidence, local GenAI-assisted parameter extraction and mathematical optimisation. Its application is **the relief phase after the immediate emergency**, not general commercial supplier discovery (PDF pp.2–4, 9–11).
- Fourteen semi-structured interviews with experts from eight humanitarian organisations were conducted between December 2022 and February 2023; the qualitative corpus also included 48 operational reports. Content analysis used NVivo, iterative coding, partial second coding and participant feedback (PDF pp.7–8).
- The local GenAI component extracts quantities and contextual factors for packaging reuse, repurposing and recycling. The authors describe source-referenced answers, checking outputs against published empirical data/interviews, flagging numerical discrepancies above 10% for manual review, and repeating queries up to 20 times for consistency (PDF pp.8–9).
- They explicitly say that only **unique events or quantities** should be aggregated, with cross-referencing and expert review to avoid double counting (PDF p.9). That is particularly relevant to a news-based knowledge graph, where one event may appear in multiple articles.
- The mathematical model has four objectives: cost, uncovered demand, packaging-waste/environmental impact and weighted absolute demand-delivery deviation, labelled fairness. It uses constrained reuse, repurpose and recycling ratios and is solved as a multi-objective MILP using AUGMECON2 in Pyomo with Gurobi (PDF pp.10–13).
- The case has five suppliers, three staging areas and five demand locations, with tarpaulins and kitchen items and two packaging types. Two flood scenarios are used: moderate/severe with 0.6/0.4 probabilities determined through consultation and historical context (PDF p.14).
- The paper reports 72 unique solutions and illustrates the trade-offs in that case (PDF p.20). It reports some solutions with over 50% lower modelled environmental impact for under 10% higher costs (PDF p.22). These are **case-specific optimisation findings under the paper's assumptions**, not gains in LLM extraction accuracy, KG quality, general environmental performance or Schain runtime.

### Useful evidence for this project

1. **Preserve document attribution through extraction.** Source 11's prompts ask for referenced source information (PDF p.9), and its collection appendix preserves title, source, date and source ID in the text corpus (PDF pp.30–34). For Schain, this translates naturally to document-backed evidence records rather than bare supplier triples.
2. **Deduplicate events, not only URLs.** The unique-event rule on PDF p.9 supports distinguishing independent corroboration from republication. Canonical URLs and content hashes are useful first steps, but two different URLs can still report the same underlying announcement.
3. **Keep quantitative claims typed.** The extraction task involves quantities, location and contextual feasibility. If quantities are added to Schain, store the original value/unit/time period and the normalised value separately, so tonnes, units, monetary amounts and percentage shares cannot be accidentally combined.
4. **Separate observations from assumptions.** The paper uses the nearest location's ratio when local information is missing (PDF p.15). That is an explicit proxy assumption, not an observation. A KG should label imputed/proxy/synthetic attributes and keep them out of the accepted factual evidence count.
5. **Keep LLM synthesis upstream of deterministic analysis.** The demonstrated integration uses extracted information as input to a structured optimisation model, with human checks. This is useful support for using an LLM for bounded extraction and conventional algorithms for graph traversal, reachability and later optimisation, rather than allowing free-form model text to become an operational calculation.

### Limits, reproducibility and modelling cautions

- This is not a supplier-relation extraction benchmark. It reports no held-out entity-resolution or supplier-edge precision/recall evaluation. The 10% discrepancy trigger is a study-specific validation rule, not a universally validated confidence threshold (PDF p.9).
- The body describes a local model as fine-tuned. Appendix 1 documents fetching and translating text, starting an unpinned PrivateGPT Docker image, and uploading documents. It does **not** specify base-model identity, training hyperparameters, labelled training data, fine-tuning procedure or a held-out validation set (PDF pp.30–34). Therefore the supplied appendix does not independently establish reproducible parameter fine-tuning; describe it conservatively as a local document-based GenAI assistant with under-specified model details.
- Repeating a query up to 20 times may test output stability, but outputs from the same source/model are not independent evidence of truth. It also increases cost. A better proposed Schain default is one validated extraction, deterministic rejection checks, bounded retries for transient/schema errors, and targeted review of disagreements.
- The environmental objective concerns packaging waste and omits transport emissions. Reuse/repurpose additional costs are assumed negligible; transportation capacities/time windows are omitted; parameters stay fixed until a rerun (PDF pp.10–12, 22, 25). These assumptions prevent transferring the reported environmental percentage to BMW/commercial supply networks.
- The fairness objective is weighted absolute unmet/overdelivered quantities, not a guarantee of equal coverage ratios or a maximin fairness measure. The authors discuss alternative fairness definitions and context dependence (PDF pp.11–12, 22, 25).
- **Reviewer observation about the printed equations:** before reusing the optimisation model, audit material conservation and demand bounds. On PDF p.13, constraint (9) uses `reuse + repurpose + recycle + landfill <= available packaging`, which, as printed, permits an unassigned remainder; the displayed uncovered-demand objective is `demand - shipments`, with no displayed shipment upper bound by demand. Those expressions should not be copied into a new solver without resolving their intended meaning against the authors' implementation. This is an inspection finding about the presented equations, not a claim that their unavailable Python code necessarily contains the same issue.
- The paper's example recycling fraction uses `recycled / (recycled + supplied)` (PDF p.9). For a new application, independently define the denominator and material accounting; do not silently treat that expression as the conventional fraction of supplied material recycled.
- Full model code/data are available only on reasonable request, according to the data-availability statement (PDF p.25); the supplied PDF alone does not support independently reproducing all reported numerical results.

### Proposed implementation adaptations — our design, not paper results

| Priority | Adaptation for Schain | Concrete behaviour |
|---|---|---|
| Now | Document/evidence identity distinct from company/relationship identity | Multiple sources support one directed supplier edge without duplicating its endpoints |
| Now | Evidence deduplication and count semantics | Display unique documents and, when known, source/event groups; never count repeated LLM responses as corroboration. These are descriptive metadata, not an acceptance gate: one explicit, supported source remains sufficient for an accepted relation |
| Now | Source-evidence verification | Require the cited passage to exist in the retrieved text and retain its location/context; report missing evidence |
| Now | Explicit uncertainty and review output | Preserve unsupported/ambiguous relations outside the default accepted graph instead of turning missing information into a confident fact |
| Now | Reproducibility manifest and content-sensitive cache | Record configuration, schema/prompt version, source hashes and counters; allow reruns without paying again for unchanged extraction work |
| Later | Typed quantity and scenario objects | Separate supplied facts from proxy estimates, including units, dates and the method/assumption behind each derived value |
| Later | Deterministic scenario or optimisation layer | Operate on validated graph inputs, define material-flow conservation explicitly, and show trade-offs instead of inventing a single optimum |

## Shared recommendation

Both sources favour strengthening the information foundation before advanced decision support. For this code upgrade, provenance, semantic consistency, deduplication, deterministic graph operations and an inspectable focal-company visualisation are the highest-value transfers. Neither source alone justifies adding a large multi-agent architecture, claiming calibrated disruption probabilities, or claiming improved extraction accuracy without an annotated comparison dataset.
