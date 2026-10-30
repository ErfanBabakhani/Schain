# Review of agentic and risk-assessment sources

Scope: complete extracted text of project sources 08, 09, 10 and 12, including references and appendices, was read. Image-based prompts and the main image-based tables in 09, 10 and 12 were additionally inspected as rendered PDF pages. Page references below use **PDF page numbers, including the journal cover page**, rather than the printed article pagination. This review supports the current INGEST/EXPAND knowledge-graph project; it does not claim that the project contains a risk-optimization or inventory-control module.

## Main conclusion for this project

These papers support improving the *evidence and control around extraction* before adding autonomous decision-making. The most useful immediate transfers are source-grounded validation, bounded retry/revision, clear separation of extracted observations from computed indicators, persistent records of the source and run, and an interface that lets a person inspect the evidence behind an edge. Only the validation pattern is directly adjacent to the current extraction pipeline; the risk rankings and inventory negotiation systems solve different tasks and need different data.

No paper in this set establishes that increasing the number of LLM agents automatically increases KG correctness or reduces runtime. A critic using the same model can still agree with an unsupported statement. Deterministic schema/endpoint checks, evidence-span checks and explicit review states should remain authoritative gates; a model's self-rated confidence should remain a model assessment.

## 08. Tool-driven logistics-hub risk assessment

**Citation:** Yinzhu Quan, Yujia Xu, Guanlin Chen, Frederick Benaben and Benoit Montreuil (2025), *Leveraging Large Language Models for Risk Assessment in Hyperconnected Logistic Hub Network Deployment*. Proceedings of the IISE Annual Conference & Expo 2025; supplied arXiv version 2503.21115v1, dated 27 March 2025. [arXiv record](https://arxiv.org/abs/2503.21115).

**What the paper does.** It develops a GPT-4o/ReAct/LangGraph pipeline for assessing candidate logistics hubs. The agent can invoke hub-data, Wikipedia, finance, NOAA storm, news, traffic and notebook tools. The notebook handles batching, token limits and persistent summaries. The study uses Georgia as a testbed, with hub locations characterized by state, latitude and longitude. After exploratory tool selection, it chooses NOAA, Wikipedia and traffic information for the reported assessment. Daily assessments for 2024 are summarized yearly; standardized risk profiles are compared with cosine similarity and clustered. The paper reports two broad clusters, one with urban operational exposure such as congestion and another with stronger severe-weather exposure. See PDF pp. 2–5; the implementation and tool descriptions are on p. 3, the case study on pp. 4–5.

**Evidence level.** This is a short framework/case-study paper. Its outputs demonstrate an orchestration and visualization approach; it does not report a controlled entity/relation extraction benchmark, a calibrated probability benchmark, a human-labeled risk accuracy evaluation or an ablation establishing that each tool improves results. The conclusion explicitly places multi-agent parallelization in future work (p. 5). Accordingly, do not describe this paper as empirical proof that a multi-agent KG pipeline is more efficient.

**Transferable ideas, proposed adaptations.**

- Preserve the notebook principle as a persistent document/run cache and resumable processing state. This is an engineering adaptation of the paper's memory/batching approach, not a measured efficiency gain for this repository.
- Use deterministic tool selection for the present bounded workflow: parse files, extract candidate claims, validate them, merge them and export them. An unrestricted ReAct planner adds little to these known steps.
- Store source-specific information separately before combining it. If risk data are added later, a weather observation, a traffic measurement and an LLM interpretation must retain their own timestamps, units, provenance and uncertainty.
- A future risk-profile heatmap or geographic cluster view needs actual location/time/event data. Similarity in news mentions alone is not a verified operational risk profile.

**Defer:** live weather/traffic/finance orchestration, risk-profile clustering and geographic hub recommendations. The current static company KG does not supply the data or decision objective needed to implement these faithfully.

## 09. MARS: source-constrained feedback and revision

**Citation:** Yinzhu Quan, Zefang Liu, Frederick Benaben and Benoit Montreuil (2026), *Leveraging large language models to enhance multi-agent risk assessment in supply chain networks*. International Journal of Production Research, published online 9 February 2026. [DOI](https://doi.org/10.1080/00207543.2026.2619562).

**What the paper does.** MARS evaluates logistics-hub site candidates with six dimensions: distance, cost, traffic accidents, congestion, crime and severe weather. It identifies 376 candidate industrial properties in seven southeastern U.S. states using specified CoStar filters (p. 7). Structured indicators are computed directly; the LLM component maps unstructured weather narratives to severity ratings. The reported experiment processes 11,858 2023 weather narratives associated with those locations (pp. 9–10). It is not a supplier relation extractor.

The central agent workflow is:

1. `RiskAgent` assigns a 1–5 severity score and supporting explanation using a predefined rubric.
2. `FeedbackAgent` evaluates severity accuracy, narrative fidelity, reason clarity and overall confidence. Its prompt requires that judgment be based on the supplied narrative and reasoning, with no introduced external knowledge.
3. `RevisionAgent` revises deficient outputs; in the illustrated prompt, severity/confidence scores below 4 trigger reconsideration of severity, while fidelity/clarity scores below 4 trigger explanation revision. The revision must not introduce new information.
4. A bounded feedback loop ends after a specified number of iterations or convergence.

See pp. 7–12, particularly the actual prompt boxes on pp. 9, 11 and 12. The example matters: an initial explanation inflated a report of one tree blocking one lane into multiple road-blocking incidents; revision removed unsupported details and reduced the severity.

**Reported findings and their limits.**

| Finding | Exact meaning and limitation | PDF pointer |
|---|---|---|
| Overall severity-accuracy rating 4.05 → 4.15; narrative-fidelity rating 4.61 → 4.72 | These are 1–5 **LLM evaluator scores**, not percentages of independently verified correct events or KG edges. | p. 16, Tables 2–3 |
| Heavy-rain scores decline after revision | Revision is not uniformly beneficial across all categories. | p. 16 |
| First revision changes severity by MAE 0.125; later changes 0.090 and 0.069 | Diminishing output changes support bounded iteration; convergence alone does not establish truth. | pp. 20–21, Table 9 |
| Independent-run severity MAE 0.32–0.40 relative to a reference run | Low temperature and a stable workflow do not eliminate changes in individual outputs. | pp. 20–21, Table 10 |
| Human benchmark: 200 randomly sampled narratives, each rated by at least two authors | Human severity ratings provide a more independent reference than the automatic judge, but the sample is limited and raters are authors. | p. 22; p. 30 instructions |
| Pearson correlation 0.81 for revised LLM severity versus 0.51 for the rule baseline | Correlation with average human severity is not exact-match accuracy. The figure compares the revised LLM and a rule baseline; it does not isolate the revision effect against an unrevised LLM on this human benchmark. | p. 24, Figure 11 |

Model comparisons in Tables 4–7 (pp. 18–19) measure outcomes in this weather-rating task. The runtime/token results should not be imported as estimates for the user's news/KG pipeline. Although the paper claims stable behavior across backbones, the judges also vary in some comparisons; this limits claims about absolute cross-model accuracy.

For ranking, the paper first normalizes features within each target's candidate set, clips at 5th and 95th percentiles, combines severity-level probabilities using weighted RMS and combines six dimensions with equal-weight RMS (p. 29). The annual occurrence calculation assumes independence over days (pp. 14–15). Such scores are a constructed relative decision index, not a general probability that a supplier will fail. The weights, spatial association and independence assumptions require separate validation before application to another problem.

**Highest-value adaptations for the present code.**

- Validate every proposed edge against its own source text. Keep the exact evidence span, document ID/URL, publication date when available, and extraction metadata. This applies the paper's narrative-fidelity principle to relation extraction.
- Distinguish “evidence text located in the source” from “relation semantically supported by the source.” An exact substring match establishes provenance; it does not by itself establish relation direction, actuality or supplier status.
- Use a compact relation-review rubric: correct subject and object, explicit supplier/customer direction, actual versus proposed relation, temporal scope, and no added outside facts. Prefer deterministic checks where possible; reserve an optional LLM revision for failed/ambiguous cases.
- Preserve before/after claims and rejection reasons. Do not silently replace one unsupported claim with a more fluent unsupported claim.
- Add an explicit `needs_review`/rejected status for unresolved assertions. Do not average critic scores into a misleading calibrated probability.
- Measure the gain on a small manually labeled corpus from this project before enabling an extra critic call by default. This should include negative examples: co-mentions, prospective contracts, indirect chains and reversed supplier/customer phrasing.

These are proposed KG adaptations. The paper supports the *pattern* but has not tested them on the user's dataset.

## 10. Practitioner constraints and human roles

**Citation:** Nicolò Trifone, Fabrizio Dallari and Alexandra Brintrup (2026), *Drivers and constraints of agentic AI in supply chain management: an exploratory study*. International Journal of Production Research, published online 19 May 2026. [DOI](https://doi.org/10.1080/00207543.2026.2669878).

**Method.** This is an exploratory qualitative study of 19 practitioners with AI/digital-innovation and supply-chain experience. Interviews lasted approximately 60 minutes. The authors report saturation at interview 15 and four additional interviews, iterative coding, independent interpretation, source triangulation and member checking; 11 of 19 participants responded to the later synthesis check (pp. 13–19). It is not a deployed-system performance experiment.

**Findings.** The study organizes support into routine automation, descriptive analysis, predictive analysis and prescriptive analysis; adoption drivers and constraints are categorized through the technology–organization–environment framework. It reports 11 drivers and 14 constraints. Table 4, visually inspected on PDF p. 17, gives the largest driver shares as time efficiency 32%, cost efficiency 21% and process visibility 10%; the largest constraint share is data quality/quantity 26%, followed by algorithm bias 15%. These percentages reflect normalized mentions and perceived salience, not percentages of firms benefiting or measured improvement magnitudes (method, p. 16).

The driver-category shares in Table 5 are descriptive analysis 36%, prescriptive analysis 29%, routine automation 24% and predictive analysis 11% (p. 18). Interviewees often already had tools for routine or predictive tasks, which reduced perceived incremental value. The study highlights validation, domain expertise, interpretation, reflective use and interpersonal mediation as complementary human roles (pp. 24–32). No interviewee supported fully delegating important supply-chain decisions to an autonomous decision-maker in the way discussed by the authors (pp. 24–26, 30–32).

**Limitations and caution.** The authors explicitly state that their results concern anticipated practitioner perceptions, not measured operational outcomes, and are not statistically representative of a larger population (p. 34). The framework assumes moderate-to-high data maturity and decisions with enough latency for human oversight (p. 32). Thus, this paper does not establish that agents will deliver any particular cost or accuracy improvement. Its text also contains an internal inconsistency: p. 29 says process-order automation did not surface, whereas p. 22 and the rendered Tables 5–6 on pp. 18–19 include it. Avoid leaning on that exclusion claim.

**Practical implication for the code and graph, as a proposed adaptation.** Make this project a clear descriptive evidence tool first. The graph should answer which entities/relations were extracted, where each claim came from, and which need review. Show document provenance, uncertain or conflicting assertions and source coverage; allow focus on one company and its immediate neighborhood. These interface decisions support the paper's practitioner emphasis on visibility, usability and validation, but are our design choices rather than a graph UI evaluated in the article.

Defer automated purchasing, supplier replacement, contracting and autonomous disruption responses. Those require actual transaction/contract data, operational constraints and a validated decision model. The existing KG alone cannot establish an implementable procurement action.

## 12. Inventory consensus and negotiation

**Citation:** Valeria Jannelli, Stefan Schöpf, Matthias Bickel, Torbjørn Netland and Alexandra Brintrup (2025), *Agentic LLMs in the supply chain: towards autonomous multi-agent consensus-seeking*. International Journal of Production Research, published online 21 December 2025. [DOI](https://doi.org/10.1080/00207543.2025.2604311). Use the supplied published version's 2025 date; earlier references in other papers refer to its 2024 preprint.

**What the paper does.** This is a three-echelon inventory-management simulation, not knowledge-graph construction. It compares standalone agents, neighbor information sharing, tool-supported agents, and neighbor negotiation around tool output. Agents have partial observability, memory of 10 observations, and three negotiation loops per adjacent pair. The experiments use Gemini 1.5 Flash/Pro, 200 simulation steps, lead time two steps, maximum order 100, temperature 0.1, and Merton jump-diffusion customer demand. Tools are linear-regression demand prediction over the previous 30 observations for cost experiments and a deliberately adapted EOQ-style calculation representing individual incentives for bullwhip experiments. See pp. 8–15 and p. 31.

**Actual results.** The supplied Appendix 4 cost table reports a direct forecasting-tool baseline of 16,645 cumulative simulated cost; Flash negotiation plus tool gives 12,790 and Pro negotiation plus tool gives 15,393. Pro information sharing plus tool gives 13,431, which is *better* than Pro negotiation in that table (p. 30). Therefore, more elaborate negotiation is not monotonically better.

For the paper's global bullwhip indicator, Table 2 reports Pro negotiation plus tool 0.25827 versus 0.76486 for Pro information sharing plus tool; Flash negotiation plus tool 0.50744 versus 0.75923 with Flash information sharing plus tool. These correspond to the authors' reported 66.2% and 33.2% reductions for those particular comparisons. However, standalone Flash is 0.36899, lower than Flash negotiation's 0.50744, so the claim of universal gradual improvement does not hold across every table entry (p. 17). Cost and bullwhip are also separate objectives: lower bullwhip does not ensure lower cost.

**Source-quality and evaluation limitations.** The study explicitly limits generalization to its three-tier simulation and Gemini models, reports prompt sensitivity, and acknowledges that broader network structures and reliable autonomous real-world operation remain future work (pp. 19–23). The numerical metric is an aggregate of coefficients of variation; it should not be silently substituted for another conventional bullwhip definition. There is also a sum-versus-product inconsistency between the displayed global formula and adjacent explanatory text on p. 9. Some prose cost-reduction percentages on p. 15 do not match the configuration labels in the appendix table, so use table values if discussing quantitative results. PDF p. 19 also misattributes a discussion of the Chen baseline to Campbell et al. (2002), whose reference is *Deep Blue*; do not repeat that citation error.

**Useful lessons for the present project.**

- Put non-negotiable constraints in code as well as prompts. The paper uses a hard order-quantity bound as a backstop (pp. 14, 19); the corresponding KG constraints are valid endpoints, allowed relation types, stable IDs, evidence requirements and bounded expansion.
- Keep source observation state separate from generated conclusions and tool-computed values. The simulation's perception/memory/action separation is a useful software boundary even without autonomous agents.
- Do not choose a larger model, extra tool or additional conversation loop solely because it sounds stronger. Compare quality, latency and cost on the actual task; the paper itself shows model/prompt/tool interactions and nonmonotonic outcomes.
- A source-focused neighborhood view is appropriate for partial information. A missing edge should mean “not established by the available evidence,” not “no relationship exists.”

**Defer:** supplier agents negotiating quantities, inventory-control policies, bullwhip metrics and cost optimization. A news-derived static KG has no required inventory, lead-time, price, demand or capacity state and should not be presented as a simulator or digital twin.

## Proposed implementation priorities

| Priority | Proposed feature | Motivation from these sources | Verification needed in this project |
|---|---|---|---|
| Now | Evidence text and provenance per assertion/edge | MARS narrative fidelity and source-only revision, pp. 9–12 | Evidence span is retrievable; assertion direction and actuality checked separately |
| Now | Schema/endpoint/range validation before graph merge | Hard constraints alongside prompts in consensus paper, pp. 14, 19 | Invalid nodes, relations and dangling edges cannot silently enter the exported graph |
| Now | Persistent cache, run metadata, bounded retries and expansion | 08 notebook/token batching, p. 3; 09 diminishing returns, pp. 20–21 | Repeated documents avoid unnecessary calls; failures do not masquerade as successful extraction |
| Now | Company focus, neighborhood filtering and evidence detail panel | Practitioner descriptive visibility/validation, 10 pp. 18, 23–25 | Filters preserve relation direction and source traceability; no invented risk score |
| Next, optional | One source-constrained semantic review for ambiguous claims | MARS feedback/revision pattern | Manual benchmark must show incremental benefit over deterministic checks and initial extraction |
| Later, separate module | Risk-event enrichment and comparative risk views | 08/09 logistics-hub demonstrations | Genuine spatiotemporal events, units, mapping rules and independent validation |
| Later, separate project | Inventory/supplier negotiation and operational decisions | 12 simulation proof of concept | Demand, inventory, constraints, objectives, simulation validation and an appropriate deployment process |

The review supports these design priorities but supplies **no measured accuracy, cost reduction or runtime improvement for the uploaded code**. Any such performance claims require a before/after evaluation on the same source corpus and model configuration.
