# SChain v0.3.0 — audit and implementation report

Audit baseline: the actual supplied v0.2.0 checkout, preserved before editing. Scope: backend, configuration, runtime scripts and offline tests. Frontend files are unchanged. No production web/LLM expansion experiment was run.

## 1. Repository architecture

SChain is a Python CLI with INGEST and EXPAND sharing one document-processing pipeline. SQLite stores document stages, search checkpoints, canonical resolution and the expansion frontier. JSON stores extraction checkpoints; Parquet stores normalized documents, evidence-level facts, rejected candidates, resolution and edge support. Optional Neo4j writes merge accepted graph records. EXPORT builds a standalone JSON graph from persisted artifacts without invoking an LLM or Neo4j.

| Layer | Main responsibilities and locations |
|---|---|
| Configuration and entry point | `config/*.yaml`, `schain/config.py`, `schain/__main__.py`; include resolution, explicit CLI overrides, model/runtime/storage policy. |
| Retrieval | `schain/sources/`; Google News, GDELT, general web and Wikipedia; publisher recovery, text extraction, caching, origin filtering, ranking and manifests. |
| Document pipeline | `schain/documents.py`, `pipeline.py`, `checkpoint.py`, `kg/extract.py`; source normalization, bounded chunks, cached extraction and resumable stages. |
| Semantics and identity | `ontology.py`, `kg/relation_evidence.py`, `kg/validate.py`, `kg/resolve.py`; ontology, source-quote grounding, entailment, direction and conservative identity. |
| Expansion | `expansion/loop.py`, `frontier.py`, `budget.py`, `state.py`; typed upstream promotion, cumulative spend, tier limits and beam scheduling. |
| Evidence and export | `kg/facts.py`, `kg/support.py`, `kg/writer.py`, `graph_export.py`; evidence records, semantic-edge support, optional graph writes and shortest-tier export. |

The audit covered all production modules, configurations, runtime/Colab material and tests. Frontend code was inspected read-only. The preceding literature review remains included in the project; these changes are justified by current-code defects and offline reproductions. Neither the source papers nor synthetic examples establish current provider yield, 4B-model precision or GPU throughput.

## 2. Disposition of the 18 hypotheses

Severity describes the pre-edit defect. A rejected hypothesis is not changed merely because it was proposed. Function names below identify the audited responsibility; accompanying audit notes retain pre-edit details.

| # | Verdict | Severity | Evidence, practical effect and decision |
|---|---|---|---|
| 1 | Confirmed | High | `sources/retrieve.py:retrieve_async/_add_full_text` fetched the Google wrapper before `enrich_search_results` recovered the publisher URL. This wasted fetches and increased snippet fallback. Reordered and bounded recovery/fetch. |
| 2 | Confirmed | High | `sources/ranking.py:rank_discovery_candidate` did not distinguish upstream from downstream; ambiguous supply agreements received similar priority. Added explicit origin-relative classes before LLM allocation. |
| 3 | Partially confirmed | Medium | `config/expansion.yaml` already had upstream queries, but eight templates per tier were redundant and deeper queries assumed industrial roles. Reduced to five complementary templates without assuming distance means mining. |
| 4 | Partially confirmed | High | `kg/relation_evidence.py:lexical_relation_gate` used YAML cues, but `check_supplies_to_direction` ignored YAML direction templates; wildcard normalization also broke intended phrase matching. YAML now drives matching and entailment prompts. |
| 5 | Confirmed | High | `check_supplies_to_direction` missed inserted-product purchase/delivery phrases and could borrow roles across sentences. It now requires local supplier/customer roles and rejects unresolved direction. |
| 6 | Confirmed | High | `kg/resolve.py:resolve_entities` and `kg/validate.py:validate_facts_frame` had no conservative model/facility type stage. Product names could pass Company-to-Company patterns. Added structural type repair and rejection audit. |
| 7 | Partially confirmed | High | `kg/resolve.py:identity_plausible/resolve_entities` protected some parent/unit examples, but qualified acronyms, cached aliases and transitive merge bridges bypassed conservative identity. Strengthened these paths; preserved canonical-ID format. |
| 8 | Confirmed | High | `expansion/loop.py:run_expand` reset its local entity document counter on every visit. Added persisted per-entity unique-document usage and global unique-document reservations. |
| 9 | Confirmed in runtime | High | `state.py:next_frontier_entity/pending_frontier_count` admitted tier <= max_tier. Export already treated the boundary as terminal. Runtime and query resolution now require tier < max_tier. |
| 10 | Confirmed | High | `kg/support.py:frontier_eligible_suppliers` returned names; `state.py:enqueue_entity` rebuilt Company IDs. Mine identity could be corrupted. Added typed candidate records and terminal non-expandable Mine defaults. |
| 11 | Confirmed | High | `graph_export.py:build_graph_bundle` used the configuration loaded during export; no expansion snapshot existed. Added `run_config.py`, snapshot identity guards and export-time snapshot consumption. |
| 12 | Partially confirmed | Medium | `config.py:log_effective_expansion_config` logged seeds/depth/scalar budgets, but omitted model, providers and runtime. Extended the concise safe log. |
| 13 | Partially confirmed | Medium | Actual `expansion.yaml` already contained all four expected seeds and depth 5; the shipped Colab overlay did not replace them with BMW/depth 2. However runtime overlays lacked production enforcement, and the notebook injected cap100 and fresh-run defaults. Protected marked runtime overlays and corrected notebook defaults. |
| 14 | Acceptance-gate hypothesis rejected | Medium scheduling inconsistency | `kg/support.py:assign_quality_status`, validator and verification stub did not require reputation or another source. Preserved that. The discovery ranker still used reputation points; removed them to make reputation fully informational. |
| 15 | Partially confirmed | Low | `verification.py` and old verification tables are compatibility remnants; `sources/independence.py:source_group_id` remains active for grouping metadata. Clarified its dedup meaning; no destructive table or schema migration. |
| 16 | Partially confirmed | High | `documents.py:make_document` removed one leading publisher suffix but could leave repeated RSS attribution; generic dash splitting also damaged ordinary titles. Parse explicit publisher metadata and clean only known attribution. |
| 17 | Partially confirmed | High | `kg/facts.py:is_expansion_relevant_fact` and export distinguished global facts from the core. But `upstream_companies` and frontier fallback ignored an all-false relevance filter. Removed fail-open promotion and unified witness gates. |
| 18 | Partially confirmed | Medium | `kg/support.py:build_frontier_reason` already exposed useful fields, but some were defaults rather than observed checks. Reasons now identify the actual witness and typed parent attachment; export reports the defensible shortest-tier core. |

## 3. Additional defects discovered independently

- **High — mixed-witness promotion:** `kg/support.py:aggregate_edge_support` combined validation/relevance flags from different evidence rows and ignored quote grounding. A runtime edge could be eligible although no single witness supported it. One complete grounded witness now suffices; unrelated invalid witnesses cannot manufacture or veto it.
- **High — resume budget leakage:** global document count restarted with each process; failed expensive attempts were free. Reservations now precede processing, persist across resume and count unique selected documents. Retry limits apply to the document across origins.
- **High — no-Neo4j accounting:** `pipeline.py:process_document` skipped COMMITTED but not successfully RESOLVED documents. Reuse now avoids new extraction/charges while preserving later RESOLVED-to-COMMITTED writes. Export counts both successful processing states and distinguishes graph commits.
- **High — final-document promotion loss:** the loop could exit at the global/runtime limit before promoting accepted suppliers. Entity finalization now runs before cap exit.
- **High — cross-seed undercoverage:** document dedup retained one origin only. An additive document-to-origin association now permits exact customer-matching contextual reuse of accepted facts without rewriting original provenance or repeating extraction.
- **High — operational failure mistaken for evidence rejection:** validator exceptions became semantic “unclear,” permanently rejecting facts. Transport/invalid-response failures now remain retryable; a genuine model judgment of unclear still rejects the candidate.
- **High — shell argument loss:** `scripts/server_run.sh` consumed its argument list before invoking tmux/SLURM children. It now preserves the original argument array, and boundary smoke depth is 1.
- **High — insufficient later-tier reserve:** even tier targets plus a beam could spend all extra allowances in early tiers. Above-target processing now reserves the planned remaining target allocation for current/deeper tiers.
- **Medium — persistence durability:** Parquet tables were overwritten directly. Shared atomic replacement protects completed files from an interrupted replacement; SQLite/Parquet remain separate stores rather than one transaction.
- **Medium — query corruption:** Google News and DuckDuckGo query construction decoded query parameters before interpolation, corrupting names containing ampersands. Proper parameter encoding preserves them.
- **Medium — search checkpoint ordering:** searches were marked DONE before their result rows were durable. Cache writes now precede completion; legacy missing-cache states are retried.
- **Medium — layered model retries:** library/SDK/extraction retries could multiply requests. Production defaults disable SDK retries and extra rate-handler attempts, leaving bounded extraction retry behavior.
- **Medium — misleading source statistics:** repeated quotes could count as multiple full-text documents; URL-wide manifest updates and omitted zero-yield documents obscured provider value. Added explicitly scoped unique-document and accepted-edge counters while retaining legacy fields.
- **Medium — target-filling navigation pages:** low-ranked supplier login/registration results could still consume slots when the pool was small. Navigation-only candidates now fail usability unless explicit upstream body evidence justifies retaining the page.
- **Low — configuration include/override details:** a diamond include was mistaken for a cycle; partial per-provider overrides lost sibling settings. Fixed stack-based cycle checking and per-provider merging. Added repeatable `--seed` for company names containing commas.

Existing shortest-path BFS, seed Tier 0 handling, cycle termination, contextual-node tier exclusion and optional Neo4j evidence records were substantially sound. They were preserved rather than broadly rewritten.

## 4. Exact files changed

Modified existing files:

- `PROJECT_DECISIONS.md`
- `PROJECT_STRUCTURE.md`
- `README.md`
- `START_HERE.md`
- `config/base.yaml`
- `config/colab_professor_expand.yaml`
- `config/expansion.yaml`
- `config/relation_evidence.yaml`
- `docs/SERVER_HANDOFF.md`
- `docs/TEST_RESULTS.md`
- `notebooks/professor_expand_colab.md`
- `pyproject.toml`
- `schain/__init__.py`
- `schain/__main__.py`
- `schain/checkpoint.py`
- `schain/config.py`
- `schain/documents.py`
- `schain/expansion/frontier.py`
- `schain/expansion/loop.py`
- `schain/expansion/queries.py`
- `schain/graph_export.py`
- `schain/ingest.py`
- `schain/kg/extract.py`
- `schain/kg/facts.py`
- `schain/kg/relation_evidence.py`
- `schain/kg/resolve.py`
- `schain/kg/support.py`
- `schain/kg/validate.py`
- `schain/llm/provider.py`
- `schain/ontology.py`
- `schain/pipeline.py`
- `schain/sources/base.py`
- `schain/sources/general_web.py`
- `schain/sources/google_news.py`
- `schain/sources/independence.py`
- `schain/sources/manifest.py`
- `schain/sources/origin.py`
- `schain/sources/ranking.py`
- `schain/sources/reputation.py`
- `schain/sources/retrieve.py`
- `schain/state.py`
- `scripts/server_run.sh`
- `tests/conftest.py`
- `tests/test_config.py`
- `tests/test_expand.py`
- `tests/test_expand_quality.py`
- `tests/test_extract.py`
- `tests/test_graph_export.py`
- `tests/test_manifest.py`
- `tests/test_pipeline.py`
- `tests/test_pipeline_reliability.py`
- `tests/test_quality.py`
- `tests/test_queries.py`
- `tests/test_resolve.py`
- `tests/test_retrieve.py`
- `tests/test_seeds_and_direction.py`
- `tests/test_server_scripts.py`

Added files:

- `docs/AUDIT_V0.3.0.md`
- `docs/audit_evidence/pre_edit_implementation_plan.md`
- `docs/audit_evidence/v030_expansion_audit.md`
- `docs/audit_evidence/v030_graph_audit.md`
- `docs/audit_evidence/v030_pipeline_audit.md`
- `docs/audit_evidence/v030_pipeline_implementation.md`
- `docs/audit_evidence/v030_ranking_audit.md`
- `docs/audit_evidence/v030_retrieval_audit.md`
- `docs/audit_evidence/v030_retrieval_implementation.md`
- `docs/audit_evidence/v030_semantics_audit.md`
- `docs/audit_evidence/v030_semantics_implementation.md`
- `docs/test-run-v030.txt`
- `schain/expansion/budget.py`
- `schain/kg/entity_types.py`
- `schain/run_config.py`
- `tests/test_expansion_budget_v030.py`
- `tests/test_graph_audit.py`
- `tests/test_ranking_v030.py`
- `tests/test_retrieval_upgrade.py`
- `tests/test_run_config.py`
- `tests/test_semantics_v030.py`

The code archive includes this report and regression tests. A preserved baseline was used for comparison. Frontend/viewer source hashes were compared before packaging.

## 5. Google News and full-text retrieval

The path is now: parse RSS title/publisher metadata; deduplicate and rank candidates; select a bounded body-fetch pool; resolve the original publisher article; fetch and clean that page; retain title/snippet if any step fails. An unresolved Google wrapper is never labeled publisher full text.

Recovery can use redirect/canonical publisher information and, for a small number of unresolved results, title plus publisher lookup with title/domain checks. Defaults allow at most 24 full-text attempts and three title-recovery lookups per entity, persisted through cached revisits. Full text remains optional: a short explicit supplier/customer statement can still pass grounding, ontology and entailment and advance from one source.

If a fetched body is unusable but the cleaned headline/snippet remains independently usable, the latter is selected explicitly; the fetched body and selection reason remain auditable. Publisher names and domains remain provenance. Only known RSS attribution is removed; ordinary dash-separated headline clauses are preserved. Early URL dedup, canonical URL dedup and title diversity reduce repeated syndicated coverage. Different wrappers/content variants may still evade deduplication.

## 6. Upstream retrieval, queries and ranking

The shared deterministic policy yields `UPSTREAM_EXPLICIT`, `DIRECTION_AMBIGUOUS` or `DOWNSTREAM_EXPLICIT`. Direction dominates secondary title/body/relevance scores. Explicit upstream relationships outrank ambiguous bilateral agreements, which outrank explicit downstream statements. Reputation contributes no scheduling points.

Five configured query families run at tiers 0–4:

1. `supplies "{entity}"`
2. `"supplier to" "{entity}"`
3. `"{entity}" sources from supplier`
4. `"{entity}" purchases from supplier`
5. `"{entity}" procurement supplier`

Buyer-side tokens are not quoted as an adjacent phrase, allowing intervening product words. Duplicate query specifications are removed. Provider limits stay Google News12, GDELT12, general web12 and Wikipedia5 per query: the configured raw-candidate ceiling falls from 328 to 205 per entity, before deduplication. This is an arithmetic ceiling, not measured retrieval volume.

Alias-aware boundary matching avoids missing Tesla because a result omits “Inc.” and avoids partial-token matches. Explicit upstream body evidence can rescue a multi-OEM headline. Ordinary report content and footer mentions no longer receive broad irrelevant penalties. A small direct-article URL preference breaks otherwise equal pre-fetch ties, preventing provider ordering from reserving all fetch slots for wrappers. There are no equal provider quotas. Downstream-only documents do not consume upstream expansion slots; downstream facts extracted from useful mixed documents or INGEST remain valid global knowledge.

## 7. Directional and relation-specific validation

The YAML policy is executable for cue matching, local role templates and LLM evidence-entailment instructions. Templates support bounded variable-length product phrases within a clause rather than adjacent-token assumptions. For example, “BMW purchases battery cells from CATL” yields CATL → BMW; the reverse is not accepted.

Supplier-to-customer language, explicit customer selection of a supplier, sourcing/purchasing from a named seller and delivery to a named customer can establish direction. A partnership, collaboration or unspecified bilateral supply agreement cannot establish it. Unresolved direction receives `ambiguous_supply_direction`. Negation, hypothetical language, unrelated sentences and arbitrary “selected as research partner” wording cannot borrow a supply predicate.

Deterministic gates save LLM work on clearly invalid candidates; surviving evidence still receives semantic entailment. Source quotes must occur in the supplied source. For frontier promotion the same witness must have ontology validity, semantic evidence validity, grounded quote, explicit direction and origin relevance. A single witness/source is enough. Unknown legacy grounding remains visible globally and is excluded from the supply-chain core.

## 8. Entity typing and conservative resolution

Structural branded model codes and “Model …” names are repaired from Company to Product before entity resolution, covering BMW i7/iX3, Tesla Model 3 and Rivian R1S without a vehicle-model blacklist. Repair reasons are logged and carried into validation audit. A mis-typed Product customer cannot be silently converted into an invented produces relationship.

Facility terms do not become Company aliases. When a facility cannot safely fit the available ontology, its Company role is rejected rather than guessed. This deliberately favors precision and can lose legitimate company names containing a facility term.

Parent, subsidiary and business-unit identity are separate from organizational relationships. Qualified acronym aliases and chain merges must satisfy conservative pairwise plausibility, including cached mappings. The Thyssenkrupp units, CATL Ningde versus CATL, BYD versus CATL, distinct battery brands and semantically different rotor descriptions are not merged merely by overlap. Existing canonical-ID normalization is retained to avoid a destructive identity migration; unusual legal-name collisions remain a known limitation.

## 9. Frontier and tier semantics

Only accepted, grounded, explicitly upstream suppliesTo evidence promotes a candidate. Typed records preserve `entity_id`, `entity_name` and `entity_type`. Company is expandable by default; Mine may be tiered, but is terminal unless enabled in ontology configuration. Enabling Mine expansion also requires suitable relation patterns and explicit evidence.

Every configured seed remains Tier 0. Tier 4 is searched to discover Tier 5; Tier 5 is stored and never searched. Contextual Product, Material, Location and Person nodes receive no supply-chain tier. Self-supply cannot manufacture a tier. Reciprocal valid global relationships may exist; shortest-path assignment and existing-node checks prevent cyclic depth inflation.

The graph retains valid global edges even when they are not expansion-relevant. Export marks the defensible shortest-tier supply-chain core using witnessed eligibility and actual parent paths. Frontier reasons identify witness fact/document IDs, origin/customer, typed parent, direction/ontology/evidence checks and promotion or rejection reasons. Beam-excluded nodes remain stored and auditable.

## 10. Checkpoints and document-budget correctness

SQLite now persists distinct entity/document reservations, attempt counts, completion, observed time and relevant metrics. Global usage counts distinct documents; per-entity usage is cumulative. A failed extraction is charged before expensive work, completed documents are reused, and interrupted completion accounting is reconciled. Retrying a shared failed document cannot obtain a fresh attempt allowance from each seed.

Additional document-origin associations are additive. Reuse projects relevance only when an accepted suppliesTo fact’s customer matches an explicitly associated origin; it never changes the original source/first-origin record. Extraction checkpoints retain stable chunk identifiers, cache outcomes and explicit partial-coverage metadata.

`data/state/run_config.json` records effective seed groups and flattened seeds, depth, document/retrieval settings, model, runtime cap, start time, version and commit when available. Changed seeds, depth, model/generation, ontology, extraction or evidence policy cannot silently reuse old checkpoints. Start a new experiment with `--restart` or separate storage. Operational cap changes are allowed and recorded as revisions. Export uses the saved experiment semantics and current output paths. Existing artifacts without a snapshot are labeled historically unverified rather than assigned invented provenance.

Elapsed runtime accumulates across completed checkpoint boundaries and resumes. The scheduler stops admitting work at the deadline or when an observed-duration reserve is insufficient; the model wrapper also checks before each new call. An already-running HTTP/model request can finish after the deadline. This is cooperative deadline enforcement, not an operating-system kill timer; an abrupt process death can lose uncheckpointed in-flight timing.

## 11. Recommended final adaptive document policy

| Tier | Target relevant documents per expanded entity | Hard maximum per entity at this tier | Expandable-entity cap |
|---|---:|---:|---|
| 0 | 8 | 12 | All configured seeds: currently 4 |
| 1 | 6 | 8 | 6 |
| 2 | 5 | 7 | 6 |
| 3 | 4 | 6 | 5 |
| 4 | 3 | 5 | 5 |
| 5 | 0 | 0 | Terminal |

Global safety ceiling: **150 distinct selected documents**. Additional scalar per-entity ceiling: **12**. Cumulative runtime budget: **110 minutes**. Bounded document attempts: **2**, with cached successful extraction chunks reused. These are ceilings and allocations, not evidence quotas.

The beam expands at most 26 entities across tiers 0–4: 4 + 6 + 6 + 5 + 5. At tier targets this plans 32 + 36 + 30 + 20 + 15 = **133 documents**. The remaining **17** slots permit selective additional coverage. Before granting above-target work, the scheduler reserves the planned remaining target allocation for later tiers. This closes the early-tier spending hole; it does not guarantee relevant suppliers exist at every depth.

Stop an entity when the candidate pool is exhausted, no meaningful candidate remains, the entity cap is reached or a global/runtime condition applies. No target is filled with downstream-only or navigation-only junk. At/after target, continuation requires explicit upstream candidates and useful recent yield; three consecutive documents without a new accepted expansion edge trigger diminishing returns. The system does not stop after the first supplier. Under-covered entities receive fair service within their tier.

Beam priority is deterministic and interpretable, using validated witness quality and source-content evidence rather than reputation or randomness. Multiple supporting documents may inform a tie/priority but are never required. Once an entity has been admitted, it retains its place across resume. Stored nodes outside the beam remain available for later review or an explicitly revised experiment.

## 12. Expected consequences and tradeoffs

No production database, fact Parquet, executed notebook or populated retrieval manifest was available for measurement. The existing 24 manifests were header-only; the existing demo was synthetic. Therefore observed average documents/entity, LLM calls/chunks/document, runtime/document, candidate acceptance ratios, provider full-text rates and frontier growth cannot be reported honestly for the baseline.

The following are planning scenarios, assuming 10 minutes total retrieval/setup overhead and effective mean document processing time including extraction, resolution and validation:

| Effective seconds per document | Time for target allocation of 133 docs, including overhead | Consequence under 110 minutes |
|---:|---:|---|
| 30 | 76.5 minutes | Target allocation is plausible; additional strong evidence may fit. |
| 40 | 98.7 minutes | Target allocation is plausible if providers and model remain responsive. |
| 50 | 120.8 minutes | Runtime will stop before all target slots; roughly 120 docs fit arithmetically. |
| 70 | 165.2 minutes | About 85 docs fit arithmetically; five-tier coverage may be incomplete. |

These are not T4 benchmarks or promises. At 40 seconds/document, caps120/150/180/200 alone imply 80/100/120/133 minutes of document work, before retrieval/setup. Cap120 leaves little headroom relative to the planned133 slots. Cap150 offers modest extra breadth. Caps180 and200 are reasonable only after measured throughput justifies them; they do not make a slow run deeper.

Graph breadth is not the beam size: each searched entity can contribute many accepted suppliers, including terminal Tier5 nodes and unexpanded nodes outside the beam. At most26 nodes are scheduled for expansion under the default beam, but total tiered nodes/edges depend on evidence yield and shared identities. No node-count target or five-tier guarantee is asserted.

The main precision/runtime tradeoff is deliberate: strong upstream evidence gets more of the limited LLM budget; explicit snippets still count; type/direction uncertainty loses recall rather than creating compounding false branches. The Colab profile extracts at most four origin-ranked chunks per document, preserves full source text and records omitted coverage. This limits calls but can miss evidence outside selected chunks. A stronger model may improve difficult coreference, implicit syntax and entity judgments; it will not fix URL order, wrong budgets, alias merges or tier errors.

## 13. Configuration changes and running the revision

- Preserve the four active seeds and max_tier5 in `config/expansion.yaml`; set global cap150 and the target/max/beam maps above.
- Change default and explicit Colab model from the inherited27B configuration to `Qwen/Qwen3-4B-Instruct-2507`. The audit found no measured runtime proving another active model. Colab uses one concurrent LLM call, max_tokens2048, request timeout120 seconds and explicit retry settings.
- Change Colab runtime200 to110 minutes. Use the explicit four-chunk Colab extraction cap with partial-coverage metadata; other profiles retain full-document behavior unless configured.
- Bound full-text attempts24 and title lookups3 per entity. Keep all four providers enabled; no equal quotas and no reputation scheduling.
- Notebook defaults now resume, inherit the document policy and inherit depth/seeds. Marked runtime overlays cannot redefine seeds/depth. Explicit CLI overrides remain supported; repeatable `--seed` preserves commas in names.

For the later authorized Colab experiment, after starting a compatible model endpoint:

```bash
python -m schain expand --config config/colab_professor_expand.yaml --no-neo4j --restart
python -m schain export --config config/colab_professor_expand.yaml
```

Use `--restart` once for this new revision/experiment; omit it for subsequent resumes. It clears local generated checkpoint/artifact state, not the Neo4j database. To continue a stopped run with an intentionally raised operational ceiling, use an explicit cap override or update the runtime cap; the snapshot records the change. Keep separate storage for different seed/depth/model experiments.

## 14. Offline verification

Consolidated command: `python -m pytest -q -m 'not integration'`.

**574 passed, 4 deselected, 15 warnings in 109.32 seconds.** The four integration-marked tests were excluded rather than run against live services. Full console output is `docs/test-run-v030.txt`. The 15 warnings are a pandas FutureWarning about empty/all-NA search-result concatenation; current nullable-cache behavior passes tests.

`python -m compileall -q schain` passed. Shell syntax and 52 runtime-script tests passed using local stubs. Frontend/viewer source hashes match the audited baseline.

A synthetic high-yield four-seed scheduling regression used the configured150-document ceiling, searched Tier4, stored Tier5 candidates and made zero Tier5 searches. This demonstrates scheduling capacity and invariants, not real supplier yield or GPU performance.

Coverage includes canonicalization before body fetch and fallback; publisher attribution; upstream ordering; YAML behavior; variable-length and ambiguous direction; model Product typing; conservative subsidiaries/caches; strict one-source support; missing/false grounding; typed Mine handling; terminal Tier5; multiple Tier0 seeds; cumulative spend and retry limits; cap-boundary promotion; cross-origin reuse; snapshot/export configuration drift; legacy Parquet; atomic persistence; and runtime runner argument forwarding. Socket/DNS guards prevent accidental external access during ordinary tests; loopback stub servers are allowed. Integration-marked live tests are excluded.

## 15. Remaining risks before a real five-tier run

1. **Model and provider quality are unmeasured.** Offline tests establish control-flow correctness, not real-world entailment precision, retrieval availability or T4 throughput. The first real run must be interpreted using recorded stage, provider and yield statistics.
2. **Five populated tiers are evidence-dependent.** A branch can legitimately stop early. Beam/global/runtime constraints may also leave eligible branches unexpanded; the graph should state this rather than invent suppliers.
3. **Runtime is cooperative.** In-flight network/model work can exceed the admission deadline until completion/timeout. A hard kernel crash can lose timing since the last checkpoint. The run is designed around110 minutes, not guaranteed exact wall-clock termination.
4. **Partial extraction trades recall for runtime.** Four selected Colab chunks can omit a useful late passage. Coverage counts and selected indices make this visible; review the omission rate before increasing the cap.
5. **Conservative semantics may reject valid complexity.** Unusual model names, organization names containing facility words, complex coordinated clauses, coreference and non-English evidence require further calibration. False splits remain preferable to false merges.
6. **Legacy facts are not retroactively corrected.** Old unknown grounding cannot advance tiers; historic alias mistakes are not globally remapped. Start the new experimental run cleanly. Canonical-ID normalization still has some suffix collisions.
7. **Publisher recovery and syndication dedup remain imperfect.** Paywalls, bot blocking, opaque Google wrappers and backend throttling can force snippet fallback. Title/domain lookup is bounded and cannot guarantee the correct original page is recoverable.
8. **Persistence is atomic per artifact, not one multi-store transaction.** Checkpoint ordering and reconciliation reduce interruption risk, but concurrent processes sharing the same storage are not supported. Use one writer per experiment.
9. **The frontend was not changed.** Consumers should use exported core/quality flags for defensible tier analysis; an older viewer can still show a broader suppliesTo selection than the backend core.

## 16. Readiness recommendation

**READY FOR FULL RUN — as a fresh, monitored Colab experiment using the supplied 4B profile and configured ceilings.**

The demonstrated tier, direction, witness, retrieval-order, budget and resume defects have regression coverage, and the consolidated offline suite passes. Start the revision as a new experiment with `--restart` or separate storage, and ensure the endpoint serves the configured model before invoking expansion. Resumes of this new experiment retain cumulative limits and snapshot identity.

The run is ready to collect the missing empirical measurements. It is not guaranteed to populate all five tiers or finish exactly at110 minutes: useful source availability, model throughput and in-flight request time determine the observed result. Review provider yield, partial-coverage records and accepted-edge samples when interpreting the exported graph. Do not treat inherited legacy frontier state or prior unverified artifacts as a validated new experiment.

The intended optimization is evidence value per available runtime: retrieve a broad pool, recover direct sources where possible, prioritize upstream relationships, preserve valid global knowledge and promote only defensible typed witnesses. Graph size is an outcome, not an acceptance criterion.
