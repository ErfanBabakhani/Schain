# Pre-edit expansion/state audit

Inspected the complete contents of `schain/expansion/{loop,frontier,queries}.py`, `schain/state.py`, `config/{base,expansion}.yaml`, tests `test_expand.py`, `test_expand_quality.py`, `test_state.py`, `test_seeds_and_direction.py`, and the relevant complete support/facts implementations plus pipeline/retrieval persistence call sites. No repository files edited during this audit.

## User hypotheses

| Issue | Verdict / severity | Evidence and practical effect |
|---|---|---|
| 8 cumulative per-entity budget | Confirmed / high | `loop.run_expand` line 185 resets `entity_processed=0` every claim; limits at 222 and fairness at 304/336 use this local count. Requeued entity may process 8 then 8 with hard max12. Nothing persists selected-document usage. |
| 9 max tier terminal | Confirmed / high | `state.next_frontier_entity` and `pending_frontier_count` use `tier <= max_tier`; `frontier.discover_upstream` checks boundary only after retrieval and extraction. Config tier5 has eight queries; only tier6 sets expand=false. Tests `test_expand_discovers_the_next_tier`, `test_expand_stops_at_tier_zero_when_configured` codify wasteful boundary behavior. |
| 10 Mine preservation | Confirmed / high | `support.frontier_eligible_suppliers` accepts Mine but returns names; `frontier.discover_upstream` passes these names to `state.enqueue_entity`, whose key always uses `canonical_id("Company", name)`. Fallback `facts.upstream_companies` excludes Mine, so behavior depends on presence of support Parquet. `entity_identifiers` also hardcodes Company. |
| 17 global vs expansion core | Partially confirmed / high | `facts.is_expansion_relevant_fact` correctly uses object=origin + suppliesTo + Company/Mine. Support eligibility correctly gates relevance. However `facts.upstream_companies` only filters relevance when at least one true exists; all-false sets fall back to all matches. `frontier.discover_upstream` repeats same pattern; when support missing this is a fail-open core promotion path. |
| 18 frontier auditability | Partially confirmed / medium | `support.build_frontier_reason` already supplies requested fields. `parent_tier_valid=True` default is never verified against state by aggregation; `generic_entity=False` similarly default. SQLite frontier stores only one parent, no edge/evidence/priority/type/stop reason. Need make actual promotion rationale explicit rather than treating default booleans as proof. |

## Additional independent issues

1. **High — resume resets global spend.** `loop.run_expand` initializes `summary.documents_processed=0` at160 and caps against it at216/297. Restarting an interrupted process permits another full cap. Failed extraction attempts also consume no document ceiling, regardless of expensive partial extraction.
2. **High — no-Neo4j repeat accounting/fairness loop.** `pipeline.process_document` skips COMMITTED at229 but leaves no-driver successes RESOLVED at319. Revisit reports them as processed again with same accepted-fact count; loop charges budget, reports duplicate facts, and may keep requeuing seeds. Stub/fake-driver tests miss this common Colab path.
3. **High — accepted suppliers lost at global/runtime boundary until resumed.** `loop.run_expand`321 breaks before refresh/discovery at325–329. A final accepted document can be stored without its supplier entering frontier in the delivered run.
4. **High — content dedup loses second origin.** SQLite documents stores one origin; `state.register_document` returns before attaching a new origin when duplicate, and `_facts_for_entity` uses only that first origin. A shared article mentioning two seeds cannot cheaply establish second-origin relevance, starving another valid branch; blindly changing origin would contaminate first branch. Additive document↔origin association + safe contextual projection needed.
5. **Medium — quality-blind breadth-first exhaustion.** `next_frontier_entity` chooses lowest tier, count of registered documents, created time. No beam and no budget reserve for deeper tiers; high Tier1 branching can spend entire cap before any Tier4 search. Seeds must remain uncapped.
6. **Medium — runtime cap only at unit boundaries.** `run_expand` checks before retrieval and individual docs, but retrieval can fetch hundreds of candidate pages and a long document can spawn many chunks. No persisted elapsed usage or estimate to avoid starting a known over-budget unit. Existing 110-minute setting is explicitly a soft cap.
7. **Medium — retriever errors permanently suppress branch.** Top-level retrieval exception marks frontier FAILED and continues; default resume only requeues PROCESSING. Some failed searches inside retrieval are retriable but entity may later be DONE, never serving failed providers again. Bounded explicit retriable status is preferable; avoid spinning.
8. **Medium — repeated whole-table I/O.** pipeline refreshes all edge support for every resolved document, then loop refreshes again per entity; `_facts_for_entity` reads all facts per entity. Correctness first; remove proven duplicate refresh, retain simple Parquet compatibility.
9. **Medium — provenance metrics are occurrence counts.** Manifest updates all provider/query rows sharing URL; summary.sent_to_prompt1 therefore counts repeated search hits as separate LLM documents. No per-provider accepted edge yield or observed document times. Header-only manifest files cannot support runtime calibration.
10. **Low — second-resolution normalization ambiguity.** Frontier re-generates canonical ID from display name instead of preserving extracted subject_id; any valid ID differing from regenerated slug can disconnect parent/origin matching. Additive entity_id/type columns solve this together with Mine preservation.

## Runtime evidence and budget recommendation

All 24 existing `logs/retrieval_manifest_*.csv` files are header-only (zero rows), and data/{processed,state,raw} contain no run artifacts. Thus no measured averages exist for runtime, docs/entity, accepted-fact ratio, full-text success, or frontier growth. Do not invent measured graph/runtime expectations. Base config is Qwen/Qwen3.8-27B; Colab runtime should be audited separately by root.

A provisional rational ceiling is 150 unique selected documents and 110 minutes elapsed. Tier targets 8/6/5/4/3/0, hard maxima12/8/7/6/5/0, with scalar hard per-entity12 as an additional ceiling. Beam counts tiers1–4 of 6/6/5/5 plus all4 seeds give target spend32+36+30+20+15=133; remaining17 allow carefully selected continuation or retries. This is a planning arithmetic, not guaranteed graph size. Nodes outside expansion beam remain stored, and accepted tier5 nodes are never searched. For 133 docs, 110 minutes allows under49.6 seconds/doc including retrieval. Without observed timings this could be optimistic. Compare hard caps:120×40sec=80min before retrieval,150×40=100min,180×40=120min,200×40=133min. Runtime should end earlier if measured recent doc time predicts overrun. Probe only first few high-quality docs in the authorized future run, not a live experiment now.

Diminishing returns: don't quit on first supplier; minimum3 processed docs where possible, stop after3 consecutive zero-new-upstream-edge docs; after tier target require explicitly upstream remaining candidate and recent gain to continue; never fill target with downstream or irrelevant candidates. Beam ranking should use explicit validated edge, origin relevance, document evidence quality (fulltext flag), stable ID tie-break; no source score or mandatory multiplicity. Select beam after prior tier candidate discovery; fairness within tier should preserve every configured seed.

## Pre-edit priorities

MUST FIX: terminal max tier; persisted unique per-entity/global usage across failure/resume; no-driver completed-doc skip/count; strict relevance fail-closed; typed canonical frontier entries and Mine terminal default; finalize discovery before cap exit.

SHOULD FIX: tier-specific budget, interpretable beam caps, measured timing/provider/accepted-edge telemetry; safe shared-document origin association; additive actual frontier stop/promotion audit; remove duplicate full-table support rebuild.

OPTIONAL CLEANUP: legacy independence schema remains untouched; retrieval retry/backoff redesign beyond bounded run behavior; broader storage refactor deferred.

### Offline confirmation

A temporary SQLite/Parquet repro (no web/LLM) confirmed:

- `next_frontier_entity(max_tier=5)` returned the tier5 `Boundary` entity.
- All-false expansion-relevance rows with no support Parquet promoted `CATL`.
- A valid `Mine` support row promoted `Deep Mine` under `company:deep-mine`.

Additional runtime overlay inspection: `config/colab_professor_expand.yaml` sets max_minutes=200, contrary to base110 and the requested 90–120 minute operating window.

## Implemented after root audit authorization

Owned code files: `schain/state.py`, `schain/expansion/loop.py`, `schain/expansion/frontier.py`, new `schain/expansion/budget.py`. Owned test edits: `tests/test_expand.py`, `tests/test_expand_quality.py`, new `tests/test_expansion_budget_v030.py`.

- Additive SQLite tables `document_origins`, `expansion_document_usage`, `expansion_run_state`; additive frontier type/expandable/priority/reason/stop/streak columns. Existing original document provenance is preserved. No independence schema migrations.
- Canonical supplier IDs/types survive frontier promotion; Mine defaults to stored terminal Tier1+ (unless ontology explicitly enables expansion), Tier5 terminal always, and actual parent row/tier is checked.
- Unique selected document reservations are written before expensive calls. Failed documents count once toward global/per-origin ceilings; retries bounded to two total attempts across all origins. Completed-ledger recovery repairs crash windows without more spending. Completed documents reused under another origin gain an association, not a new LLM call.
- Global cap/runtime elapsed persist across resumes. In-flight call deadline passed to metered pipeline; no new LLM calls start after remaining deadline. Latest observed mean document-attempt time×1.15 prevents starting a likely overrun. Abrupt kill can lose elapsed in the unfinished unit, and an in-flight request can exceed the wall-clock boundary; do not describe this as strict cancellation.
- Tier targets8/6/5/4/3, maxima12/8/7/6/5 with scalar12 extra ceiling, terminal tier5 target0. At target, yield to other pending entities. More documents require explicit upstream direction and accepted upstream evidence; stop after three consecutive zero-new-upstream-edge documents once at/above target and minimum3. Meaningless portal/navigation candidates are rejected before budget allocation; fewer strong results need not fill target.
- Configured quality-ranked beams admit all4 seeds then6/6/5/5 expandable entities. Already admitted entities retain slots on resume; excluded nodes remain stored with `frontier_beam_cap`. No reputation gate or mandatory supporting-source count.
- Above-target spend is allowed only after reserving unserved planned target coverage for current/deeper beam slots. Planning arithmetic133 targetdocs leaves17extras under150. Exhausted entities release their remaining target requirements. This prevents high-yield shallow branches exhausting budget before Tier4.
- Suppliers are finalized before cap exits; startup also repairs accepted-facts→frontier crash windows even if no budget remains. No-neo4j completion is respected, and same-batch duplicate candidates are rechecked before reservation.
- Persistent `run_metrics.json` contains cumulative document/timing/LLM totals, current quality, frontier tier counts and provider manifest summary. Manifest writes one selected provider row per processed document and accepts edge key sets.

### Validation evidence

Focused scheduling/state/expansion/seed-direction/manifest suites:109 passed (one non-failing retrieval pandas FutureWarning). New scheduling module reached19 passing tests after crash-window cases were included. Final root-wide offline suite will supersede this subtotal.

The adversarial **synthetic** high-yield scheduler test (no live LLM/web) searched all4 seeds and tiers0–4 with150 unique documents, searched noTier5, and stored artificial node counts4/48/37/30/20/15. This proves scheduling/depth budget feasibility under abundant accepted fixtures; it is **not** a prediction of real graph size, precision, recall, source yield or T4 runtime.
