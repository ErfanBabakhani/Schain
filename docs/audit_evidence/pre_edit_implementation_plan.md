# Audit-first implementation decision
Audit completed across all production modules, configuration, runtime scripts, tests and frontend read-only before repository edits. Six detailed audit notes record exact pre-edit functions, reproductions and severities. Baseline source preserved in analysis/v020_audited_baseline.

## MUST FIX BEFORE NEXT RUN
- Canonical publisher URL before full-text fetch; preserve explicit snippet fallback and publisher provenance.
- Shared executable direction policy with inserted-product phrases, sentence boundaries and conservative ambiguity; type and identity safeguards.
- Same evidence witness must meet all promotion gates; fail-closed relevance; preserve typed canonical frontier identity; max tier terminal.
- Persistent global and per-entity document accounting; no-Neo4j completed reuse; final discovery before cap exit; safe cross-origin reuse.
- Effective run snapshot, export snapshot semantics, explicit Colab 4B/110-minute settings, runtime-overlay protection.
- Propagate operational LLM validation failures; preserve shell runner arguments.

## SHOULD FIX
- Upstream-first deterministic ranking; five complementary queries; alias-aware filtering, early URL dedup and bounded fetch shortlist.
- Tier budgets 8/6/5/4/3/0 targets, 12/8/7/6/5/0 maxima, per-entity12/global150/runtime110; beam all seeds/6/6/5/5/terminal.
- Stop for exhaustion, no meaningful candidate, hard ceilings; diminishing returns after sufficient breadth; measured runtime scheduling.
- Provider/document/LLM metrics, typed witness/promotion reasons, atomic Parquet persistence, bounded retries.

## OPTIONAL CLEANUP
- Preserve legacy independence fields/tables and graph JSON schema; explain dedup-only terminology.
- Defer broad storage redesign, external infrastructure, frontend changes, real model/provider calibration.

No live web/LLM experiment will be run. Budget and throughput predictions remain conditional because no production run measurements exist.
