# Retrieval ranking / queries / policy audit (pre-edit)

Scope: read complete `sources/ranking.py`, `origin.py`, `retrieve.py`, `base.py`, providers, manifest, reputation and independence helpers; `expansion/queries.py`; all expansion/evidence/source-reputation configuration; query tests and relevant quality/expansion/seed/direction tests; validation/support call sites. The repository contains no production Parquet/SQLite run artifacts. Existing manifest files are from fixture test runs and must not be used as real provider/runtime measurements.

## Requested hypotheses

| Issue | Finding and severity | Exact current responsibility | Practical effect |
|---|---|---|---|
| 2 upstream ranking | Confirmed; **high** | `schain/sources/ranking.py:110–168`, `rank_discovery_candidate`; `:15–38` phrase constants | No UPSTREAM_EXPLICIT / DIRECTION_AMBIGUOUS / DOWNSTREAM_EXPLICIT distinction. Identical score 90 for “X supplies BorgWarner”, “BorgWarner supplies Stellantis”, “BorgWarner signs supply agreement with X” under identical provider/query. A downstream headline spends the same scarce LLM budget as useful upstream evidence. Generic “names”, “wins”, “expand”, MOU also falsely count as explicit relation language. |
| 3 queries | Partially confirmed; **medium** | `config/expansion.yaml:123–201`; `schain/expansion/queries.py:25–42`; `schain/config.py:resolve_tier_queries` | Several upstream templates already exist; do not claim absent. However eight templates × four providers per entity means 32 searches and up to 328 raw candidates. Parts/component/battery suppliers are overlapping; deep tiers shift to feedstock/refining/mines without role evidence. Exact “sources from” / “purchases from” can miss inserted product phrases. |
| 4 YAML executable | Partially confirmed; **high** | `schain/kg/relation_evidence.py:54–88`, `:98–180`; `schain/kg/validate.py:63–108,330`; YAML templates | Positive/insufficient/ambiguous lists are consumed in lexical gates. Direction templates are never consumed; `_fill_template` unused; direction words hardcoded at 138–170. LLM entailment phrase lists independently hardcoded, even calling supply agreement “Strong”. `_contains_cue` normalizes away `.*` before trying to compile it, so configured wildcard cue fails. |
| 14 reputation | Reject/frontier gate hypothesis **rejected**; ranking semantics **partially confirmed**, medium | `schain/kg/support.py:build_frontier_reason`, `assign_quality_status`; `schain/sources/ranking.py:147–152` | One accepted source with score 0 can advance. There is no independent-source gate. But source score still adds up to 15 to discovery rank, contradicting strict “informational only” and changing which candidates survive budget truncation. Removing that rank prior preserves metadata. |
| 15 legacy independence | Compatibility retained; **low** | `sources/independence.py:source_group_id`; `sources/retrieve.py:116`; `verification.py`; `state.py` old tables | Group id remains provenance/dedup compatibility metadata. Verification is explicit no-op. Do not destructively migrate. Old YAML thresholds are mostly documentation; dedup helper still reads some independence keys. Docstrings naming independent support are misleading and can be clarified. `merge_group_ids` is currently uncalled. |
| Provider diversity | Telemetry incomplete; **medium** | `sources/manifest.py:134–156`; `retrieve.py:431–459` | Only raw candidate counts by provider, global full-text and global LLM counts. No selected docs, unique full-text success, accepted upstream edge attribution by provider. Stable tied ranking inherits source/provider query order and favors Google News when scores equal. No genuine observed dominance can be concluded without real run artifacts. |

## Verified reproductions (offline)

Ran existing environment Python with literal SearchResults and no network/model:

- All three upstream/downstream/ambiguous examples above rank **90.0**.
- `check_supplies_to_direction('CATL','BMW','BMW purchases battery cells from CATL.')` → false / ambiguous.
- Same function with `BMW selects CATL as supplier.` → false / ambiguous.
- `check_supplies_to_direction('X','BorgWarner','X supplies Stellantis. BorgWarner also supplies Stellantis.')` → **true**, falsely bridging sentences.
- YAML-only regex positive cue `names .* as .* supplier` fails on `BMW names CATL as its battery supplier.`
- Current query URL construction with `Electric & Hybrid Vehicle Technology International supplier` yields q=`Electric ` and an unwanted second query parameter.

## Additional findings

1. **High: evidence direction window crosses unrelated sentences.** `relation_evidence.py:152–163` uses first entity mentions and any cue in their entire window. Can infer X→BorgWarner from X→Stellantis and an unrelated next sentence. Direction engine should use sentence/clause-bounded templates and checked endpoint positions, retaining unknown/ambiguous when subject roles are not explicit.
2. **High: full text downloaded before URL dedup/relevance.** `retrieve_async:356` calls `_add_full_text` on all `fresh`, with dedup only at 380–390. One article across eight queries fetched eight times. Must dedup before fetch and cap full-text pool based on rank, without throwing away strong short evidence.
3. **Medium: alias filtering too late.** `retrieve_async:394` calls `is_relevant` requiring literal normalized entity before alias-aware origin stage at 409. Tesla Inc. normalizes to Tesla, but true aliases BMW/Bayerische Motoren Werke and CATL/long form still lost early. Ranking too uses substring match with no word boundaries. Use explicitly configured alias variants consistently, never invent overlap aliases.
4. **Medium: origin drift filter can reject useful body evidence.** `origin.py:108–109` rejects an article whose title names two other OEMs even when body explicitly states X supplies origin. Instead strong local direction should override title-based heuristic; preserve weak-body rejection.
5. **Medium: publisher/source wrappers dominate completeness score.** `ranking.py:163` searches whole article for “cookie”, “privacy policy”; one retained footer demotes good full text by 40. `sustainability report` is a negative despite general-web documentation explicitly listing it as a valuable source. Restrict portal penalties to title/snippet and remove reports from negatives.
6. **Medium: URL encoding bug.** `general_web.py:_search_duckduckgo`, `google_news.py:search` insert decoded `QueryParams[...]` into URLs. Ampersands in legitimate company names produce split queries; use serialized params.
7. **Low: title dedup not near duplicate.** `diversify_by_title` uses a normalized equality key; publisher suffix variation prevents clustering; candidate docs may spend double extraction cost on syndication. Clean known publisher attribution before dedup; avoid a strict multi-source concept.

## Proposed bounded implementation (after authorization)

MUST: deterministic origin-relative classifier with 3 states; classes are ranking only, never relation acceptance. Use shared configured direction templates/policy compiler with unknown opposite endpoint token bounded by clause. Upstream class dominates every full-text/style subscore; mixed upstream/downstream document remains upstream-priority because useful evidence exists. Attach class/reason/score to SearchResult/exportable retrieval audit. Explicitly preserve downstream candidates at lower priority (global facts remain valid). Remove source-reputation rank prior. Preserve explicit snippets.

SHOULD: five complementary main templates (supplies entity, supplier to entity, entity sources components from supplier, entity purchases components from supplier, entity procurement supplier) with one optional tier-vocabulary discovery fallback, no arbitrary industrial-role assignment. Omit tier5 queries. Deduplicate identical rendered queries. Avoid equal provider quotas. Ensure direct full-text article wins tied semantics but no score-based source gate.

SHOULD: share YAML relation cues/direction template consumption with KG validator owner; policy path must reach retrieval as well as validation. Semantic template match indicates retrieval priority, never substitutes ontology+quote-grounding+entailment acceptance.

Tests: upstream>ambiguous>downstream across reputation and provider; variable-length source/buyer phrases; mixed-fact article; alias boundaries; arbitrary new cue/template changes behavior; upstream explicit snippet remains; downstream retained; irrelevant portal gets demoted; no reputation rank effect.

## Implemented after root authorization

Changed repository files:
- `schain/sources/ranking.py`: shared executable-policy classifier; strict upstream > ambiguous > downstream ranking bands; source score accepted only for compatibility and ignored; configured aliases and word-boundary mentions; direct/full-text modest preference; portal/title penalties only; no annual/sustainability report or footer penalty; discovery direction/rank annotations; short explicit evidence preserved.
- `schain/sources/origin.py`: consistent boundary/alias checks; shared policy-aware cues; explicit upstream body can override OEM-heavy title veto without widening weak product-comparison acceptance.
- `schain/expansion/queries.py`: deterministic identical-query-spec deduplication (distinct provider/limit specifications remain distinct).
- `tests/test_ranking_v030.py`: direction, variable-length products, mixed article, no reputation influence, full-text footer, reports, portal, aliases, boundary, policy, snippets and duplicate-query regressions.
- `tests/test_queries.py`: reflects root's five complementary common upstream templates across stored tiers 0–4; tier 5 terminal; no deep industrial-role assumption.

Final isolated validation: `python -m pytest -q tests/test_ranking_v030.py tests/test_queries.py` → **50 passed in 0.65 s**, no network/model calls.
Earlier wider targeted suite before concurrent retrieval edits: **99 passed in 3.43 s**. After concurrent retrieval edits a wider suite encountered a tool-level network approval cancellation in older retrieval test paths; reported to root/retrieval owner for mock repair. This is not reported as a completed/passing suite.

No frontend edits, external infrastructure, live web experiment, source-count threshold, or schema migration made here.

### Independent integration review and follow-up

Found and fixed (root authorization): **high**, `loop.run_expand` admitted every ambiguous candidate before its soft target while ranker merely demoted portals. Concrete portal/login result passed all retrieval filters; two good documents could therefore be padded with six navigation pages. Added `ranking.discovery_candidate_is_meaningful` and called it from `is_usable_discovery_result`: reuse the existing navigation vocabulary, reject navigation-only titles without explicit upstream body, and require substantive relation cues after ignoring navigation clauses. Ambiguous bilateral supply agreements remain candidates; real upstream evidence remains usable despite a portal title. A full article about corporate EV market news cannot become relation evidence solely from its supplier-portal footer. Expansion/retrieval owners informed of helper.

Verification after follow-up: `python -m pytest -q tests/test_ranking_v030.py tests/test_queries.py tests/test_expand_quality.py` → **73 passed in 1.06 s**.

Read-only integration observations: cached discovery direction/rank are recomputed; alias variants propagate correctly through retrieval and rank/filter paths; existing accepted global facts are not deleted when downstream-only discovery candidates are skipped; explicit document-origin associations recontextualize previously accepted facts without more LLM calls.

Pending root/retrieval-owner scope decision: direct general-web URLs and unresolved Google wrappers tie before full-text shortlist. Offline pool of 24 explicit Google wrapper results plus one equally explicit direct publisher article gives all 24 full-text slots to Google wrappers, direct article rank 25. A small direct-article URL preference solves this without provider quotas/reputation. Also consider explicit-title fallback when a fetched but irrelevant page body would otherwise discard strong short evidence.

Root-authorized direct-URL follow-up is implemented: a **3-point direct article URL signal** lets equivalent direct pages enter the full-text shortlist before unresolved Google wrappers. It stays within the existing direction bands and never uses source reputation or provider quotas. Offline 24-Google-wrapper + 1-direct-page regression verifies the direct article gets a fetch slot, including when its provider label is itself Google News. Final ranking/query/expand-quality targeted suite: **74 passed in 1.12 s**. Retrieval owner is implementing the separately authorized unusable-body → explicit-title/snippet fallback with fetched body retained for audit.
