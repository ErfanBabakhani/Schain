# Implemented retrieval changes (v0.3.0)

Files owned/changed:
- schain/sources/base.py: additive publisher/raw-title, canonical/fetch attempts and rank metadata; provider HTTP error context; duplicate title/snippet collapse.
- schain/sources/google_news.py: correctly encoded queries; parse RSS publisher name/source homepage separately; remove only known trailing publisher attribution from headline and description.
- schain/sources/general_web.py: correctly encoded DuckDuckGo queries.
- schain/sources/retrieve.py: canonical wrapper resolution before body fetch; redirects plus explicit canonical/og:url metadata; reject Google consent/sign-in as articles; bounded exact-title+publisher-host recovery; direct URL fetch coalescing; pre-ranked bounded body shortlist; persisted per-entity attempts; alias-aware relevance; result-cache durability before DONE; legacy missing-cache recovery; nullable old Parquet loading; additive metadata propagation.
- schain/sources/manifest.py: cumulative same-run CSV reopen and atomic flush; canonical/publisher/doc/rank fields; exact used-provider credit; per-provider raw/ranked/selected/fulltext/fetch/accepted-edge metrics; deduplicated unique processed-document counts and preservation across skipped resumes.
- schain/sources/independence.py: clarify legacy metadata only; preserve interfaces/schema; grouping is not safe for document dedup.
- schain/sources/reputation.py: clarify informational scope; actual source scoring kept.
- schain/documents.py: known publisher metadata only, never guess by dash on fulltext; atomic normalized Parquet writes.
- tests/test_retrieval_upgrade.py: 15 focused offline regression tests.
- tests/test_retrieve.py: explicitly mock canonical resolution in synthetic orchestration fixtures.
- tests/test_manifest.py: terminal-tier fixtures and distinct-document statistic expectations; consistent provider/query identity.
- tests/test_seeds_and_direction.py: publisher test now supplies real publisher metadata.

Config consumed (root owns YAML): expansion.retrieval.max_full_text_fetches_per_entity=24; max_title_lookups_per_entity=3; max_candidates_per_entity optional ceiling (0 means unlimited ranked pool; LLM budget remains controlled by expansion). Full-text and title lookup attempts persist across revisit/resume. A failure does not cause endless retry; a fresh run can refresh retrieval.

Tests executed:
`python -m pytest -q tests/test_retrieval_upgrade.py tests/test_sources.py tests/test_retrieve.py tests/test_documents.py`
=> 67 passed, 13 existing-style pandas concat FutureWarnings, 4.40s.
Combined manifest/seed integration => 34 passed, one integration failure sent expansion owner: missing DUPLICATE_CONTENT mark on document skip branch; owner will restore retrieval summary logging too. Root should report final consolidated suite result, not partial agent counts.

No provider quota, no reputation acceptance/ranking gate, no second-source verification, no frontend change, no live experiment.

Boundaries: opaque Google News wrappers may remain unresolved after bounded attempts; explicit title/snippet survives. No live provider hit rate or full-text yield is measured. HTTP retries are not endless; provider HTTP failures recorded FAILED rather than successful empty search. Broad body reads are capped, but a running HTTP/LLM request may finish after the cooperative runtime deadline. Supplier tables and downloaded HTTP PDFs are not newly supported in this patch. Existing legacy cached text may still require a clean run to benefit from source cleanup.
