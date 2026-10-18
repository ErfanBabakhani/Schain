# Implemented semantic corrections

Audit source: v030_semantics_audit.md (original file/function/line references and confirmed probes).

## Exact files changed

- config/relation_evidence.yaml: version 2 executable bounded-phrase templates, product-span directional variants, clear semantic policy documentation.
- schain/kg/relation_evidence.py: shared YAML compiler, actual direction-template execution, Unicode/word-boundary matching, sentence/negation/coordination guards, variable product phrase slots, local role classifier and active policy prompt renderer. Missing policy fails clearly; reloading detects file modification.
- schain/kg/entity_types.py (new): small structural branded-model Product hints; facility Company role uncertainty. No vehicle-model blacklist or external calls.
- schain/kg/resolve.py: type repair before cache grouping; saved method reports repairs; whole acronym guard; qualified acronym business units remain separate; all-pairs grouping replaces transitive unsafe merges; cheap cached identity guard preserves configured trusted aliases.
- schain/kg/validate.py: defense for both direct Company-mislabeled inputs and resolver-repaired Product endpoints; facilities rejected as Company; active YAML rendered in entailment prompts; operational model/parse/missing-result errors remain retryable; rejected Parquet writes atomic.
- schain/ontology.py: compact explicit vehicle Product and facility/publisher role instructions, retaining extraction prompt size gate.
- tests/test_resolve.py: replace old CATL Ningde→parent expectation with legitimate CATL acronym→full-name cache attachment.
- tests/test_semantics_v030.py (new): direction/policy/type/cache/repair/failure regressions.

## Preserved

One explicit accepted document is sufficient. Source score does not gate validation, support or frontier. The changes add no independent-source mechanism. Deterministic direction checks precede model entailment and never substitute for it. Global canonical_id and name normalization are unchanged to preserve old artifacts. No frontend edits or infrastructure.

## Design limits

Patterns are an intentionally conservative English-language grammar, with up to 16 intervening tokens for product phrases and local word-boundary entity matches. They can miss unusual syntax, pronouns, very long spans and multilingual relationships; do not loosen them solely to increase graph size. Model entailment still handles semantic identity, tense and broader qualifiers. Type repair handles explicit model-code structure (BMW i7/iX3, Tesla Model 3, Rivian R1S), not every marketing model name. Facilities have no general-purpose allowed ontology type, so doubtful Company roles are rejected. No additional LLM type-repair call was justified for deterministic cases; uncertain cases are rejected rather than guessed.

Existing old accepted facts are not rewritten by these changes. Unsafe resolution-cache mappings are not reused for newly processed mentions, but a fresh run is needed to reassess previously accepted artifacts. Global group/holding suffix normalization remains a known compatibility-sensitive limitation; no destructive ID/schema migration performed.

## Offline validation

42 new semantics regressions; representative checks include custom YAML direction changing production behavior and appearing in the model prompt, inserted product phrases, false direction from cross-sentence mentions and research-partner selection, negation, incomplete supplier agreement, vehicle Company/Product rejection before any relation-repair call, all requested Thyssenkrupp unit splits, unsafe cached merges, and retryable model outages. Full scoped results are reported by root after integration.
