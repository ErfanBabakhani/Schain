# SChain v0.2.0 semantics audit (before edits)

Scope fully read: kg/validate.py, kg/relation_evidence.py, kg/resolve.py, names.py, ontology.py, ontology.yaml, relation_evidence.yaml; quality, resolver, name, extraction and seed/direction tests; pipeline resolve/validation integration. All observations below are from supplied code, with no live LLM/web run.

## Issue 4 — partially confirmed, HIGH

`validate_facts_frame` (kg/validate.py:330, 422–428) loads YAML and passes it into lexical and directional checks. `lexical_relation_gate` (kg/relation_evidence.py:74–84) DOES consume positive/insufficient/ambiguous cues; hypothesis that policy is wholly documentary is rejected. However `check_supplies_to_direction` (:138–170) uses hard-coded supplier/customer cue tuples and never reads either YAML direction-template list. `_fill_template` (:91–95) is unused. `EVIDENCE_BATCH_PROMPT` (kg/validate.py:69–116) duplicates relation cue prose and contradicts the YAML by listing “supply agreement” among strong evidence before qualifying it later. The LLM prompt never receives YAML. `_contains_cue` (:44–46) normalizes away `.*` before trying to compile it, so YAML wildcard positive cues also do not work.

Minimal recommendation: one small wildcard-safe policy template compiler; production direction matcher consumes YAML templates with word boundaries and sentence/clause scoping; render actual active relation policy in entailment prompt. Retrieval classifier should import shared policy matcher/cues. Preserve semantic entailment after deterministic checks.

## Issue 5 — partially confirmed, HIGH

Works now: `CATL supplies battery cells to BMW`, `CATL sources lithium from Albemarle`, and a plain bilateral supply agreement is rejected as ambiguous. Confirmed misses: “BMW purchases battery cells from CATL” and “CATL delivers battery cells to BMW” return false; “BMW selects CATL as supplier” also returns false. Cause: adjacency-dependent `purchases from`/`delivers to` strings and limited verb inflections in kg/relation_evidence.py:138–170.

More serious independent false positives reproduced: “BMW selected CATL as a research partner”, “CATL never supplies BMW”, “CATL supplies Tesla. BMW is based in Munich.” all return true for CATL→BMW. Name order plus an unscoped cue is not explicit direction; this leaves all rejection responsibility to a small LLM. The first `.find` of each name also loses later valid clauses. Recommend endpoint-bound templates, sentence-bound variable product slots, role completion (`as … supplier`), and local negation guards. Do not infer a supply relationship from selecting an arbitrary partner or cross-sentence co-occurrence.

## Issue 6 — confirmed, HIGH

There is no Company name/type plausibility stage. kg/validate.py:370–386 rejects only generic categories. Schema validation checks supplied type labels rather than whether those labels are credible. A Company-labelled “BMW i7” therefore passes Company→Company ontology validation if the model also approves evidence. `looks_like_facility` and alias guards in kg/resolve.py:81–85/299–301 prevent a fresh plant alias from collapsing into its company, but do not stop a separately Company-labelled facility being an expandable node. `pipeline.process_document` (:278–283) saves resolution before validation, allowing bad type/name cache entries to persist.

Recommendation: conservative name-structure + evidence type plausibility before resolution/cache (or within resolver), with Product repair for unmistakable branded model patterns and a logged audit property. Reject facility-as-Company and uncertain Company mentions from relations when repair cannot be grounded. Keep vehicle logic structural, not a model-name blacklist. Validation must defend direct callers too. Avoid adding broad LLM type calls for every entity.

## Issue 7 — partially confirmed, HIGH

Existing protections WORK for direct BYD/CATL, Blade Batteries/Ultium, ventilated/carbon-sleeved rotors, facility/owner, and Thyssenkrupp vs each named Steel/nucera/Aerospace unit. Tests cover these. `identity_plausible` (:144–159) correctly refuses most nested business-unit names. Do not replace working safeguards.

Independent holes:

* `_is_acronym_of` (:97–104) accepts prefix/partial initials, then `identity_plausible` (:130–136) accepts an acronym token anywhere in a larger name. Thus CATL Ningde merges with Contemporary Amperex Technology despite the forbidden configured alias test; `test_new_alias_attaches_to_an_existing_canonical_entity` explicitly enshrines that unsafe merge. `ABC` is considered both Alpha Beta Chemicals and Alpha Beta Chemicals Subsidiary.
* `sanitize_llm_groups` (:163–193) union-finds safe pairs; a safe pair bridge bypasses a directly unsafe pair. Reproduced one group `[Alpha Beta Chemicals, ABC, Alpha Beta Chemicals Subsidiary]` although the two full names fail identity_plausible. Use complete-link partitioning (every pair must be plausible), deterministic order, no fuzzy transitive closure.
* cached entries (:270–285) bypass all current plausibility guards. Existing unsafe facility/subsidiary entries remain trusted forever. Apply cheap plausibility protection to cache entries except explicit trustworthy configured aliases (which may be lexically disjoint trade/legal names).
* `normalization_key` (names.py:14–17, 34–36) globally strips group/holdings/holding as if legal suffixes for every entity type. This can silently collapse parent holding entities and operating companies before any LLM guard. Compatibility implications mean assess separately; at minimum stop generic holdings normalization for new identities with documented old-artifact/cache behavior. BMW Group already has an explicit trusted alias.

Preserve configured lexically disjoint known aliases such as Nornickel→Norilsk Nickel. Never infer identity from owns/suppliesTo; relationships remain independent facts.

## Priority

MUST: executable direction templates + variable product spans + local direction/negation guards; type plausibility; remove partial/acronym-token bypass and transitive merge bridging. SHOULD: active YAML in entailment prompt; cached identity guard; safe missing-policy behavior; review holdings normalization compatibility. OPTIONAL: stricter overlapping LLM group sanitation, relation-validation result cache to avoid repeating entailment for repeated chunk evidence, clean stale docstrings referring to reputation gates.

## Offline probes executed

Direct import script called check_supplies_to_direction/identity_plausible/sanitize_llm_groups. No test/code edits and no live services. Custom policy `{subject} transfers .* into {object}` was ignored, demonstrating direction templates are dead configuration. All concrete probe outputs above were observed, not hypothesized.
