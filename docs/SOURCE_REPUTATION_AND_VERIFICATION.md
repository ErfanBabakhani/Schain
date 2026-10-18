# Source Reputation and Edge Quality (V1)

## Design principle

The pipeline answers:

> Did the retrieved text explicitly support this supply-chain relation?

It does **not** require two or three independent websites to say the same thing.

```
search → rank → fetch (full text preferred; title/snippet allowed)
→ Prompt 1 → Prompt 2 → ontology gate → evidence entailment
→ accepted fact (+ source reputation metadata)
→ semantic edge aggregation
→ frontier from valid suppliesTo Company→Company
```

## What was removed

Independent-source counting, multi-source status ladders
(`candidate` / `strong_candidate` / `supported` / `verified` from source counts),
and targeted second/third-source verification searches are **disabled**.

`config/base.yaml` keeps `verification.enabled: false`.
`schain/verification.py` is a no-op stub for import compatibility.

## Source reputation (informational)

`config/source_reputation.yaml` and `schain/sources/reputation.py` remain.

Scores appear on documents, facts, evidence, and graph export so a frontend or
analyst can judge quality. Reputation does **not**:

- reject a fact
- block an edge
- set `frontier_eligible = false`
- require full text

Google News wrappers may score `0` / `retrieval_wrapper` and still produce an
accepted, frontier-eligible edge when ontology + evidence checks pass.

## Frontier eligibility

A `suppliesTo` edge advances the frontier when:

1. relation is `suppliesTo`
2. subject type is expandable (`Company`)
3. ontology pattern is valid
4. evidence validation supports the relation
5. subject is not a generic category phrase (e.g. "chip suppliers")

One explicit valid source is enough. Title/snippet evidence is allowed when explicit.

## Edge quality object

```json
"quality": {
  "status": "accepted",
  "verification_status": "not_required",
  "max_source_score": 94,
  "average_source_score": 72,
  "ontology_valid": true,
  "evidence_validated": true,
  "frontier_eligible": true,
  "contested": false
}
```

`support_count` = number of unique **accepted** documents for the semantic edge.
`independent_support_count` is deprecated and exported as `null`.

## Deduplication stays

URL / content-hash / document-id / fact-id dedup still runs. That is not
independent-source validation.
