# SChain v0.3.0 targeted reliability fixes

## Scope and compatibility

This change is intentionally additive. The synchronous `ingest` and `expand` entry
points, ontology, canonical IDs, Neo4j writer, graph-export schema, tier direction,
provider abstraction, checkpoint layout, and resume state machine are unchanged.

Existing fact Parquet files remain readable. Fresh facts add four nullable provenance
columns: `original_evidence`, `evidence_repair_attempted`,
`evidence_repair_status`, and `evidence_repair_reason`. Graph export ignores these
additional columns unless a future consumer explicitly uses them. Existing extraction
checkpoints are still readable; changed prompt text naturally produces a different
per-chunk prompt hash rather than deleting or corrupting an old checkpoint.

## Root causes and fixes

### Dangling extraction endpoints

Confirmed causes:

- The inline JSON in `SupplyChainERTemplate.DEFAULT_TEMPLATE` referenced node `1`
  while declaring only node `0`.
- The parsed `Neo4jGraph` could carry that structurally invalid relationship through
  extraction.
- `graph_to_facts()` then logged and dropped any relationship whose endpoint was not
  present in entity-resolution output, losing the original edge payload from normal
  persisted diagnostics.

The prompt example now declares both nodes and explicitly requires stable local IDs,
declared endpoints, verbatim evidence, and no invented structure. Immediately after
chunk graphs are combined, `enforce_referential_integrity()` removes dangling edges
before resolution and returns lossless diagnostics containing document/chunk identity,
missing IDs, evidence, reason, and the original relationship payload. The pipeline
stores those diagnostics both beside the extraction and in the idempotent
`storage.extraction_integrity` artifact. It never fabricates a node and does not attempt
an unsafe structural repair.

### Paraphrased evidence

Confirmed cause: validation required the normalized model evidence to be a source
substring and had no recovery path. A semantically correct paraphrase was therefore
rejected before direction and entailment checks.

Recovery now considers sentence/newline-bounded source passages and makes a passage
eligible only when the literal extracted subject and object mentions and a configured
relation cue occur in it. Explicit negatives are ineligible. Token overlap only ranks
already-eligible passages; it cannot make one eligible, and equal best candidates fail
closed. The selected value is copied from the source and rechecked by the original
normalization rule. Ontology, lexical, direction, polarity, and model entailment checks
still run afterward. A repaired candidate is rejected when semantic validation is not
available. Original and final evidence and the repair outcome are retained separately.

The supply-direction policy gained bounded same-clause appositive templates so the
confirmed Rimac sentence can preserve its descriptive phrase without making direction
ambiguous. The gap is bounded, clause-local, negation guarded, and still followed by
semantic validation.

### Sequential event-loop lifecycle

Confirmed in code: extraction called `asyncio.run()` separately for splitting and for
every document extraction. Each call creates and closes a loop while the same public
LLM/provider object may be reused across documents. A deterministic loop-bound client
surrogate reproduced the unsafe multi-loop lifecycle.

The exact live `OpenAILLM`/httpx failure was not reproduced against a real vLLM server
in the offline environment, so binding inside that dependency remains the plausible
mechanism behind the observed `Event loop is closed` symptom. The minimal fix runs
splitter and extractor coroutines on one process-wide background event loop, with
bounded concurrent chunks and clean process shutdown. The public pipeline remains
synchronous and retry/checkpoint behavior is unchanged.

### Auditability and semantic precision

Optional audit tracing is disabled by default. CLI runs now assemble one comprehensive
companion JSON beside the graph from a flushed internal JSONL journal. It records run,
document and chunk context; exact call-boundary inputs; available assistant responses;
call timings; retries; parsed structure; integrity and relationship-loss results;
resolution; candidate facts; grounding, repair and validation decisions; graph
inclusion; and cache/resume events. Full details and activation instructions are in
`docs/AUDIT_TRACE_JSON.md`. Direct `process_document` callers using the legacy YAML
configuration continue to receive the earlier per-document JSONL trace.

Configure it with:

```yaml
audit:
  enabled: true
  capture_content: false
  path: data/processed/audit
```

For CLI runs, the environment-variable or YAML switch writes
`<graph-path-without-.json>.audit.json`. The response `content` exposed by
`neo4j-graphrag` is labelled assistant message content; unavailable raw HTTP metadata
is explicitly marked unavailable and no post-processed graph is mislabeled as raw.

## Rimac examples

- Before: `Rimac Technology -locatedIn-> Croatia` referenced an undeclared node.
  Resolution lacked the endpoint and fact construction discarded the edge. Now the
  edge is stopped before resolution and an `extraction_integrity` diagnostic retains
  the complete extracted relationship and `missing_relationship_endpoint` reason.
- Before: the model's shortened “Rimac Technology supplies ... BMW Group” quote was
  not a substring of the appositive source sentence and was rejected immediately.
  Now the unique full source sentence is recovered verbatim, retained alongside the
  original model wording, and accepted only after supplier/customer direction and
  entailment validation.
- “Rimac supplies batteries used in the BMW i7” still cannot support
  `Rimac Technology -produces-> BMW i7`: the recovery eligibility check requires a
  production cue, so quote recovery fails rather than rewriting the fact.
- Negated supply, reversed direction, collaboration-only language, and incidental
  location mentions remain rejected by deterministic and/or semantic checks.

## Verification

Baseline on the available Windows interpreter:

```text
python -m pytest -q
109 failed, 465 passed, 4 skipped in 68.87s
```

Those baseline failures were dominated by the Python Store interpreter returning
`[Errno 9] Bad file descriptor` for the repository's durability `fsync` path and by
Linux shell-script tests on Windows. A WSL focused baseline had 64 passes and four
provider-construction failures because its environment lacked the declared `openai`
package.

Final deterministic reliability suite:

```text
python3 -m pytest -q tests/test_reliability_fixes_v030.py
11 passed in 2.41s
```

Final affected-path and interface compatibility suite (after installing the declared
WSL `openai` dependency):

```text
355 passed in 50.07s
```

It covers extraction, quality, pipeline/resume, ingest, expand, semantics, graph
export/integrity, tier direction, Neo4j writer preparation, CLI/configuration, state,
entity resolution, run configuration, documents, and the new reliability tests.
The complete WSL suite produced 565 passes, 5 skips, and 18 environment-related
failures in 168.08 seconds. Remaining failures require absent WSL retrieval packages
(`lxml`, `trafilatura`) or exercise server bootstrap/virtualenv behavior on a Windows
mounted filesystem; none is in a changed module. Four provider tests also passed
separately on Windows (`4 passed in 5.20s`).

## Remaining limitations

- No extraction system guarantees recall. The prompt now makes produces/location and
  negative distinctions clearer, but a live-model benchmark is still needed to measure
  recall and false-positive changes.
- Structural repair is intentionally not attempted. A dangling edge is retained as a
  diagnostic and excluded; fabricating the entity from an ID would be unsafe.
- Quote recovery is sentence/newline bounded and fails on ambiguity. Supporting text
  spread across sentences will remain rejected unless the model supplies an exact,
  independently sufficient quote.
- The actual remote vLLM/httpx failure was not exercised because tests are offline.
  The regression proves all three documents use one live loop with one client object.
- Full-content audit files can contain sensitive document/model content; capture remains
  explicit opt-in and operational retention controls are the deployer's responsibility.
