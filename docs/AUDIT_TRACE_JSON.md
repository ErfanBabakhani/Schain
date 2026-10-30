# Complete audit trace JSON

SChain can write one self-contained audit JSON beside every graph export. It is off by
default and does not add model calls or change extraction, validation, checkpoint, or
graph-export decisions.

## Enable it

```bash
export SCHAIN_AUDIT_TRACE=1
export SCHAIN_AUDIT_FULL_CONTENT=1   # optional and sensitive
python -m schain ingest --input data/corpus --no-neo4j
```

`SCHAIN_AUDIT_TRACE=1` enables the companion artifact. `SCHAIN_AUDIT_FULL_CONTENT=1`
also stores normalized source text, chunk text, actual rendered model inputs, available
assistant response content, parsed graphs, evidence, and resolution payloads. Both
variables are ordinary environment variables, so they work unchanged when passed in
the `env` mapping of a Colab `subprocess.run` call.

For graph path `outputs/run.v1.json`, the output is exactly
`outputs/run.v1.audit.json`, calculated as:

```python
audit_path = graph_path.with_name(graph_path.stem + ".audit.json")
```

YAML `audit.enabled` and `audit.capture_content` remain compatible alternatives. The
environment variables take precedence by enabling their respective behavior.

## Contents

The top-level schema is versioned as `1.0` and contains:

```json
{
  "audit_schema_version": "1.0",
  "run": {},
  "configuration": {},
  "documents": [],
  "chunks": [],
  "llm_calls": [],
  "extraction_graphs": [],
  "integrity_checks": [],
  "entity_resolution": [],
  "candidate_facts": [],
  "validation_decisions": [],
  "repair_attempts": [],
  "accepted_facts": [],
  "rejected_facts": [],
  "relationship_outcomes": [],
  "graph_export": {},
  "events": [],
  "errors": [],
  "summary": {}
}
```

The chronological `events` list is the lossless source for the summarized sections.
Its monotonic sequence numbers remain unambiguous when asynchronous calls overlap.
Stable IDs join documents, chunks, extraction attempts, LLM calls, extracted
relationships, candidate facts, final facts, and exported edges.

Every actual `MeasuredLLM.invoke` and `MeasuredLLM.ainvoke` call is captured at the
existing invocation boundary. In full-content mode, `request` is the exact input,
message history, system instruction, and keyword arguments passed to the wrapped
`neo4j-graphrag` interface. `assistant_message_content` is the real
`LLMResponse.content` returned by that interface. The underlying provider HTTP body,
token usage, and finish reason are recorded as unavailable with a reason when that
interface does not expose them; they are never reconstructed or mislabeled as raw.

The trace also distinguishes:

- processed, capped, resumed, and cached chunks;
- every parsed relationship retained or dropped for a missing endpoint;
- deterministic, cached, configured-alias, and LLM entity resolution;
- candidate construction, every executed or skipped validation check, and final
  acceptance/rejection;
- evidence and ontology repair attempts and their linked model calls;
- accepted fact IDs aggregated into each exported edge and accepted facts omitted by
  export filtering.

The graph JSON schema and contents are not extended with audit fields.

## Failure and cache behavior

Events are appended to an adjacent internal JSONL journal and flushed during the run.
Normal completion or a controlled exception assembles the final JSON with an atomic
replacement. Failed or partially failed runs set `run.complete` to `false`, retain the
error and completed events, and do not claim success. The journal is removed only after
successful final assembly.

A chunk checkpoint hit records its cache key/path and reused parsed graph, without
creating an LLM call. Historical prompts and responses that the existing checkpoint
format never stored are explicitly `unavailable_from_chunk_checkpoint`. Normal
document-level resume decisions are likewise recorded without fake model calls.

## Security and size

Full-content traces may contain confidential documents, prompts, responses, and
evidence. Keep them in the ignored `outputs/` directory, restrict access, define a
retention policy, and do not commit or publish them. Configuration keys resembling
passwords, authorization values, API keys, secrets, and tokens are redacted; endpoint
userinfo is removed. SChain never intentionally records authentication headers.

Large inputs can create large audit files because full-content mode does not silently
truncate content. Use metadata-only tracing by omitting `SCHAIN_AUDIT_FULL_CONTENT`
when complete payload retention is unnecessary.
