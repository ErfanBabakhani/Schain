# Ingest reliability and extraction recall fixes

This note documents the focused v0.3.0 corrections made from baseline
`dce12472d8d28774dfe85d58ed68231783b3103d`. The original architecture,
ontology, validation stages, checkpoints, and graph formats remain in place.

## Confirmed root causes and corrections

1. **Incomplete semantic batches** — `schain/kg/validate.py` previously raised
   from `_batch_evidence_validate` whenever one requested ID was absent, losing
   the other valid decisions and failing the document. Responses are now checked
   for missing, duplicate, unknown, malformed, and invalid entries. Valid
   decisions are retained, only unresolved IDs are retried in bounded smaller
   batches, and exhausted IDs are rejected as
   `semantic_validation_incomplete`. Transport failures remain retryable
   operational failures rather than semantic rejections.
2. **Dangling extracted endpoints** — `schain/kg/integrity.py` previously only
   recorded and removed dangling relationships. It now performs conservative,
   ontology-compatible recovery in this order: exact ID, a unique existing-node
   name match, a repeated identical extracted identity, or a subject-bound
   verbatim location phrase. Ambiguous, unsupported, and type-incompatible cases
   are still removed. Recovered relations receive every normal downstream check.
3. **Low `produces` recall** — the omission was already present in raw extraction
   responses, before parsing or Candidate Fact creation. `schain/ontology.py` now
   balances the extraction instructions with concise manufacturing positives and
   hard negatives and requires a final pass over every allowed relation type.
   This is a prompt correction, not a bypass or deterministic injection of facts.
4. **Rejected valid short-name aliases** — `schain/kg/resolve.py` treated every
   parent/subset name pair as unsafe even when the same document used an
   unambiguous short form. It now permits explicit equivalence or collision-free
   in-document shorthand, while continuing to separate facilities, collisions,
   subsidiaries, divisions, and similar legal entities. Every LLM-proposed merge
   is audited as accepted or refused.
5. **Misleading completion and counters** — failed document counters disappeared
   because `run_ingest` only accumulated a normally returned summary, and graph
   export completion could describe an empty frontier rather than ingest
   processing. Failed live summaries now propagate their counters. Additive
   completion metadata distinguishes `export_success`,
   `pipeline_processing_complete`, `documents_failed`, `documents_skipped`, and
   `graph_has_accepted_facts` in the graph bundle and export audit event.

An inherited resume-path defect was also exposed by these tests: `origin_name`
could be unbound when resuming from an extracted checkpoint. It is now initialized
before the stage branch, and repaired checkpoints are persisted.

## Configuration and audit behavior

The existing evidence-validation configuration accepts two optional keys:

```yaml
validation:
  evidence:
    incomplete_batch_max_attempts: 2
    incomplete_retry_batch_size: 1
```

New trace events are `semantic_validation_batch_response`,
`semantic_validation_targeted_retry`, `semantic_validation_unresolved`,
`endpoint_already_valid`, `endpoint_id_recovered`,
`endpoint_node_reconstructed`, `endpoint_recovery_ambiguous`,
`endpoint_unrecoverable`, and `alias_merge_evaluated`. Audit capture is passive;
an audit-on/off regression verifies identical semantic decisions for the same
mocked responses.

## Measured results

The supplied audit recorded 38 Candidate Facts, 8 dangling relationships, and a
10-candidate evidence request whose valid nine-item response omitted
`75a578701628bbff`. The old global exception resulted in zero persisted accepted
facts and a zero-edge graph.

Replaying the supplied parsed extraction graph through the corrected integrity
stage produced 46 input relationships, 46 retained relationships, 8 reconstructed
endpoints, and 0 unresolved integrity failures. This replay measures structural
recovery only; it does not claim those relationships were semantically accepted.

| Metric | Supplied failed run | Corrected result available here |
|---|---:|---:|
| Document completed | No | Mocked incomplete-response path completes; live Qwen run not measured |
| Incomplete evidence batch | Yes (9/10 returned) | Missing ID isolated and retried; exhausted IDs fail closed |
| Relationships lost to missing endpoints | 8 | 0 in artifact replay (8 conservatively reconstructed) |
| Raw `produces` relationships proposed | 4 | Live Qwen result not measured |
| Alias duplicates | Present | Covered by deterministic document-aware regressions; live count not measured |
| Accepted validated facts | 0 due to failure | Live Qwen result not measured |
| Final graph edges | 0 due to failure | Live Qwen result not measured |
| Completion trace | Audit truthful, graph ambiguous | Explicit processing/export/fact status covered by regression tests |

No precision, recall, F1, accepted-fact count, or final graph-edge count is
reported for a new model run because the configured vLLM endpoint was unavailable.
Prompt regressions demonstrate the intended manufacturing coverage and hard
negatives, but they are not substitutes for live-model benchmark metrics.

## Verification performed

```text
python3 -m pytest -q tests/test_ingest_reliability_recall.py
15 passed in 0.53s

python3 -m pytest -q tests/test_ingest_reliability_recall.py tests/test_reliability_fixes_v030.py tests/test_audit_trace_json.py tests/test_graph_integrity.py tests/test_resolve.py tests/test_validation.py tests/test_strong_llm_recall_upgrade.py tests/test_pipeline.py tests/test_ingest.py
118 passed in 13.65s

python3 -m pytest -q -m "not integration" --ignore=tests/test_server_scripts.py
623 passed, 4 deselected, 15 warnings in 154.21s
```

The unfiltered offline command was also run on Windows:

```text
python3 -m pytest -q -m "not integration"
634 passed, 41 failed, 4 deselected, 15 warnings in 189.98s
```

All 41 failures were in `tests/test_server_scripts.py`: Windows could neither
execute `.sh` files (`WinError 193`) nor pass its `C:\...` paths to Bash as POSIX
paths. The platform-independent rerun above passed. The configured local vLLM
endpoint at `127.0.0.1:8000` was not listening, so a fresh Qwen benchmark run was
not performed.

## Fresh Colab evaluation

Use a fresh runtime (or restart it after pulling the commit) so imported Python
modules cannot remain stale. The existing File Ingest notebook does not need code
changes. Pull or cherry-pick these changes onto `mirza`, install the repository as
the notebook normally does, and use a new run directory or `--restart` only for
that isolated benchmark run. Do not reuse the failed `EXTRACTED` checkpoint when
claiming an end-to-end comparison.

Enable `SCHAIN_AUDIT_TRACE=1` and `SCHAIN_AUDIT_FULL_CONTENT=1`, then verify that
the new audit contains targeted semantic retry events, endpoint recovery outcomes,
alias decisions, per-fact validation for repaired relations, non-zero failed-run
counters when applicable, and completion metadata matching the companion graph
JSON. A fair report must calculate precision/recall from the fresh accepted facts,
not from raw extracted relationships or this structural artifact replay.
