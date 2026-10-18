# Pipeline/runtime implementation handoff

Owned implementation complete (after root audit approval). No frontend changes; no live experiment.

## Files changed

- schain/checkpoint.py — shared atomic_write_parquet(frame,path), temporary sibling/fsync/replace. Other agents reuse it for primary artifacts.
- schain/pipeline.py — no-driver RESOLVED skips without recounting; remains committable later with Neo4j; rediscovered changed text at original URL resumes saved unfinished original document ID/text; atomic resolution Parquet; clears effective run snapshot on explicit restart; actual measured call/stage/document metrics; optional chunk cap and upstream selection context; immutable extraction metrics and cumulative per-doc telemetry saved under extractions/metrics/.
- schain/kg/extract.py — extraction calls/cache-hit/retry/chunk-count/coverage telemetry; explicit configurable retry count; never retries runtime-deadline exception; optional positive max_chunks_per_document. With known origin, chunks rank through the shared executable direction policy (upstream > ambiguous > downstream); selected chunk order and IDs preserve original indices. Without origin uses document order. Full normalized source remains intact, omitted coverage is visibly logged and persisted.
- schain/llm/provider.py — transparent MeasuredLLM invocation meter, per-stage counts and elapsed; cooperative runtime deadline blocks new sync/async calls; explicit SDK max_retries=0 and rate_limit_max_attempts=1 defaults eliminate prior nested retry multiplication; explicit max_tokens forwarding/validation; request/retry settings logged; endpoint logs remove credentials/query.
- schain/ingest.py — successful-doc metric totals in summary.
- scripts/server_run.sh — preserves original argv across tmux/srun/sbatch; smoke tier1; intentionally selected weight override is served under configured API model alias so requests match; generic model-loading wording.
- tests/test_pipeline.py — configured model provenance assertion.
- tests/test_pipeline_reliability.py — noNeo skip/later commit, original-URL retry, metrics/full-source preservation, atomic Parquet interrupted-write protection, upstream chunk choice, bounded retries, restart snapshot removal, validator transport retry, deadline guard.
- tests/test_extract.py — output/retry controls.
- tests/test_server_scripts.py — tier1/new configured model, spaced argv preservation through batch/tmux and interactive plan, weight override alias.

## Validation

Before edits: 42 original pipeline/reliability/ingest/upgrade/reference tests passed (6.62s).
After edits: 59 pipeline/reliability/extract/ingest tests passed (10.20s).
After edits: 52 server-script tests passed (83.85s), local HTTP/TCP stubs only. Early run failed because root's new external-network guard also blocked loopback; guard corrected to permit local stubs and complete suite passed. One added validation test's expected exception text was corrected to actual fail-closed retry message.
Compileall succeeded after changes.

## Runtime implications and limits

These metrics count application model invocations including failures; internal SDK/rate retries are disabled by default so they reflect request workload without hidden amplification. LLM seconds across concurrent chunks can exceed document wall time, which is intentional; scheduler uses wall elapsed. Runtime cap is cooperative: guard blocks new calls after deadline, but a request already in flight can finish/fail up to configured timeout; retrieval phase also checks boundaries. Colab concurrency1 and request timeout120 bound practical model overshoot to an in-flight request, not all remaining document chunks.

Per-doc metrics JSON has last_attempt (actual current-attempt accounting), stable top-level extraction_metrics (source coverage retained across validation-only resume), total_llm_calls and total_elapsed_seconds. Exporter agent has wired stable coverage metadata into graph documents. Chunk cap is optional and explicitly visible; it does not establish complete article coverage. Some upstream evidence crossing chunk boundaries can be missed; overlap and better models are experiments, not proven accuracy improvements here.

No measured real T4 runtime/yield exists in repository. Do not use offline test runtime as model throughput or claim graph-size/accuracy improvements from stubs. Operational stage checkpointing is verified. A limited real endpoint/provider calibration is still needed before promising 90–120 minutes or five tiers actually discovered.

Optional not changed: runner dry-run creates logs/SLURM plan files, legacy external reference comparison not alias aware, Neo4j restart retains preexisting global graph (documented current compatibility policy). Root handles run snapshot/Colab config. Validation transport failure semantics were fixed by semantics owner.
