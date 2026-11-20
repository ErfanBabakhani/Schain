# SChain — Adaptive, Token-Budgeted Document Selection

**Implementation specification for Codex**  
Status: proposed update; implement and evaluate locally, not a claim of an existing implementation.  
Repository: `ErfanBabakhani/Schain` (use the checked-out branch and inspect its actual architecture before editing).

## 1. Objective and scope

SChain discovers multi-tier supply-chain knowledge graphs by searching for documents about a target entity, extracting ontology-constrained facts with an LLM, validating evidence, and expanding the graph. The update should improve the number of useful, validated supply-chain facts discovered per token and reduce unnecessary full-document downloads.

Keep the existing entity/frontier selection, tier-expansion semantics, ontology, graph storage, extraction model, evidence validation, and server/UI contracts wherever possible. Modify candidate retrieval limits and the document-selection stage, and add batched snippet extraction. Do not replace the existing extraction pipeline or introduce a new LLM for ranking.

The conceptual basis is the earlier project report, *Token-Budgeted Document Selection for SChain* (`report (1).pdf`): optimistic marginal graph reward minus an adaptive virtual token price, subject to a hard token budget. Its three processing arms are snippet, full text, and both. This implementation preserves that formulation while adding an explicit linear reward estimator and a practical batching mechanism. Do not claim that the AUPD regret guarantee transfers to graph-dependent rewards, adaptive candidate sets, or batched feedback without a separate proof.

**Primary experiment:** the current SChain selector with a 200-candidate pool versus the proposed selector with the same 200-candidate pool and the same token budget. The goal is not to improve results merely by allowing more search results or more LLM tokens.

## 2. Configuration and compatibility

Add a selectable policy, e.g. `legacy` versus `adaptive_linucb`. Names below describe desired semantics; map them to the repository's actual configuration and naming conventions rather than creating parallel, conflicting settings.

```yaml
# Illustrative configuration, not a claim about existing config keys.
retrieval:
  max_candidates_per_entity: 200
  results_per_query: 30                 # requested, when provider supports it
  max_queries_per_entity: 12            # configurable, preserve query planner
  max_candidates_per_domain: null       # no domain-based quota
  max_processed_documents_per_entity: null  # optional user-defined upper limit
  deduplicate_candidates: true
selection:
  policy: adaptive_linucb               # legacy remains available
  zero_gain_early_stopping: false       # adaptive mode only
  actions: [snippet, full_text, both]   # skip is always available
  context_dimension: 8
  reward_parameter_count: 24
  warmup_candidates: 200                # total across project/model lifetime, not per entity
  initial_action_prior:
    skip: 0.30
    snippet: 0.60
    full_text: 0.05
    both: 0.05
  warmup_temperature: 1.0
  random_seed: 42
  linucb_alpha: 0.5
  ridge_lambda: 1.0
  persist_model: true
  cost_reference_tokens: 1000
  primal_reward_weight_V: 1.0
  initial_virtual_queue: 0.0
embedding:
  model: BAAI/bge-small-en-v1.5
  device: cpu
  batch_size: 32
  cache: true
ner:
  model: en_core_web_sm
extraction:
  snippet_batch_size: 10                # maximum; dynamically reduce if needed
  full_document_batch_size: 1
  batch_validation: true                # reuse existing support where available
reward:
  new_supply_edge: 1.0
  independent_corroboration: 0.2
  verified_correction: 1.0
  new_auxiliary_edge: 0.1
  clip_max: 5.0
budget:
  metric: input_tokens                  # matches earlier report; log output too
  total_input_tokens: null              # must be supplied for budgeted experiments
  per_entity_input_tokens: null         # optional additional cap
  target_tokens_per_candidate: null     # derive only from a known budget/horizon
  max_tokens_per_document: null          # optional, separate from request cap
  max_input_tokens_per_request: null     # derive from actual model context limit
  max_output_tokens_per_request: null    # use existing safe generation settings
```

Required semantics:

- Collect up to **200 unique candidate documents per entity**, not 200 processed documents or 200 per query. Request more candidates from search providers where supported, paginate if available, and deduplicate before applying the pool cap. A provider returning fewer results is not an error.
- Remove domain count restrictions in this experiment. Do not restrict a useful source merely because several documents come from its domain. Deduplicate the same URL or identical/syndicated content using appropriate existing mechanisms; duplication and domain quotas are different concepts.
- The maximum number of processed documents is optional, default `null`. If a user sets it to 20, it is a hard upper limit of 20 unique processed documents, not a target or a requirement to process all 20. Without a configured limit, selection and budget determine the count. Keep a separate count of attempted documents/calls.
- Remove old implicit document limits, fetch limits, and zero-gain patience **from the adaptive path only** when they would silently stop the selector. Preserve real safety limits, network timeouts, context limits, cancellation, and explicitly configured limits. Inspect all call sites; changing one configuration field may not remove a hidden truncation elsewhere.
- Keep the legacy policy unchanged except for shared, explicitly configured candidate retrieval and experiment budget controls. Do not change its extraction behavior to make the proposed method look better.
- Preserve existing saved project settings and API compatibility; new fields need defaults and validation. Expose policy, candidate cap, optional document limit, budgets, batch size, and initial action preferences through the existing settings mechanism. Encode unlimited values consistently with the repository's conventions.
- Do not silently treat an unset global budget as zero or invent a budget. Adaptive mode must have a clear hard budget for budget-controlled experiments. If unrestricted runs are supported, label them explicitly and define how cost pacing works for them.

## 3. Processing actions and the 24-parameter model

For each candidate `d`, choose exactly one action during an expansion decision:

| Action | Input to extraction LLM | Initial preference |
|---|---|---:|
| `skip` | None; no fetch or extraction | 30% |
| `snippet` | Title plus available search snippet(s), separately identified | 60% |
| `full_text` | Retrieved full document content (with ordinary provenance/target metadata) | 5% |
| `both` | Full document content plus title/search snippet(s), explicitly separated in one extraction input | 5% |

The combined share of actions requiring a full-document retrieval is 10%. The 5%/5% split is a **configurable design default**, not a user-mandated ratio. `both` is included to preserve the three processing arms of the earlier report and the requested 3 × 8 learned weight matrix. `skip` is a fourth, deterministic no-processing option; it does not need a learned reward model. If `both` is disabled, there are only 16 active reward coefficients; do not describe those as 24 learned coefficients.

Do not interpret these probabilities as quotas, enforce a fixed count of each action, process every candidate, or perform snippet-then-upgrade as an additional sequential policy. For this version, `both` means one joint extraction input, not two extraction passes. Deduplicate near-identical snippet/body passages where practical while preserving provenance. For a full-text input that exceeds context limits, use the existing chunking/selection strategy within the budget and clearly log any truncation. Never label snippet-only fallback as successful full-text extraction.

The embeddings below always use title and snippet as inexpensive features. Even a skipped candidate may be embedded and scored. That is not the same as submitting it to the extraction LLM.

## 4. Exactly eight document-context features

Let `x(d, v, H)` be an 8-dimensional column vector for document `d`, target company `v`, and current graph/evidence history `H`. Compute features using only information available **before choosing the extraction action**. Do not use full text, subsequent extracted facts, future graph state, or rewards from the current action.

| Index (1-based) | Feature | Definition |
|---:|---|---|
| 1 | Bias | Constant 1. |
| 2 | Supplier relevance | Maximum normalized cosine similarity to target-specific upstream/supplier-list reference queries. |
| 3 | Irrelevant content | Maximum normalized cosine similarity to target-specific downstream, aftermarket, or generic procurement-guideline queries. |
| 4 | Target company match | Binary indicator that a normalized target name or known alias occurs in the title/snippet (word/phrase boundaries; no arbitrary substring matches). |
| 5 | Official corporate source | Binary indicator from an existing or curated known-company/domain mapping; an official corporate site may belong to the target or another identified company. Unknown/unmapped sources = 0. |
| 6 | Search rank | Reciprocal of the best original provider/query rank (1-based) observed for the URL. |
| 7 | Document novelty | One minus maximum clipped cosine similarity to previously successfully processed title/snippet embeddings relevant to this target; 1 if no relevant history. |
| 8 | Entity novelty | Fraction of distinct recognized organization mentions not already resolved to a company in the current graph; 0 if no recognized organizations. |

All eight features are bounded in [0, 1]. `supplier_relevance` and `irrelevant_content` are two distinct signals; the bandit learns their coefficients and signs. For semantic relevance, use `clip(cosine, 0, 1)` consistently (do not change feature scaling between runs). For entity novelty, normalize organization names and use existing canonical IDs/aliases conservatively; NER is a proxy, not a validation of company identity. Exclude empty and generic placeholder mentions with conservative rules. Do not create graph nodes during context calculation.

No additional feature dimensions for tier, company supplier count, provider categories, historical yield, source quality, token cost, or the raw embedding. Cost is handled by the primal-dual decision rule, not learned as a reward feature. These are deliberately omitted to keep the first model compact. Missing/empty snippets should use the title if available; if neither exists, log an unscorable candidate and skip it. Log unknown official-source mapping and NER fallback status, but do not add dimensions. If the NER dependency is unavailable, fail clearly in benchmark mode or use an explicitly configured deterministic fallback; do not silently change feature definitions across evaluation runs.

### 4.1 Embedding implementation

Use frozen `BAAI/bge-small-en-v1.5` via the repository's existing embedding stack if compatible, otherwise a lightweight `sentence-transformers` integration. Encode title and snippet concatenated with a clear separator, with normalization and model-appropriate truncation. The embedding is 384-dimensional; it is not the bandit's context dimension. Batch all candidate encodings (default inference batch size 32). Do not fine-tune BGE or Qwen.

Suggested reference strings (substitute canonical company name and, when needed, an unambiguous alias):

```text
Positive:
- Companies supplying products directly to {company}
- Lists of named suppliers and upstream procurement contracts for {company}

Negative:
- Customers purchasing products from {company}
- Aftermarket replacement parts compatible with {company} products
- Generic supplier requirements and procurement guidelines without named suppliers
```

Apply the BGE retrieval query instruction `Represent this sentence for searching relevant passages: ` to reference queries, not to the candidate passages. Compute similarities using matrix multiplication on normalized vectors. Encode N candidates and K references, not N×K text pairs. Cache by model/version, normalized input and instruction; cache reference embeddings per company. Cache static similarities, ranks and organization mentions. Recompute graph-dependent novelty features after each feedback batch; use incremental updates to maximum similarities when practical.

## 5. Reward model: three independent 8D rows

Use the traditional disjoint linear model, with three processing arms but a **single shared-across-entities model state**. This deliberately supersedes the earlier 11-parameter partially shared proposal. There are no separate bandits per entity or tier.

For `a ∈ {snippet, full_text, both}`:

```text
E[r(d,a) | x,H,a] ≈ w_a^T x(d,v,H)
W = stack(w_snippet, w_full_text, w_both) ∈ R^(3×8)
```

`W` contains 24 learned reward coefficients. The model also stores regularized sufficient statistics, not additional free reward coefficients:

```text
For each a:
    A[a] = ridge_lambda * I_8       # shape (8,8)
    b[a] = zeros(8)                 # shape (8,)
    w[a] = solve(A[a], b[a])        # shape (8,)
```

Equivalent matrix shapes: `A[3,8,8]`, `b[3,8]`, `W[3,8]`. A block-diagonal 24D action-encoded linear model is mathematically equivalent. Prefer the simpler per-action statistics unless existing abstractions make the block form easier.

For each eligible document-action pair:

```text
mu = x @ w[a]
sigma = sqrt(max(0, x @ solve(A[a], x)))
U = mu + alpha * sigma
```

For a matrix of N document contexts `X[N,8]`, vectorize predictions and uncertainty using linear solves with multiple right-hand sides. Never compute an explicit inverse inside the candidate loop. `skip` has utility, marginal reward and incremental extraction cost 0, and no statistics.

After observing reward `r` for the actual selected processing action, update only that row:

```text
A[a] += outer(x_at_decision, x_at_decision)
b[a] += r * x_at_decision
w[a] = solve(A[a], b[a])
```

This is online ridge regression. No gradient training, backpropagation, external labels or LLM fine-tuning is required. Use the immutable decision-time context, not a context recomputed after graph mutation. Do not give unselected actions reward zero as a training label. A successful extraction that finds no useful fact is real feedback with zero reward.

Persist state and action counts across entities/tiers and resume within a project. Reset or explicitly initialize equivalent state across experimental replicates. Do not silently share learned weights between unrelated projects or between baseline and proposed runs. Version the feature mapping/action set and reject incompatible checkpoints or reset with explicit logging.

**Statistical limitation:** each row learns only from its own selected documents. At 200 candidates, the nominal warmup gives about 120 snippet, 10 full-text and 10 both observations before budget/score effects. Rare arms may be poorly estimated. Keep ridge and uncertainty; monitor counts and learning curves. If these rows are too sparse, evaluate the previously discussed partially shared or two-processing-arm model as an ablation, not an undisclosed replacement of the requested 24-parameter design.

## 6. Selection and adaptive budget control

At decision time use the current graph and history. The objective is maximize expected cumulative validated marginal graph gain under a hard token budget; with clipping, the optimized reward is a bounded proxy for that gain.

For each eligible processing pair compute:

```text
S(d,a) = V * U(d,a) - Q * (estimated_cost(d,a) / c_ref)
S(d,skip) = 0
```

`V` is a fixed configurable reward weight; `Q ≥ 0` is the virtual token price; `c_ref` is a fixed token normalization scale. High `Q` favors cheaper actions. This is the normalized implementation of the report's reward-minus-cost rule, not a new optimization objective.

### 6.1 Initial preferences / warm-up

During the first `warmup_candidates` distinct finalized/attempted candidate decisions (including explicit skips) in a fresh model, take candidates in a deterministic order (e.g. current retrieval ranking with stable tie-breaks). Sample the action from a **prior-weighted softmax**:

```text
logit(a) = log(p0[a]) + S(d,a)/temperature
P(a|d) = softmax(logit(a) over feasible actions)
```

Use stable log-sum-exp, mask infeasible actions, omit actions whose configured prior is zero from warm-up, and always retain `skip` as the safe option. If scores are equal and all actions are feasible, probabilities coincide with the configured prior. The realized proportions need not equal 10/60/30 because scores, availability and remaining budget change. Use a seeded RNG and log the actual sampled probabilities and outcomes. No fixed per-entity quotas or forced allocations. A warm-up skip is a completed candidate decision and provides no reward-model label.

After warm-up, use the optimistic greedy rule over the remaining eligible document/action pairs, not a permanent prior or probability quota:

```text
(d*, a*) = argmax of S(d,a), subject to hard feasibility
if best score <= 0 or no feasible pairs: stop current pool
else: process the winning pair
```

In the greedy phase do not permanently discard a candidate merely because its current score is negative; it may become useful after a graph/model update. Finalize remaining candidates as skipped when the pool terminates, and record the reason. Do not manufacture counterfactual labels for their processing arms.

### 6.2 Token cost estimation and hard reservations

Keep cost separate from reward (no reward-per-token division inside the reward model). The earlier report defines cost in input tokens; use that as the default experiment metric. Record input and output tokens, latency, and search/fetch counts as separate metrics. Charge all LLM calls attributable to the run (including query planning, extraction, validation, retries, etc.) against the global input-token budget, not just successful extractions. Report fixed and selection-related costs separately; identical fixed overhead does not justify excluding it from the hard cap. If using a different budget metric, configure it identically for all methods and label it.

- Snippet cost: tokenize prompt/ontology plus title and snippet; for batching estimate the shared prompt contribution at planned batch size, including IDs/separators and estimated validation overhead. Recompute at actual batch size before launch.
- Full/Both cost before fetch: use a conservative configurable estimate based on available metadata and/or previous observed full-text costs and existing input limits. Do not fetch all 200 full documents to estimate costs. After choosing a full action, retrieve only that document, then tokenize actual constructed prompt. For Both include extra title/snippet text. Log prediction versus actual usage.
- Keep an upper reservation for each pending request/batch, including planned validation or independently reserve validation/retry calls before launching them. Atomically enforce `spent + reserved <= budget`, both global and optional per-entity caps. Release unused reservation when actual usage is known. If metadata does not provide actual usage, use a conservative tokenizer/accounting estimate and mark the source. No silent overspend on retry. Respect the model context limit including output allowance.
- If actual full-text cost does not fit remaining budget, use budget-safe existing truncation/chunking if permitted and logged, or cancel the processing action and rescore. Do not secretly run a different action or train on fabricated feedback. Fetch failures may be retried within bounded existing policies; if the action cannot execute, log it distinctly. Decide consistently whether operational failures are counted as zero yield when evaluating end-to-end throughput; exclude malformed/missing extraction results from reward-model updates unless valid feedback can be established.
- Enforce any optional per-document token cap separately from the per-request cap. In a merged snippet request, allocate shared prompt and validation cost consistently across participating documents (e.g. equal overhead split plus each document's own tokens, with deterministic rounding). Always reserve and charge the full request exactly once. The sum of assigned costs must equal charged costs; the batch is not free simply because a prompt is shared.

### 6.3 Virtual queue and meaning of the average budget

Maintain the report's normalized queue update:

```text
Q <- max(0, Q + (actual_cost - b_target)/c_ref)
```

Apply it once per finalized candidate decision, including a skip with cost zero, using cost allocated to that document. For a known decision horizon `T` and available selection budget `B_sel`, set `b_target = B_sel/T` (T counts candidate decisions, not just selected documents). Subtract or reserve non-selection overhead from the available selection budget. Do not mix the number of selected documents and the number of candidates in this denominator. When an entity batch is subject to an explicit budget, `T` can be the actual deduplicated batch size and `B_sel` its available selection budget. If only a global budget is known and the number of future entities/candidates is unknown, require a configured pacing target or a clearly defined remaining-budget/estimated-horizon schedule; do not claim a known exact horizon. Keep the queue across candidate batches within its budget window; reset only for a new explicitly defined budget window.

A skip's zero cost here is **incremental extraction cost**; shared search/query-planning overhead is already charged to the run. If a pool terminates, finalize distinct skipped candidates once for candidate-based accounting; do not repeatedly decay the queue on the same candidate or reopen a stopped pool just to make its scores positive. Queue pacing is a heuristic and does not itself enforce the hard budget. If a user additionally wants a strict average **per processed document** or a per-request maximum, implement that separately with an explicit definition/constraint; do not equate it with this candidate-based target.

## 7. Batched title/snippet extraction

Goal: amortize the shared ontology, instructions, output schema and target-company prompt across multiple short documents. Default maximum batch size: 10; full/both documents remain single-document jobs. Combine snippets only for the same target entity and compatible extraction schema/settings in the first implementation.

- Gather selected snippet jobs until the batch is full, another action should take priority, the token/doc limit binds, or the pool ends; flush partial batches. In the greedy phase, start with the globally best feasible pair. If it is a snippet, add further snippet pairs based on score while no higher-priority non-snippet pair should run. Limit to one in-flight action per document. Warm-up may queue sampled snippets, flushing before a sampled full/both job and at the end of the pool. Never force all selected jobs to wait for a full batch.
- Evaluate batch feasibility with exact constructed prompt tokens, conservative output allowance, validation budget, and pending reservations before invoking the LLM. Recompute costs for partial batches. Reduce batch size if required. The score uses estimated amortized cost, whereas the reservation enforces the true aggregate cost. This greedy batching is an implementation heuristic; it does not solve an exact combinatorial batch knapsack.
- Freeze the chosen context and action when each job is committed; no feedback-dependent model update while the batch is being assembled. After the batch returns, validate and commit relations per document using a deterministic selected order, compute each document's marginal reward, update statistics for each observed reward, then recompute graph-dependent features/scores before the next selection batch. This is batched/delayed feedback, not strictly immediate feedback.

Use explicit immutable document IDs, target company, and separate input blocks. Instruct the extraction model to treat document content as data, to extract only relations directly supported by the individual document, not mix evidence across IDs, and to return an empty array for documents with no supported facts. Reuse the existing ontology/schema and validation mechanisms rather than introducing incompatible relation fields. Illustrative response shape (adapt field names):

```json
{
  "documents": [
    {"document_id": "d1", "facts": []},
    {
      "document_id": "d2",
      "facts": [
        {"subject": "Supplier A", "predicate": "suppliesTo", "object": "Target B",
         "evidence": "Exact supporting text from d2"}
      ]
    }
  ]
}
```

Reject unknown IDs, duplicate conflicting IDs, cross-document or missing evidence and malformed output. A missing document in an otherwise usable response is not a known zero-reward document. Retry/split only within budget and cancellation rules; log failures independently. Preserve per-document URL, provider, exact evidence quotation and evidence scope (`snippet`, `full_text`, or `both` plus actual quote origin where possible). Snippets may truncate or distort claims; do not accept a relation just because it was extracted. Preserve provenance and allow downstream full-source verification. In Both mode deduplicate facts produced by the two evidence scopes.

## 8. Reward design: validated marginal graph improvement

Do not reward all extracted entities or relations indiscriminately. Use only facts accepted by the existing evidence/ontology validator and canonicalized against the current graph. Acceptance is a proxy for validity, not independent ground truth. The policy should favor new upstream supply-chain information and useful evidence rather than graph inflation.

| Event | Raw reward (default) |
|---|---:|
| New, accepted, target-relevant upstream `suppliesTo` edge | +1.0 |
| New independent evidence for an existing relevant supply edge | +0.2 / max(1, n) |
| Verified, committed correction of a previously wrong relevant relation | +1.0 |
| New, useful, accepted auxiliary ontology edge connected to the relevant graph | +0.1 |
| Duplicate fact, same-source/syndicated corroboration, rejected evidence, generic non-company entity, irrelevant relation | 0 |

`n` is the number of prior independent supporting source groups for the edge. Different URLs are not automatically independent; use conservative content/source provenance grouping. Where independent corroboration or correction is not supported reliably in existing code, record it but default that component to zero until a real check/transaction is implemented. Do not invent corrections or source independence. A correction must actually improve the graph and be independently validated according to the existing policy. Do not reward a correction twice as a new edge plus correction for the same logical change. Auxiliary gain is deliberately small; only include ontology-valid relations that contribute to the relevant supply-chain graph, not unrelated facts collected from a page.

Compute reward against the graph state **before committing that document**; then commit accepted changes. For a multi-document snippet batch, use the fixed selection order for credit allocation, progressively updating the graph so that multiple documents asserting one edge cannot all receive novelty reward. Count each canonical directed edge at most once per document; distinguish upstream supplier from downstream customer, aftermarket compatibility, generic requirements, and general corporate collaboration. Existing canonicalization/direction rules remain authoritative. Graph changes and reward logging must be atomic or safely idempotent.

```text
raw_reward = 1.0 * N_new_supply
           + sum(0.2 / max(1,n_e) for each new independent support)
           + 1.0 * N_verified_corrections
           + 0.1 * N_new_auxiliary
r = min(raw_reward, 5.0)/5.0
```

Example: a batch returns the same edge from three documents, of which the second is independent and the third is a syndicated copy. Raw rewards are 1.0, 0.2, and 0; normalized rewards 0.20, 0.04, and 0. If the first document also yields a new useful `produces` edge, its raw reward becomes 1.1 and normalized reward becomes 0.22.

Save raw components and actual uncapped edge counts as well as normalized reward. Clipping bounds the target for LinUCB but saturates high-yield documents; compare unclipped validated graph gain in the experiment and optionally ablate clipping. Do not penalize hallucinations with negative reward in this first implementation; rejected facts receive zero, and false-positive rates are evaluated independently. Cost belongs in the virtual-price rule, not in the reward formula.

## 9. Integration flow and pseudocode

Inspect existing config, retrieval, expansion, ingestion, extraction, validation, graph update, persistence, and server entrypoints before coding. Integrate at the existing document-selection boundary. Prioritize Expand first; isolated reusable selection/scoring code should support a future Ingest integration without changing ingestion semantics by default. For Ingest, any context not obtainable before processing needs a documented adapter; do not silently substitute new features or modify the ontology.

```text
initialize project policy, budget ledger, queue, model/checkpoint
for target_entity in existing frontier order:
    collect up to configured unique candidate cap using current search planner
    embed title/snippet candidates and target-specific reference queries once
    cache static features; initialize remaining candidate pool

    while remaining and not cancelled:
        respect optional processed-document cap and hard budget
        X = build feature matrix(static metadata, current graph/history)
        if warmup active:
            sample feasible actions using prior softmax in deterministic candidate order
            finalize explicit skips; form at most one processing batch
        else:
            score eligible (document,action) pairs with current W/A/Q
            if no positive feasible pair: finalize pool as skipped; break
            select best pair; if snippet, collect eligible snippet jobs as in §7

        freeze context, action, cost estimates, selection order, scores
        verify exact request fit and reserve tokens before every call
        invoke existing extraction path (batched snippet or single full/both)
        validate per-document outputs using existing ontology/evidence rules
        settle actual token costs and release unused reservations
        for each finalized decision in stable order:
            if valid extraction result:
                compute marginal reward and commit graph changes idempotently
                update only the selected action's A and b using frozen x
            if skip or failure: handle outcome explicitly, no fabricated labels
            update queue with allocated actual cost (once per decision)
            save telemetry, checkpoint, and candidate state
        refresh graph/history features for unresolved candidates
        continue until no useful feasible actions, explicit cap, or budget exhaustion

    return accepted facts through existing graph/frontier integration
finalize run metrics; persist state; preserve existing stop/pause/resume behavior
```

Implementation detail: warm-up skip decisions update the queue once when finalized; if preparing a snippet batch, either flush before changing decision phases or buffer pending outcomes and commit in original decision order. Do not count a candidate twice if a request is retried. Reserve pending document slots as well as token costs to prevent exceeding the optional document cap. Keep unprocessed candidates eligible until final termination. Explicit user cancellation must stop outstanding work safely and preserve recoverable state.

## 10. Logging, persistence and observability

Record structured per-candidate decisions, including skips, and per-LLM-request accounting. Reuse existing project DB/storage and migrations. At minimum record:

| Field group | Required information |
|---|---|
| Identity | Run/project ID, entity ID, tier, candidate/document ID, canonical URL, provider/query provenance and original rank |
| Decision | Policy, phase, immutable 8D context with feature/version names, selected action, estimated reward, uncertainty, score and scores for alternative feasible actions, estimated cost, warm-up probabilities/seed, skip/failure reason |
| Execution | Batch/request IDs, actual evidence scope, input/output tokens and accounting source, allocated per-document cost, retrieval status, retries, timings, truncation |
| Feedback | Canonical accepted/rejected relation counts, marginal new supply edges, corroboration/correction/auxiliary components, raw and normalized reward, validation status |
| State | Selection-order index, model action counts and checkpoint version, queue before/after, global and entity budget spent/reserved/remaining |

Persist model statistics, feature map version, RNG state (or reproducible counter/seed), decision counters, queue/window state and candidate processing status. Use existing transactional or idempotent mechanisms so crash/resume does not double-apply reward updates, facts or charged costs. Keep checkpoint writes consistent with graph commits and use replayable decision IDs if atomic cross-store transactions are unavailable. Never serialize credentials, provider keys or arbitrary private request headers into telemetry.

Expose a minimal run summary in existing status/export interfaces: candidate count, skip/snippet/full/both counts, actual full downloads, successful/failed extraction counts, total tokens, remaining budget, batch sizes/calls, accepted new upstream edges, reward by action, and stop reason. Do not require a UI redesign. Preserve old exports and add fields in a backwards-compatible way.

## 11. Experimental protocol

### Required main comparison

| Component | A: Legacy SChain | B: Proposed SChain |
|---|---|---|
| Seed company, ontology, max tier | Same | Same |
| Candidate limit per target entity | 200 | 200 |
| Query planner, retrieval/providers | Same configuration | Same configuration |
| Domain cap | None | None |
| Candidate snapshots for common targets | Identical | Identical |
| Extraction LLM and validation rules | Existing Qwen 4B / existing validator | Same Qwen 4B / existing validator |
| Selection | Existing selector and stopping behavior | 24-weight LinUCB + primal-dual |
| Extraction input strategy | Existing legacy behavior | Snippet / full text / both / skip |
| Hard token cap and accounting | Identical | Identical |
| New optional processed-document cap | Unset unless explicitly requested for both | Unset unless explicitly requested for both |

Do not impose a fixed 20-document count. If legacy has its own intrinsic stopping rules, preserve them and report actual token use; do not silently modify the baseline by introducing a minimum number of processed documents. A total budget cap is not a requirement to spend every token. Report output-versus-consumed-token curves so early stopping does not create a misleading equal-consumption claim.

Cache raw search responses/candidate snapshots per entity and shared query configuration where feasible. Store content snapshots or consistent retrieval metadata for common URLs. In closed-loop recursive expansion, algorithms may discover different targets; do not force them to visit the same frontier solely to reuse search caches. Reuse common-target results and record search costs, including how offline caching was used. Compare end-to-end graph outcomes and optionally add a fixed-pool selector experiment to isolate document selection. No training on outcomes from the competing test run.

Use the same extraction/validation versions and budget accounting for both methods. Evaluate on several seeds/runs if resources permit, with separate model/graph/queue state and controlled seeds. BMW is a natural first pilot, not the only evidence of generalization. If provider coverage or timestamps differ, disclose it. Compare performance at several feasible token checkpoints; log warm-up versus later performance separately.

### Metrics

Primary: cumulative independently verified unique relevant upstream supply edges at a fixed input-token budget; independently verified edges per 1,000 input tokens. Report the operational proxy (validator-accepted canonical edges per token) separately. Additional metrics: valid suppliers per tier, verified reachable depth, precision from a blind manual/reference audit of sampled or pooled relations, unresolved/unverifiable claims, duplicate/false relation rate, total graph reward (raw and clipped), actual input and output tokens, provider searches, full-text fetches, LLM call counts, wall-clock runtime, and action distribution. If an exhaustive reference graph is unavailable, do not claim true recall; label any reference coverage as such.

The main A/B comparison evaluates the combined update. It cannot attribute a gain uniquely to adaptive action choice, ranking or batching. Optional ablations on the same candidate pools and caps: snippet-only, full-text-only, adaptive without snippet batching, and random or current rank-based selection with the same action mix and batching. The original smaller-pool legacy run can be a historical reference, not the main controlled baseline. Compare actual batch sizes and evidence quality, not just nominal call reduction. Use separate fresh checkpoints for ablations.

## 12. Required tests and acceptance criteria

Add deterministic unit and integration tests using mocked search providers, LLM responses and token usage. No live services or GPU should be required for the core algorithm tests. Use existing project test conventions and run relevant regression tests.

| Area | Required check |
|---|---|
| Retrieval | More than the old candidate limit survives the pipeline; cap is 200 unique documents; no domain cap; deduplication retains provenance; provider limits are handled safely. |
| Optional caps | Unset processed-document cap does not introduce an arbitrary limit; a configured cap of 20 prevents a 21st attempted distinct processing job; pending batch jobs count toward the cap. |
| Context | Exact ordered 8D vector, finite values in [0,1]; deterministic feature computation; empty/missing text handling; cached queries; dynamic novelty updated only from available graph/history. |
| Model | A shape (3,8,8), b/W shape (3,8); 24 learned coefficients; numerical solve agrees with ridge regression; only selected arm updates; a true zero reward updates A, but skip/missing outcomes do not train any row. |
| Warm-up | Equal-score feasible actions reproduce configured prior probabilities; seeded random behavior; masked infeasible actions never chosen; warm-up counter persists across entities; proportions are not quotas. |
| Selection | Higher-reward feasible pair wins at equal cost; increasing Q can favor a cheaper alternative; zero/negative best score stops the greedy pool; unresolved candidates can be rescored after feedback. |
| Budgets | No global/entity/request cap violation with mocks, including batch prompt overhead, retries, validation, cancellation, partial batches, oversized full docs, and failed fetches; cost shares sum exactly to request costs. |
| Queue | Normalized update matches formula; known candidate horizon uses correct denominator; skip accounted once; unknown horizon requires explicit pacing policy; no repeated queue decay on finalized candidates. |
| Batch | Ten compatible snippets can use one extraction request; partial flush works; each output maps to its input ID; no cross-document evidence; missing or malformed output is not fabricated zero feedback. |
| Reward | New relevant edge rewarded once; duplicate zero; independent support diminishing; unrelated/invalid relation zero; accepted auxiliary fact correctly weighted; clipping correct; multiple documents in a batch don't double-count novelty. |
| Full/Both | Selected full document alone is fetched; Both has separate snippet and body text in one logical extraction; fallback never mislabeled; accepted facts deduplicated; no hidden extraction path bypasses budget. |
| Persistence | Resume restores statistics, phase/counters and ledger; repeated decision/retry cannot duplicate edge reward or cost; feature-version mismatch detected. |
| Regression | Legacy selection remains available; old settings load; ontology, direction validation, BFS/frontier, exports, pause/resume/cancel still work; no unrequested deployment. |

Acceptance: deliver an implementation only when the configuration, both selectors, reward model, snippet batching and budget ledger are connected end to end, tests have run, and a reproducible pilot command/config exists. Do not report synthetic test data as real experimental findings.

## 13. Codex execution instructions

1. Inspect the checked-out repository and project instructions. Locate all current candidate/result caps, per-domain restrictions, document-count limits, retrieval and expansion loops, extraction/validation interfaces and token counters. Produce a short mapping from this spec to actual modules before changing them. Preserve unrelated local changes.
2. Implement incrementally: shared retrieval configuration and legacy compatibility; 8D context builder; three-arm LinUCB state; budget ledger/queue; adaptive selector; snippet batch extraction and per-document reward; persistence and telemetry. Reuse existing components rather than duplication.
3. Add unit tests for each component and an integration test of the full pipeline. Make model downloads/dependencies explicit and optional for legacy mode. Avoid changing the default legacy behavior for existing configurations unless required by an explicit selected policy.
4. Provide example configs and a small reproducible experiment harness for A/B runs. Use local mocks/fixtures for tests; do not invoke billable LLM/search APIs, push changes, deploy to the remote server, or overwrite running project data without explicit authorization.
5. Return a concise implementation report: changed files, actual configuration key names, how to run legacy and proposed policies, test commands/results, cost/accounting assumptions, behavior that remains unimplemented, and any conflicts with this spec. Do not invent repository paths or claim tests passed if not run.

## 14. Decisions and open issues

- **Fixed for this update:** 200 candidates per entity by default; optional processed-document cap, no domain quota; eight document features; three learned processing arms / 24 reward coefficients; skip outside the learned matrix; online ridge / LinUCB, primal-dual cost control; batched snippet extraction; legacy-vs-adaptive evaluation.
- **Chosen defaults, not mandated optimal values:** 5% full-only and 5% both within the requested 10% full-retrieval preference, 200-candidate warm-up, maximum snippet batch size 10, alpha 0.5, ridge 1, V 1, token normalization 1,000, reward weights and clipping. All should be configurable and validated in a pilot. The 24 coefficients are not a guarantee of adequate samples, particularly for rare processing arms.
- **Need explicit run settings:** global token budget and optional per-entity or per-document caps; queue pacing target/horizon when the candidate horizon is unknown. If the existing project's mean/max token settings express a different constraint, preserve their semantics separately, document the mapping, and do not quietly replace them with the candidate-based queue target.
- **Implementation limitations to check:** existing entity canonicalization, independent-source detection, correction workflow, token accounting coverage, provider pagination, and Qwen 4B's batch extraction reliability. Corroboration/corrections must be zero-reward when not reliably implemented. A useful batch size is empirical.
- **Theory:** preserve the report's objective and reward-minus-cost mechanism. The linear feature mapping, batched selection, prior-weighted warm-up, action-dependent rewards, and graph-dependent marginal gain introduce modeling assumptions; no original AUPD theorem or regret bound is asserted for this implementation.
- **Scope:** this spec describes desired changes, not verified current repository code. Check the actual branch before applying any names or assumptions. The immediate deliverable is a local, tested Expand implementation and an evaluation harness; reuse in Ingest is a follow-up unless existing integration is straightforward and backwards compatible.
