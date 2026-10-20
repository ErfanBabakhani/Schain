# Deep discovery and safe Neon resume

This change extends the existing `bmw-supply-chain-v2` experiment. It does not create,
clone, reset, or delete an experiment. The approved profile is
`deep_discovery_v1`, the model remains `Qwen/Qwen3-4B-Instruct-2507`, and Neo4j Save
remains disabled.

## Effective limits

| Tier | document target | document maximum |
|---:|---:|---:|
| 0 | 80 | 200 |
| 1 | 80 | 160 |
| 2 | 60 | 120 |
| 3 | 50 | 100 |
| 4 | 40 | 80 |
| 5 | terminal | 0 |

The profile permits 20,000 cumulative experiment documents, 250 full-text attempts,
100 title-resolution attempts, and 600 ranked candidates per company. It starts with
28 upstream-oriented queries and can add component, LLM-planned, and official/PDF
families in four bounded rounds. Diminishing-return stopping requires source-domain and
query-family coverage; a zero-yield entity runs the bounded recovery ladder instead.

## One-time operation

1. Back up the Neon database.
2. Stop every SChain worker and wait for task/outbox leases to expire.
3. Check status and inspect the read-only plan:

   ```bash
   python -m schain.distributed.maintenance status --experiment-id bmw-supply-chain-v2 --json
   python -m schain.distributed.maintenance extend --experiment-id bmw-supply-chain-v2 --new-max-tier 5 --topup-tier 1 --plan
   ```

4. Confirm the plan contains Tier-0 Toyota recovery, all Tier-1 companies, the four
   terminal Tier-2 companies, no active workers/leases, and no immutable manifest
   mismatch.
5. Apply once:

   ```bash
   python -m schain.distributed.maintenance extend --experiment-id bmw-supply-chain-v2 --new-max-tier 5 --topup-tier 1 --apply --confirm-id bmw-supply-chain-v2
   ```

The apply transaction takes an experiment advisory lock, repeats worker and lease
checks, archives the previous manifest, records the new profile and cumulative limit,
adds generation-specific top-up tasks, and reopens terminal nodes. Existing completed
tasks, documents, facts, aliases, and evidence are not updated or deleted. Repeating the
same apply returns the stored maintenance result.

Use
[`SChain_Neon_v2_SAFE_RESUME_TIER5_AFTER_CODEX.ipynb`](../notebooks/SChain_Neon_v2_SAFE_RESUME_TIER5_AFTER_CODEX.ipynb)
for Colab. It uses `resume_existing_only=True`, 40-task sessions with optional safe
auto-continue, streaming logging, the approved config, and Save disabled.

## Frontier/path audit

Every accepted `suppliesTo` fact is recorded in `schain_discovery_paths` with its path
depth and one of these outcomes: new unique supplier, existing entity at the same or a
shallower tier with a new path, shorter path, relationship to another customer,
self-edge, or valid fact omitted from Frontier. Global `schain_frontier.tier` remains the
minimum observed upstream distance. The relationship/path record is retained even when
the supplier already exists, so peer and cyclic discoveries do not duplicate or move a
company to a deeper global tier.

## Reproducible comparison

Export before/after snapshots and run:

```bash
python -m schain.distributed.benchmark before.json after.json --labels labelled_edges.json
```

The report includes unique verified suppliers, upstream edges, document relevance,
full-text ratio, false negatives, precision, runtime, and cost. Precision and false
negatives are `null` without labels; no improvement is assumed or invented.

## Remaining operational risks

- Web providers can rate-limit, change results, or block full-text retrieval.
- The cumulative document ceiling is checked before a task claim; concurrently leased
  tasks can finish after another task reaches the ceiling.
- PostgreSQL integration tests require a disposable `SCHAIN_TEST_POSTGRES_URL`; never
  point tests at the production Neon database.
- The migration deliberately refuses inferred equivalence between `company:toyota` and
  `company:toyota-motor`. Only an existing evidenced alias can resolve that identity.
