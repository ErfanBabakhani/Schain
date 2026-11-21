# Legacy ranking and Adaptive LinUCB benchmark

Use `selection-only` to measure candidate selection under controlled inputs. Both arms
visit the same configured seed companies in the same order and see the same frozen
title and snippet pool. Both use the same extraction, snippet batching, context limit,
validation, alias resolution, and duplicate handling. Adaptive can choose a snippet
or skip. Newly found companies appear in the relational results. They do not become
search targets. The fixed pool, exact token gate, or an operational stop ends the run.

Use `full-pipeline` to compare the complete expansion systems. Both arms start with
the same seeds and optional baseline facts and call the existing Expand scheduler.
Each arm can discover and search different companies from the frozen inventory.
Legacy retains its configured document limits and stopping rules. Adaptive uses its
normal processing actions and token pacing. Both have the same hard input token limit.
An arm can end with budget left when its scheduler reaches another stopping condition.

## PowerShell setup

Use Python 3.10–3.14 with the project and the adaptive extra installed. Run the commands
from the checkout. Configure the existing model endpoint in the benchmark config or an
overlay. If the endpoint uses an alias for the model, set `llm.tokenizer_model` to the
actual matching tokenizer and ensure its chat template matches the serving configuration.
Cache that tokenizer and the configured BGE embedding model before the experiment.
The tokenizer and BGE loader use local files during replay. Install the configured spaCy
NER package and use a fixed model or serving revision across repetitions.

```powershell
Set-Location D:\research\kuo\project\Schain
python -m pip install -e '.[adaptive,dev]'
python -m spacy download en_core_web_sm
$cfg = 'config/selection_benchmark.yaml'
$root = 'data/selection-benchmark/bmw-experiment-v1'
$selection = "$root/inputs/selection-v1.json"
$full = "$root/inputs/full-v1.json"
```

The default config uses the seeds in `config/expansion.yaml`. To benchmark a single
company, use this example overlay and assign its path to `$cfg` before preparation.
Use `llm` and `expansion.selection` in an overlay to pin model and bandit settings.

```powershell
New-Item -ItemType Directory -Path "$root/inputs" -Force | Out-Null
@'
include: [config/selection_benchmark.yaml]
seeds:
  ev_manufacturers: [BMW]
'@ | Set-Content -Encoding utf8 "$root/inputs/bmw.yaml"
$cfg = "$root/inputs/bmw.yaml"
```

## Record inputs and validate them

`prepare` makes live provider requests and can download source pages. It records
ordered query plans, provider parameters, individual provider results, ranks,
shortlisted candidates, and content hashes. Query planning during this preparation uses
the existing deterministic heuristic. Shared preparation uses zero LLM requests and
its own temporary workspace. A recorded failure is available in the archive when
`--allow-search-errors` is explicitly selected. This option accepts partial provider
coverage for the dataset. Without it, a failed provider query stops preparation.
Choose a new dataset filename and version when inputs change.

```powershell
python -m schain.benchmark prepare --mode selection-only --config $cfg --dataset $selection --version selection-v1
python -m schain.benchmark dry-run --mode selection-only --config $cfg --dataset $selection
```

For full pipeline, pass `--company 'TIER:NAME'` for each company and tier you want
covered beyond the seed targets. The recorder freezes each company's search stages
and the outcome of full text retrieval, including fetch failures. A run that reaches
a missing company, tier, or download reports an explicit replay error. The inventory
below is an example for BMW. Supply the relevant inventory for the intended depth.
To start from an existing accepted graph, append `--initial-facts PATH` to either
preparation command. It reads an SChain facts Parquet export and embeds a verified copy.
Each method receives its own copy. Without this option, both begin with an empty graph
and the configured seeds.

```powershell
python -m schain.benchmark prepare --mode full-pipeline --config $cfg --dataset $full --version full-v1 --company '1:CATL' --company '1:Panasonic' --company '2:LG Chem'
python -m schain.benchmark dry-run --mode full-pipeline --config $cfg --dataset $full
```

`dry-run` validates checksums and config compatibility. It makes zero search or
LLM requests, loads zero model weights, and creates no repetition state. Use it
before sending real work to the model.

## Execute and resume

Only `run` makes LLM requests. The input token allowance is **200,000 per method,
per seed, per mode** by default. Use the same `--budget`, `--warmup`, and
`--batch-size` in both arms when overriding defaults. `--seed` identifies the
independent repetition and seeds Adaptive's warmup sampling. The legacy ranker
is deterministic. Endpoint generation can still vary across requests.

```powershell
foreach ($seed in 42, 43, 44) {
    python -m schain.benchmark run --mode selection-only --policy legacy --config $cfg --dataset $selection --workspace $root --seed $seed
    python -m schain.benchmark run --mode selection-only --policy adaptive_linucb --config $cfg --dataset $selection --workspace $root --seed $seed
}

python -m schain.benchmark run --mode full-pipeline --policy legacy --config $cfg --dataset $full --workspace $root --seed 42
python -m schain.benchmark run --mode full-pipeline --policy adaptive_linucb --config $cfg --dataset $full --workspace $root --seed 42
```

For example, resume Adaptive seed 42 with the same inputs and settings as before.

```powershell
python -m schain.benchmark run --mode selection-only --policy adaptive_linucb --config $cfg --dataset $selection --workspace $root --seed 42 --resume
```

Every repetition has its own directory under the workspace. Reuse the original
directory and settings on resume. Budget, learner, and processed document checkpoints
persist there. Press Ctrl+C once to request a stop at the next work boundary.
`--session-minutes` adds an optional session limit. Both stops produce incomplete
results. A forced termination may leave a request charged without a reusable response
and may interrupt the normal Expand checkpoint sequence. The next invocation
retains the previous spend. Reports inspect the repetition lock to tell an active
session from an interrupted one. Elapsed time for an abrupt shutdown is an
estimate from the last saved database write. Resume preserves the original token allowance.

Input accounting uses the configured tokenizer and chat template for each outbound
model request, including retries in the extraction or validation code. SDK retries are
disabled in this benchmark. The first request that cannot fit is recorded with the
remaining balance and is stopped before reaching the provider. The ledger records output
tokens when the provider supplies usage metadata and counts how many responses
contained that metadata. A model may be running elsewhere, but retrieval replay
does not issue live provider searches or article downloads.

## Compare and review

```powershell
python -m schain.benchmark report --workspace $root
```

The report command reads stored artifacts and never invokes the experiment. Read
`reports/comparison.json`, `reports/repetitions.csv`, `reports/summary.csv`, and
`reports/paired_differences.csv`. Each repetition also has its own JSON and CSV.
Means and sample standard deviations use completed repetitions with matching
dataset, experiment settings, code, and tokenizer fingerprints. Paired differences
use seeds with completed runs from both methods.

The primary metric counts distinct, evidence supported Company-to-Company
`suppliesTo` edges in supplier-to-buyer direction, after subtracting baseline
edges. The report includes yield per 10,000 input tokens and target coverage.
Full pipeline tier counts use the shortest upstream path to a seed and omit
disconnected companies. The warmup and post-warmup rows attribute an edge to
its first qualifying witness. Check the insufficient learned performance flag
before drawing conclusions about the trained policy.

Give reviewers `reports/review.csv` or `reports/review.json`. Keep
`reports/review_mapping.private.json` with the experiment operator. Each blinded
relationship includes evidence and a randomly assigned ID. Reviewers can fill
`true`, `false`, or `unsure` in the label column of a copy of the CSV.

```powershell
python -m schain.benchmark report --workspace $root --labels "$root/reports/review-labeled.csv"
```

Human precision is the fraction of correct relationships among those labeled
`true` or `false`. The report includes the denominator and shows an unknown
precision when no usable labels have been supplied. The experiment measures
accepted yield and reviewed precision. Measuring recall requires a reference dataset.

## Local tests

```powershell
python -m pytest -q tests/test_selection_benchmark.py tests/test_adaptive_config.py tests/test_adaptive_selection.py tests/test_adaptive_budget_reservations.py tests/test_input_token_budget.py tests/test_snippet_batch.py
```
