# SChain 0.3.0 — start here

This is an updated copy of `Schain-mirza1(1).zip`, reviewed against the main article and
all 11 additional supplied papers. The backend audit and corrected experiment policy are in `docs/AUDIT_V0.3.0.md`.
The earlier paper comparison remains in `docs/UPGRADE_REVIEW.md`. The frontend is unchanged.

## Open the graph immediately

Open `examples/graph-explorer.html` in a modern browser. It works offline and uses
fictional demo data, explicitly labeled. Press **Open graph JSON** and choose one or
several existing `supply_chain_graph.v1.json` files. The viewer does not send data away.

- The initial view shows supplier-to-customer edges, with upstream suppliers on the left.
- Choose one configured seed, a maximum tier, an entity type or an evidence state.
- Search by name or alias; matching entities and their immediate neighbors remain visible.
- Select a node to inspect it and optionally focus on its neighborhood.
- Select an edge for its exact source quotes, source links, model and grounding status.
- Use **Relationships** for a paginated table, or **Save SVG** for the visible drawing.
- **Save filtered JSON** preserves the current filtered subset for inspection.

The drawing is capped at 300 nodes / 1,000 edges, with an explicit message when capped.
The table and JSON include the entire filtered subset. This is a small/medium graph
explorer, not a benchmarked million-node renderer. Multiple loaded files are selected
one at a time, not merged. Browsers cannot silently scan a local directory; use the
file picker to select your exports. Offline demo source names are not real evidence.

## Build an HTML viewer for your own graph

From the extracted project directory (Python 3.10+):

```bash
python -m schain.viewer --input outputs/supply_chain_graph.v1.json --output outputs/my-graph.html
```

This standalone module uses only the Python standard library. Open `outputs/my-graph.html`.
The main CLI also offers the same operation after project dependencies are installed:

```bash
python -m schain view --input outputs/supply_chain_graph.v1.json --output outputs/my-graph.html
```

## Install and process actual documents

Use Python 3.10–3.14, with the declared dependencies. The validation environment used
Python 3.12 and neo4j-graphrag 1.19.0. The wider supported Python/dependency range has not
been tested as a matrix. Run these commands in the project root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
chmod +x scripts/*.sh
```

Set actual model endpoint/model ID in `config/base.yaml`, and credentials in `.env`.
The default model is `Qwen/Qwen3-4B-Instruct-2507`; its endpoint must be started separately
and serve that model ID. Real model availability and throughput were not measured here. `--no-neo4j` removes
only the database requirement; actual extraction still needs a working model endpoint.

Put your source `.txt`, `.md` or `.pdf` files in `data/corpus/`, then run:

```bash
python -m schain ingest --config config/ingest.yaml --input data/corpus --no-neo4j
python -m schain export --config config/ingest.yaml
python -m schain view --input outputs/supply_chain_graph.v1.json --output outputs/my-graph.html
```

For an actual Neo4j graph, configure Neo4j then run INGEST without `--no-neo4j`.
Use the same files/config to resume from resolved facts and commit them to the database.

EXPAND is node-centric and seed-driven. For the intended Colab profile:

```bash
python -m schain expand --config config/colab_professor_expand.yaml --no-neo4j --restart
```

Use `--restart` once for the new experiment; omit it to resume. Four active seeds are
Toyota, Tesla Inc., BMW and BYD Co., Ltd., all Tier 0. Tier 5 is terminal. The profile
uses 150 unique selected documents, a 110-minute cumulative deadline, tier-specific
targets and a quality-ranked beam. It explicitly caps extraction at four origin-ranked
chunks per document and records partial coverage. No second source is required.

To intentionally change seed selection, use repeated `--seed` arguments (safe for
names containing commas), or the compatible `--seeds` option. Seeds/depth/model/policy
changes require a new experiment. Operational ceiling changes can resume with an
audited config revision. Saved experiment settings govern later export.

Tier is the shortest defended supplier path to any configured seed. The existing viewer
is unchanged; use exported core/quality flags for backend supply-chain-core analysis.

## Evaluate extraction quality

Create a fully annotated held-out JSON file:

```json
{
  "documents": [
    {
      "doc_id": "actual-document-id",
      "triples": [
        {"subject_id": "actual-supplier-canonical-id", "relation": "suppliesTo", "object_id": "actual-customer-canonical-id"}
      ]
    },
    {"doc_id": "actual-negative-document-id", "triples": []}
  ]
}
```

Replace the illustrative IDs with actual IDs from your documents/graph. Independently
label **all ontology-allowed relations** in each included document, including negatives.
Keep these documents out of prompt/example tuning. Then:

```bash
python -m schain evaluate --config config/ingest.yaml --gold path/to/gold.json --output outputs/evaluation.json
```

This reports exact directed relation precision, recall and F1. Duplicate evidence does
not inflate the score. Predictions in documents outside the gold set are explicitly
excluded. It does not evaluate isolated entities, source truth or temporal validity.
The existing `validate` command measures external-reference overlap; it is not precision.

## Migration and reproducibility

1. Keep the original ZIP and existing datasets/checkpoints as your comparison baseline.
2. Extract this release into a separate directory. Do not mix old/new fact files when
   comparing changes. For Neo4j comparison, select a separate database or instance.
3. Existing graph JSON opens in the viewer. Missing legacy validation values remain unknown.
4. Resume retains completed document checkpoints. To apply new grounding rules or recover
   old discarded quotes, reprocess the source corpus in fresh storage paths. Chunk caches
   help recover incomplete extraction; they do not revalidate already completed documents.
5. `--restart` clears the configured local generated data/checkpoints. It does **not**
   delete old Neo4j nodes/edges. MERGE is additive and cannot retract obsolete evidence.
6. New fact IDs include source quote and chunk. The semantic graph still has one edge per
   supplier/relation/customer; evidence counts can increase without creating new edges.
7. Atomic JSON and Parquet replacements improve interruption recovery. SQLite/Parquet
   are still separate stores; concurrent writers to the same storage are unsupported.

No live model, retrieval providers or Neo4j integration was exercised in this review.
The included graph is synthetic. See `docs/TEST_RESULTS.md` for the actual test outcome.
