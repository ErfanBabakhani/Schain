# External datasets

## Sectivia supply-chain dataset (validation reference)

The architecture brief names the Sectivia open supply-chain dataset as the external
reference for sanity-checking supplier-customer relationships: roughly 350 directed
supplier-to-customer relations across ~214 listed companies, released as CSV/JSON under
CC BY 4.0.

**Status: no public download location was found, so nothing is downloaded automatically.**

Searches performed (August 2026), all without a hit for a dataset or organisation of that
name:

- `Sectivia open supply chain dataset supplier customer relationships CC BY 4.0 download`
- `Sectivia supply chain dataset github 350 supplier relations 214 companies`
- `"Sectivia" company open dataset supply chain`

No URL has been guessed or invented here. If you have the original reference (a paper,
a Zenodo/figshare record, or a vendor page), add it to this file and place the files as
described below.

### How to use it once obtained

Place the edge list under `data/external/sectivia/` (gitignored, so nothing is committed):

```text
data/external/sectivia/
└── relations.csv
```

The loader accepts CSV, JSON or JSONL and needs only two columns. Column names are
matched case-insensitively from these alternatives:

| Role     | Accepted column names                                  |
|----------|--------------------------------------------------------|
| supplier | `supplier`, `supplier_name`, `source`, `from`, `seller` |
| customer | `customer`, `customer_name`, `target`, `to`, `buyer`    |

Minimal example:

```csv
supplier,customer
CATL,BMW AG
Ganfeng Lithium Co Ltd,CATL
```

Company names are normalized with the same canonical-id rule the graph uses, so
`BMW AG` and `BMW` compare equal.

### Running the comparison

```bash
.venv/bin/python -m schain validate --config config/ingest.yaml
```

It prints how many reference edges we also extracted (`coverage`), how many companies the
two sets share, and examples of matched and missing edges. Extracted edges absent from the
reference are counted but not treated as errors: the reference covers only its own company
set, so most of our edges are simply outside its scope.

Any other supplier-customer edge list works the same way — pass it with `--reference`.

## AlMahri evaluation corpus

AlMahri et al. evaluate on a manually annotated corpus of 1,277 Wikipedia sentences
covering six entity types and four relation types. **We do not have that annotated
package, and it has not been reconstructed or approximated here.** If it becomes
available, it can be ingested directly:

```bash
.venv/bin/python -m schain ingest --config config/ingest.yaml --input path/to/corpus
```

The 13 selected companies from that work are kept in `config/expansion.yaml` — the OEM
seeds are active, and the battery and mining groups are commented out just below them for
strict selected-company replication.
