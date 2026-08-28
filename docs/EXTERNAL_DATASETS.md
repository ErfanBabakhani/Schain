# External datasets

## Sectivia supply-chain dataset (validation reference)

**Status, plainly:**

- Sectivia was selected **conceptually** as V1's auxiliary supplier-customer validation
  dataset, on the strength of the architecture brief's citation (roughly 350 directed
  supplier-to-customer relations across ~214 listed companies, CSV/JSON, CC BY 4.0).
- Its **current official public download source could not be verified.** Five searches
  (below) found no dataset, repository record or organisation of that name.
- **No dataset was fabricated, and none was silently substituted for it.** No URL, file or
  record has been invented, and no other dataset has been wired in as a stand-in. The
  nearest open candidate the searches surfaced, the TheyBuyForYou procurement knowledge
  graph on Zenodo, is public-tender RDF rather than a company-to-company supplier list for
  listed firms; it answers a different question, so using it would be a scope change
  requiring an explicit decision, not a fix.
- `data/external/sectivia/` is the **expected location** if the dataset is obtained later.
  The directory is gitignored, so nothing gets committed.
- The validator accepts the **documented supplier/customer CSV or JSON schema** below, so
  the dataset (or any equivalent edge list) plugs in without code changes.

Nothing else in the pipeline depends on this file: only `python -m schain validate` reads
it. INGEST, EXPAND and the graph output are unaffected by its absence.

Searches performed (August 2026), none returning a match:

- `Sectivia open supply chain dataset supplier customer relationships CC BY 4.0 download`
- `Sectivia supply chain dataset github 350 supplier relations 214 companies`
- `"Sectivia" company open dataset supply chain`
- `Sectivia dataset zenodo OR figshare OR huggingface supply chain supplier customer graph open data`
- `open access supplier customer relationship network dataset listed companies CC BY download zenodo`

If you have the original reference (a paper, a Zenodo/figshare record, or a vendor page),
add it here and place the files as described below. Replacing Sectivia with a different
dataset should be an explicit decision, recorded in PROJECT_DECISIONS.md.

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
