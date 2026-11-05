# SChain

Supply-chain knowledge graph extraction and upstream expansion, with a SQLite API and a local graph explorer.

## Server

Python 3.10–3.14; the commands below use Python 3.12. Run from the repository root.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[server]'

export SCHAIN_DATA_DIR=/path/to/durable-local-disk/schain
mkdir -p "$SCHAIN_DATA_DIR"
chmod 700 "$SCHAIN_DATA_DIR"
export SCHAIN_SERVER_CONFIG="$PWD/config/server.yaml"
export SCHAIN_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

python -m uvicorn backend.api:create_app --factory --host 127.0.0.1 --port 8001
```

Set the model endpoint and model name in `config/server.yaml` before creating projects. The endpoint must already be running and support OpenAI-compatible chat completions with JSON responses. Model weights and inference services are managed separately.

Keep the token in a private environment file outside the repository and reuse it after restarting the API. Store runtime data outside the checkout on a durable local filesystem, not NFS or temporary scratch. Databases initialize automatically and use SQLite rollback journals.

In another terminal, activate the same environment and start the worker with the same data directory:

```bash
. .venv/bin/activate
python -m backend.worker --data-dir /path/to/durable-local-disk/schain --cpu-threads 1
```

Run both processes under an approved service manager or scheduler to survive SSH disconnection. Do not run persistent services on a university login node without permission. SIGTERM requests a graceful worker stop; after an unexpected exit, restart it with the same directory. Only one worker can own that directory.

## Local frontend

From a matching checkout on your own computer:

```bash
python -m pip install -e .
python -m backend.frontend --output outputs/server-frontend
python -m http.server 8080 --bind 127.0.0.1 --directory outputs/server-frontend
```

Open an SSH tunnel in another local terminal:

```bash
ssh -N -L 8001:127.0.0.1:8001 your-account@server
```

Open **http://127.0.0.1:8080**, connect to **http://127.0.0.1:8001**, and enter the token. The two generated HTML files can also be copied to another computer and served with Python. Direct `file://` access is unsupported.

Create a project, set its seeds and limits, then select Start. Pause and wait for `paused` before changing settings or uploading UTF-8 `.txt`, `.md`, or PDF documents up to 10 MiB. Resume continues expansion or retries a paused/failed ingestion job. Seeds and maximum tier cannot change after the first job; create a new project for a different experiment.

The API stays on the server; frontend files stay local. Keep the default loopback binding. If the local frontend port changes, set `SCHAIN_CORS_ORIGINS` to its exact origin before starting the API.

## Storage and limits

Each project has its own SQLite database, documents, extraction checkpoints and run history. SQLite stores accepted facts, nodes, edges and evidence; Parquet files remain compatible with the research workflow and can be repaired from accepted checkpoints.

Budget exhaustion pauses the run. An exhausted frontier completes without further searches. The frontend shows up to 5,000 edges; the graph API supports pagination. Graph publication currently rebuilds the semantic bundle after each document. Real model accuracy and deployment-specific resource limits must be checked on the target server.

For backup, pause projects, stop the worker and API, and copy the entire runtime directory. Restore with both services stopped. Do not run the CLI against an active server project's files.

## CLI and tests

The existing CLI uses its own configuration and storage paths:

```bash
python -m schain ingest --config config/ingest.yaml --input /path/to/documents --no-neo4j
python -m schain expand --config config/expansion.yaml --no-neo4j
python -m schain view --input outputs/supply_chain_graph.v1.json --output outputs/graph.html
```

```bash
python -m pip install -e '.[server,dev]'
python -m pytest tests -q
```

Shell deployment tests require Linux. Model, Neo4j and PostgreSQL integration tests require their respective services; ordinary unit tests use fixtures.

Research workflows: [Colab](docs/SAVE_AND_COLAB_RESUME.md), [distributed expansion](docs/DISTRIBUTED_NEON_EXPAND.md), [external datasets](docs/EXTERNAL_DATASETS.md).
