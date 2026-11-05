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

For local development, keep these settings in a private root-level `.env.local` (safe placeholders are in `.env.example`). Create it only if absent; do not overwrite an existing file:

```dotenv
SCHAIN_API_URL=http://127.0.0.1:18001
SCHAIN_API_TOKEN=replace-with-your-existing-backend-token
SCHAIN_SSH_HOST=your-ssh-alias
SCHAIN_FRONTEND_PORT=8080
```

In Windows PowerShell, establish your existing SSH tunnel in one terminal, then start the local frontend in another:

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 18001:127.0.0.1:8001 your-ssh-alias
```

```powershell
python -m backend.dev_frontend --env .env.local
```

The command builds only changed frontend files, binds only to `127.0.0.1`, checks backend authentication and opens the explicit `/index.html?build=...` entry point. The page automatically loads projects through `/api`; Reconnect does not ask for credentials. Failed connections use bounded backoff. `--no-browser` leaves browser opening to you. Match `SCHAIN_API_URL` to the local SSH forwarding port; do not replace an existing token when changing ports. Restart the command after editing `.env.local`. If the port is occupied, the command identifies an existing SChain development server where possible; stop it in its original terminal or change `SCHAIN_FRONTEND_PORT`. It never terminates processes.

Development responses use `Cache-Control: no-store` and identify their mode/build in `X-SChain-Mode`, `X-SChain-Frontend` and the HTML `schain-frontend` meta tag. `/` and `/index.html` serve the same verified, immutable startup snapshot. Rebuilding or replacing files on disk cannot switch a running server to the demo. Restart after source changes to load a new snapshot. The demo remains a separate entry point under `demo/SChain_Graph_Explorer/`.

The Python process supplies the bearer token. The browser receives only a separate temporary anti-CSRF nonce. The local server checks Host/Origin, permits only the required API routes and methods, and serves only the generated frontend assets. It never starts SSH or reads SSH key files; the alias is just a reference for diagnostics. No backend update or CORS change is needed for this same-origin proxy, even if its local port changes.

This proxy is development-only and trusts processes/users on the local machine. Keep `.env.local` private and Git-ignored, do not share its contents, and do not expose or forward the local frontend port. It is not production authentication infrastructure.

For a static build with the existing direct authenticated connection instead, use the following commands. They do not read `.env.local`:

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

Open **http://127.0.0.1:8080**, select Connect, and enter the API token. The address defaults to **http://127.0.0.1:8001**. Copy the entire generated directory, including `assets/`, if using another computer. It needs only a static server after building. Direct `file://` access is unsupported.

The server interface is the dark Graph Explorer with a project selector and New Project action. It starts empty and never loads demo datasets. `visualization/` owns the shared renderer; the standalone demo and `python -m schain view` also use it. Offline JSON import stays separate from authenticated server projects.

If another interface appears, check the absolute directory printed by the build command and restart the local static server with that exact `--directory`. Old servers can keep port 8080 occupied. On Linux/macOS use `lsof -nP -iTCP:8080 -sTCP:LISTEN`; on Windows use `Get-NetTCPConnection -LocalPort 8080 -State Listen`, then `Get-Process -Id <OwningProcess>`. Stop only the identified old static server, or choose port 8081 and set backend `SCHAIN_CORS_ORIGINS=http://127.0.0.1:8081`. Rebuild and hard-refresh the page; generated asset URLs carry a content revision to avoid stale JavaScript and CSS.

Create a project, set its seeds and limits, then select Start. Pause and wait for `paused` before changing settings or uploading UTF-8 `.txt`, `.md`, or PDF documents up to 10 MiB. Resume continues expansion or retries a paused/failed ingestion job. Seeds and maximum tier cannot change after the first job; create a new project for a different experiment.

The API stays on the server; frontend files stay local. Keep the default loopback binding. For direct static connections only, changing the local frontend port requires matching `SCHAIN_CORS_ORIGINS` on the backend.

## Storage and limits

Each project has its own SQLite database, documents, extraction checkpoints and run history. SQLite stores accepted facts, nodes, edges and evidence; Parquet files remain compatible with the research workflow and can be repaired from accepted checkpoints.

Budget exhaustion pauses the run. An exhausted frontier completes without further searches. The frontend loads all graph pages, including isolated nodes, and checks revisions between pages before replacing the displayed graph. Status polls do not download an unchanged graph. Graph publication currently rebuilds the semantic bundle after each document. Large graph layouts and real model accuracy need validation on the target workload.

## Updating an existing server

After the changes have been reviewed, committed and pushed, use the existing checkout:

```bash
git switch server_section
git pull --ff-only origin server_section
. .venv/bin/activate
python -m pip install -e '.[server]'
```

Pause active projects and wait for `paused` before restarting the worker. Restart the API and worker through their existing service manager or scheduler with the same private environment and `SCHAIN_DATA_DIR`. For foreground processes, stop the API with Ctrl+C, send SIGTERM/Ctrl+C to the worker, wait for exit, then rerun the two commands in the Server section. If user systemd units are already named `schain-api` and `schain-worker`, use:

```bash
systemctl --user restart schain-api.service schain-worker.service
systemctl --user status schain-api.service schain-worker.service
```

Use your actual unit/job names if different. Keep the existing token, uploads, model files and runtime directory. This frontend update requires no SQLite schema migration; existing projects remain compatible. Rebuild the local frontend from the matching updated checkout.

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

Browser tests: install `.[server,dev,browser]` and run `python -m pytest tests/test_frontend.py -q`. They use installed Edge on Windows or Playwright Chromium elsewhere; `SCHAIN_BROWSER_EXECUTABLE` can select an existing Chromium-compatible executable. Tests skip explicitly if browser tooling is unavailable.

Shell deployment tests require Linux. Model, Neo4j and PostgreSQL integration tests require their respective services; ordinary unit tests use fixtures.

Research workflows: [Colab](docs/SAVE_AND_COLAB_RESUME.md), [distributed expansion](docs/DISTRIBUTED_NEON_EXPAND.md), [external datasets](docs/EXTERNAL_DATASETS.md).
