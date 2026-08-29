#!/usr/bin/env bash
# Shared helpers for the SChain server scripts.
#
# Deliberately small. Nothing here duplicates pipeline logic: extraction, expansion,
# checkpointing, resume, deduplication and Neo4j writes all stay in the Python CLI.
# These functions only locate the repository, read configuration through the project's
# own loader, and check whether services are up.

# --- repository and environment ----------------------------------------------------

# Repository root, resolved relative to this file. No absolute paths are hard-coded.
repo_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

REPO_ROOT="$(repo_root)"
VENV_PY="$REPO_ROOT/.venv/bin/python"

log()  { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '%s  WARNING: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '%s  ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# Load .env into the environment. Values are never printed by these scripts.
load_env() {
    local env_file="$REPO_ROOT/.env"
    [ -f "$env_file" ] || return 0
    set -a
    # shellcheck disable=SC1090
    . "$env_file"
    set +a
}

# "set" / "not set", so a secret's presence can be reported without its value.
secret_state() { [ -n "${!1:-}" ] && echo "set" || echo "not set"; }

# --- project configuration ---------------------------------------------------------

# Read the values the scripts need straight from the project's config loader, so the
# shell and the application can never disagree about model, endpoint or paths.
# Usage: eval "$(config_values config/expansion.yaml)"
config_values() {
    local config_file="$1"
    "$VENV_PY" - "$config_file" <<'PY'
import shlex
import sys

from schain.config import load_config, neo4j_settings, resolve_path

config = load_config(sys.argv[1])
llm = config.get("llm", {})
neo4j = neo4j_settings(config)


def emit(key, value):
    print(f"{key}={shlex.quote(str(value))}")


emit("CFG_LLM_PROVIDER", llm.get("provider", "local"))
emit("CFG_LLM_MODEL", llm.get("model", ""))
emit("CFG_LLM_BASE_URL", llm.get("base_url", ""))
emit("CFG_NEO4J_URI", neo4j["uri"])
emit("CFG_NEO4J_DATABASE", neo4j["database"])
emit("CFG_STATE_DB", resolve_path(config["storage"]["state_db"]))
PY
}

# --- service health ----------------------------------------------------------------

# http://host:8000/v1 -> http://host:8000
llm_origin() { printf '%s' "${1%/v1}"; }

# bolt://host:7687 -> host 7687
neo4j_host() { local u="${1#*://}"; printf '%s' "${u%%:*}"; }
neo4j_port() { local u="${1##*:}"; printf '%s' "${u%%/*}"; }

# TCP reachability, using bash's own /dev/tcp so no extra tool is required.
tcp_open() {
    local host="$1" port="$2" timeout="${3:-2}"
    timeout "$timeout" bash -c "exec 3<>/dev/tcp/$host/$port" 2>/dev/null
}

# A TCP connect only proves that something is listening on the port - any unrelated
# process holding 7687 would look like Neo4j. The reuse-or-start decision depends on this
# answer, so confirm with the project's own driver, exactly as the pipeline will.
neo4j_reachable() {
    local uri="$1"
    tcp_open "$(neo4j_host "$uri")" "$(neo4j_port "$uri")" 2 || return 1
    [ -x "$VENV_PY" ] || return 0     # cannot verify further; the port is open
    [ -n "${SCHAIN_NEO4J_SKIP_BOLT_CHECK:-}" ] && return 0
    # Hard timeout: the driver completes the TCP connect and then waits indefinitely for
    # a Bolt handshake that a non-Neo4j listener never sends.
    timeout "${SCHAIN_NEO4J_PROBE_TIMEOUT:-8}" "$VENV_PY" - "$uri" <<'NEO4JCHECK' 2>/dev/null
import os
import sys

import neo4j

try:
    with neo4j.GraphDatabase.driver(
        sys.argv[1],
        auth=(os.environ.get("NEO4J_USERNAME", "neo4j"),
              os.environ.get("NEO4J_PASSWORD", "")),
        connection_timeout=5,
    ) as driver:
        driver.verify_connectivity()
except Exception:
    raise SystemExit(1)
NEO4JCHECK
}

# vLLM exposes /health; any OpenAI-compatible server exposes /v1/models.
llm_healthy() {
    local base_url="$1" origin timeout="${SCHAIN_HTTP_PROBE_TIMEOUT:-3}"
    origin="$(llm_origin "$base_url")"
    curl -fsS --max-time "$timeout" "$origin/health"   >/dev/null 2>&1 && return 0
    curl -fsS --max-time "$timeout" "$base_url/models" >/dev/null 2>&1 && return 0
    return 1
}

# --- GPU and scheduler -------------------------------------------------------------

gpu_count() {
    have nvidia-smi || { echo 0; return; }
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' '
}

# PIDs of the pipeline itself.
#
# Matching on argv fields rather than on the raw command line matters: a plain
# `pgrep -f "m schain"` also matches any shell whose own command line happens to mention
# it, including the script doing the checking. Here argv[0] must be a python binary and
# argv[1..2] must be "-m schain", which only a real pipeline process satisfies.
schain_pids() {
    [ -d /proc ] || return 0
    local pid args
    for pid in $(pgrep -f -- "-m schain" 2>/dev/null); do
        [ -r "/proc/$pid/cmdline" ] || continue
        mapfile -d "" -t args < "/proc/$pid/cmdline" 2>/dev/null || continue
        case "${args[0]:-}" in *python*) ;; *) continue ;; esac
        [ "${args[1]:-}" = "-m" ] && [ "${args[2]:-}" = "schain" ] && echo "$pid"
    done
}

in_slurm_allocation() {
    # True inside srun/sbatch, and when server_run.sh re-invoked itself in a job.
    [ -n "${SLURM_JOB_ID:-}" ] || [ -n "${SCHAIN_IN_ALLOCATION:-}" ]
}
slurm_available()     { have sbatch || have srun; }

# Where a submitted production job records its id, so status and stop can find it.
slurm_job_file() {
    printf '%s' "${SCHAIN_RUNTIME_DIR:-$REPO_ROOT/.runtime}/slurm/current_job"
}

# Java major version, or empty when java is missing. Neo4j 5 needs 21 or newer.
java_major() {
    have java || return 0
    java -version 2>&1 | head -1 | sed -n 's/.*version "\([0-9][0-9]*\).*/\1/p'
}
