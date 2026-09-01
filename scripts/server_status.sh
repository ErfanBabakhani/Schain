#!/usr/bin/env bash
# What is SChain doing right now? Read-only.
#
#   ./scripts/server_status.sh [config-file]

set -uo pipefail
# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

CONFIG_FILE="${1:-config/expansion.yaml}"
cd "$REPO_ROOT" || exit 1
load_env

TMUX_SESSION="${SCHAIN_TMUX_SESSION:-schain}"
item() { printf '  %-22s %s\n' "$1" "$2"; }

echo "== run =="
if have tmux && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    item "tmux session" "'$TMUX_SESSION' active  (attach: tmux attach -t $TMUX_SESSION)"
else
    item "tmux session" "none (no run is active in tmux)"
fi
pids="$(schain_pids | tr '\n' ' ')"
item "schain process" "${pids:+running (pid ${pids% })}${pids:-not running}"

if slurm_available; then
    job_file="$(slurm_job_file)"
    job_id="$( [ -f "$job_file" ] && cat "$job_file" )"
    if [ -z "$job_id" ]; then
        item "slurm job" "none submitted from this checkout"
    else
        # squeue knows queued and running jobs; sacct also knows finished ones.
        live="$(squeue -h -j "$job_id" -o '%T on %N' 2>/dev/null)"
        if [ -n "$live" ]; then
            item "slurm job" "$job_id  $live"
            item "stop with" "scancel $job_id"
        elif have sacct; then
            item "slurm job" "$job_id  $(sacct -n -X -j "$job_id" -o State,Elapsed 2>/dev/null | head -1 | xargs)"
        else
            item "slurm job" "$job_id (no longer queued)"
        fi
    fi
fi

echo
echo "== services =="
if values="$(config_values "$CONFIG_FILE" 2>/dev/null)"; then
    eval "$values"
    item "neo4j" "$(neo4j_reachable "$CFG_NEO4J_URI" && echo "reachable ($CFG_NEO4J_URI)" || echo "not reachable ($CFG_NEO4J_URI)")"
    item "llm endpoint" "$(llm_healthy "$CFG_LLM_BASE_URL" && echo "healthy ($CFG_LLM_BASE_URL)" || echo "not healthy ($CFG_LLM_BASE_URL)")"
else
    item "config" "not readable (PyYAML missing or config error)"
fi
if have docker; then
    container="${SCHAIN_NEO4J_CONTAINER:-schain-neo4j}"
    item "neo4j container" "$(docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null | grep "^$container " || echo 'not running')"
fi

echo
echo "== checkpoint state =="
if [ -x "$VENV_PY" ] && [ -n "${CFG_STATE_DB:-}" ] && [ -f "$CFG_STATE_DB" ]; then
    "$VENV_PY" - "$CFG_STATE_DB" <<'PY'
import sys

from schain import state

connection = state.connect(sys.argv[1])
try:
    print(f"  {'documents':<22} {state.document_counts(connection) or 'none yet'}")
    print(f"  {'frontier':<22} {state.frontier_counts(connection) or 'none yet'}")
    print(f"  {'frontier by tier':<22} {state.frontier_tier_counts(connection) or 'none yet'}")
finally:
    connection.close()
PY
else
    item "state db" "not created yet (no run has started)"
fi

echo
echo "== logs =="
latest="$(ls -1t "$REPO_ROOT"/logs/*.log 2>/dev/null | head -1)"
if [ -n "$latest" ]; then
    item "latest" "$latest"
    echo "  --- last 12 lines ---"
    tail -12 "$latest" | sed 's/^/  /'
else
    item "latest" "no logs yet"
fi

if have nvidia-smi; then
    echo
    echo "== gpu =="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader 2>/dev/null | sed 's/^/  /'
fi
