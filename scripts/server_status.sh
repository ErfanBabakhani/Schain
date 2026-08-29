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

echo
echo "== services =="
if [ -x "$VENV_PY" ] && values="$(config_values "$CONFIG_FILE" 2>/dev/null)"; then
    eval "$values"
    item "neo4j" "$(neo4j_reachable "$CFG_NEO4J_URI" && echo "reachable ($CFG_NEO4J_URI)" || echo "not reachable ($CFG_NEO4J_URI)")"
    item "llm endpoint" "$(llm_healthy "$CFG_LLM_BASE_URL" && echo "healthy ($CFG_LLM_BASE_URL)" || echo "not healthy ($CFG_LLM_BASE_URL)")"
else
    item "config" "not readable (.venv missing?)"
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
