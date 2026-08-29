#!/usr/bin/env bash
# One-command server runner for SChain.
#
#   ./scripts/server_run.sh                 # EXPAND (the main server experiment)
#   ./scripts/server_run.sh ingest <path>   # INGEST a file or directory
#   ./scripts/server_run.sh --smoke-test    # tiny run against the real services
#
# It prepares services and launches the existing Python CLI. It contains no pipeline
# logic: extraction, expansion, checkpointing, resume, deduplication and Neo4j writes
# all stay in `python -m schain`. Resume is the CLI's own default, so re-running this
# script after any interruption continues from the SQLite checkpoint.
#
# Services already running before this script started are reused and never stopped.
# Services this script starts are stopped when the workload finishes, so a 27B model
# does not sit on the GPU afterwards.

set -uo pipefail
# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
    cat <<'USAGE'
Usage: ./scripts/server_run.sh [mode] [options]

Modes:
  expand                 discover suppliers from the configured seeds  (default)
  ingest <path>          process documents already in hand

Options:
  --smoke-test           tiny run against the real LLM and Neo4j, then stop.
                         EXPAND: --max-tier 0 --max-documents 3.
                         INGEST: one small document.
                         Runs in the foreground so you see the result immediately.
  --restart              clear checkpoint state and generated data first (destructive
                         to intermediate data; Neo4j content is left alone)
  --config <file>        override the config file for the chosen mode
  --input <path>         input for ingest (same as the positional argument)
  --foreground           do not use tmux, even if it is available
  --dry-run              print the plan and the exact commands, change nothing
  -h, --help             show this help

Defaults come from config/base.yaml, config/expansion.yaml and .env. Server-specific
settings are SCHAIN_* variables in .env; see .env.example.

A full run detaches into a tmux session so it survives an SSH disconnect. The log path
is printed when it starts; ./scripts/server_status.sh shows how it is doing.
USAGE
}

# --- arguments ---------------------------------------------------------------------

MODE="expand"
INPUT=""
CONFIG_FILE=""
SMOKE=0
RESTART=0
FOREGROUND=0
DRY_RUN=0

case "${1:-}" in
    expand|ingest) MODE="$1"; shift ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --smoke-test) SMOKE=1; FOREGROUND=1; shift ;;
        --restart)    RESTART=1; shift ;;
        --foreground|--no-tmux) FOREGROUND=1; shift ;;
        --dry-run)    DRY_RUN=1; FOREGROUND=1; shift ;;
        --config)     CONFIG_FILE="${2:-}"; shift 2 ;;
        --input)      INPUT="${2:-}"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        -*)           die "unknown option $1 (see --help)" ;;
        expand|ingest)
            # A bare mode word is the mode wherever it appears, not an input path.
            if [ -z "$INPUT" ]; then MODE="$1"; else die "unexpected argument $1"; fi
            shift ;;
        *)            INPUT="$1"; shift ;;
    esac
done

# The config file names do not match the mode names, so map them explicitly.
if [ -z "$CONFIG_FILE" ]; then
    case "$MODE" in
        expand) CONFIG_FILE="config/expansion.yaml" ;;
        ingest) CONFIG_FILE="config/ingest.yaml" ;;
    esac
fi
cd "$REPO_ROOT" || die "cannot enter $REPO_ROOT"
load_env

# Server-specific settings: defaults here, overridden in .env. One obvious location.
TMUX_SESSION="${SCHAIN_TMUX_SESSION:-schain}"
VLLM_MAX_MODEL_LEN="${SCHAIN_VLLM_MAX_MODEL_LEN:-16384}"
VLLM_GPU_UTIL="${SCHAIN_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_EXTRA_ARGS="${SCHAIN_VLLM_EXTRA_ARGS:-}"
VLLM_STARTUP_TIMEOUT="${SCHAIN_VLLM_STARTUP_TIMEOUT:-1800}"
NEO4J_CONTAINER="${SCHAIN_NEO4J_CONTAINER:-schain-neo4j}"
NEO4J_IMAGE="${SCHAIN_NEO4J_IMAGE:-neo4j:5}"
NEO4J_DATA_DIR="${SCHAIN_NEO4J_DATA_DIR:-$REPO_ROOT/neo4j_data}"
NEO4J_STARTUP_TIMEOUT="${SCHAIN_NEO4J_STARTUP_TIMEOUT:-120}"
SMOKE_INPUT="${SCHAIN_SMOKE_INPUT:-docs/SupplyChain_KG_V1_Architecture_Brief_Final.pdf}"

# --- ownership, so cleanup only touches what this run started ----------------------

OWNED_VLLM_PID=""
OWNED_NEO4J=""

cleanup() {
    local status=$?
    if [ -n "$OWNED_VLLM_PID" ] && kill -0 "$OWNED_VLLM_PID" 2>/dev/null; then
        log "stopping vLLM (pid $OWNED_VLLM_PID) started by this run"
        kill "$OWNED_VLLM_PID" 2>/dev/null
        for _ in $(seq 1 30); do
            kill -0 "$OWNED_VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$OWNED_VLLM_PID" 2>/dev/null
    fi
    if [ -n "$OWNED_NEO4J" ]; then
        log "stopping Neo4j container '$OWNED_NEO4J' started by this run (data is kept)"
        docker stop "$OWNED_NEO4J" >/dev/null 2>&1
    fi
    # Checkpoint state and graph data are never removed here.
    exit $status
}
trap cleanup EXIT INT TERM

# --- 1. tmux: survive an SSH disconnect --------------------------------------------

if [ "$FOREGROUND" -eq 0 ] && [ -z "${TMUX:-}" ] && [ -z "${SCHAIN_NO_TMUX:-}" ]; then
    if have tmux; then
        if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
            die "a run is already active in tmux session '$TMUX_SESSION'.
  Attach:  tmux attach -t $TMUX_SESSION
  Status:  ./scripts/server_status.sh
  Stop:    tmux kill-session -t $TMUX_SESSION"
        fi
        log "starting in tmux session '$TMUX_SESSION' (survives SSH disconnect)"
        tmux new-session -d -s "$TMUX_SESSION" \
            "cd $(printf '%q' "$REPO_ROOT") && SCHAIN_NO_TMUX=1 $(printf '%q' "$0") $(printf '%q ' "$@")"
        echo
        echo "  attach:  tmux attach -t $TMUX_SESSION"
        echo "  status:  ./scripts/server_status.sh"
        exit 0
    fi
    warn "tmux is not installed; running in the foreground (an SSH disconnect will stop the run)"
fi

# --- 2. environment report ----------------------------------------------------------

"$REPO_ROOT/scripts/server_preflight.sh" "$CONFIG_FILE"

# --- 3. virtual environment ---------------------------------------------------------

STAMP="$REPO_ROOT/.venv/.schain-install-stamp"
ensure_venv() {
    if [ ! -x "$VENV_PY" ]; then
        log "creating .venv"
        [ "$DRY_RUN" -eq 1 ] && { echo "  would run: python3 -m venv .venv && .venv/bin/python -m pip install -e ."; return; }
        python3 -m venv "$REPO_ROOT/.venv" || die "could not create .venv"
        "$VENV_PY" -m pip install --quiet --upgrade pip
    fi
    # Reinstall only when the dependency declaration is newer than the last install.
    if [ ! -f "$STAMP" ] || [ "$REPO_ROOT/pyproject.toml" -nt "$STAMP" ]; then
        log "installing/updating project dependencies (pyproject.toml changed)"
        [ "$DRY_RUN" -eq 1 ] && { echo "  would run: .venv/bin/python -m pip install -e ."; return; }
        "$VENV_PY" -m pip install --quiet -e "$REPO_ROOT" || die "pip install failed"
        touch "$STAMP"
    else
        log "reusing .venv (dependencies unchanged)"
    fi
}
ensure_venv

# Configuration comes from the project's own loader, so it cannot drift from the app.
if [ "$DRY_RUN" -eq 1 ] && [ ! -x "$VENV_PY" ]; then
    CFG_LLM_BASE_URL="http://localhost:8000/v1"; CFG_NEO4J_URI="bolt://localhost:7687"
    CFG_LLM_MODEL="(unknown until .venv exists)"; CFG_NEO4J_DATABASE="neo4j"
else
    # Capture first: eval of an empty string would succeed and leave variables unset.
    if ! CONFIG_OUTPUT="$(config_values "$CONFIG_FILE" 2>&1)"; then
        die "could not read $CONFIG_FILE:
$CONFIG_OUTPUT"
    fi
    eval "$CONFIG_OUTPUT"
fi

# --- 4. compute allocation (only if SLURM is present and we must start vLLM) --------

need_gpu=0
if ! llm_healthy "$CFG_LLM_BASE_URL"; then need_gpu=1; fi

if [ "$need_gpu" -eq 1 ] && slurm_available && ! in_slurm_allocation && [ "$(gpu_count)" -eq 0 ]; then
    missing=""
    [ -z "${SCHAIN_SLURM_PARTITION:-}" ] && missing="$missing\n  SCHAIN_SLURM_PARTITION"
    if [ -n "$missing" ]; then
        printf 'Missing server configuration:%b\n\n' "$missing" >&2
        cat >&2 <<'HINT'
This machine has SLURM, no GPU is visible in this shell, and the LLM endpoint is not
running, so a GPU allocation is needed. Set the value(s) above once in .env.

Discover them on this cluster with:
  sinfo -o '%P %G %l'        # partitions, their GRES and time limits
  sacctmgr show assoc where user=$USER format=account,partition,qos

See docs/SERVER_REQUIRED_INFO.md.
HINT
        exit 1
    fi
    SLURM_ARGS=(--partition="$SCHAIN_SLURM_PARTITION"
                --gres="${SCHAIN_SLURM_GRES:-gpu:1}"
                --time="${SCHAIN_SLURM_TIME:-08:00:00}"
                --cpus-per-task="${SCHAIN_SLURM_CPUS:-8}")
    [ -n "${SCHAIN_SLURM_ACCOUNT:-}" ] && SLURM_ARGS+=(--account="$SCHAIN_SLURM_ACCOUNT")
    [ -n "${SCHAIN_SLURM_MEM:-}" ]     && SLURM_ARGS+=(--mem="$SCHAIN_SLURM_MEM")
    [ -n "${SCHAIN_SLURM_JOB_NAME:-}" ] && SLURM_ARGS+=(--job-name="$SCHAIN_SLURM_JOB_NAME")

    log "requesting a GPU allocation: srun ${SLURM_ARGS[*]}"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  would exec: srun ${SLURM_ARGS[*]} $0 $* (allocation is released when this script exits)"
    else
        # Re-exec inside the allocation. When the workload finishes the script exits,
        # srun returns and the allocation is released - no idle reservation is left.
        trap - EXIT INT TERM
        exec srun "${SLURM_ARGS[@]}" env SCHAIN_NO_TMUX=1 SCHAIN_IN_ALLOCATION=1 "$0" "$@"
    fi
fi

# --- 5. Neo4j: reuse if reachable, otherwise start a container ---------------------

ensure_neo4j() {
    if neo4j_reachable "$CFG_NEO4J_URI"; then
        log "Neo4j already reachable at $CFG_NEO4J_URI - reusing it (it will not be stopped)"
        return 0
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  would start Neo4j container '$NEO4J_CONTAINER' from $NEO4J_IMAGE, data in $NEO4J_DATA_DIR"
        return 0
    fi
    have docker || die "Neo4j is not reachable at $CFG_NEO4J_URI and docker is not available.
Start Neo4j yourself, or set NEO4J_URI in .env to point at an existing instance."
    [ -n "${NEO4J_PASSWORD:-}" ] || die "NEO4J_PASSWORD is not set in .env, so a Neo4j container cannot be started."

    mkdir -p "$NEO4J_DATA_DIR"
    if docker ps -a --format '{{.Names}}' | grep -qx "$NEO4J_CONTAINER"; then
        log "starting existing Neo4j container '$NEO4J_CONTAINER'"
        docker start "$NEO4J_CONTAINER" >/dev/null || die "could not start container $NEO4J_CONTAINER"
    else
        log "creating Neo4j container '$NEO4J_CONTAINER' (data persisted in $NEO4J_DATA_DIR)"
        # APOC is deliberately not installed: V1 writes plain Cypher MERGE only.
        docker run -d --name "$NEO4J_CONTAINER" \
            -p "$(neo4j_port "$CFG_NEO4J_URI")":7687 -p 7474:7474 \
            -e NEO4J_AUTH="${NEO4J_USERNAME:-neo4j}/${NEO4J_PASSWORD}" \
            -v "$NEO4J_DATA_DIR":/data \
            "$NEO4J_IMAGE" >/dev/null || die "could not create container $NEO4J_CONTAINER"
    fi
    OWNED_NEO4J="$NEO4J_CONTAINER"

    log "waiting for Neo4j (up to ${NEO4J_STARTUP_TIMEOUT}s)"
    for _ in $(seq 1 "$NEO4J_STARTUP_TIMEOUT"); do
        neo4j_reachable "$CFG_NEO4J_URI" && { log "Neo4j is up"; return 0; }
        sleep 1
    done
    die "Neo4j did not become reachable within ${NEO4J_STARTUP_TIMEOUT}s. Logs: docker logs $NEO4J_CONTAINER"
}
ensure_neo4j

# --- 6. vLLM: reuse if healthy, otherwise start it ---------------------------------

ensure_llm() {
    if llm_healthy "$CFG_LLM_BASE_URL"; then
        log "LLM endpoint already healthy at $CFG_LLM_BASE_URL - reusing it (it will not be stopped)"
        return 0
    fi

    local host port vllm_bin
    host="$(neo4j_host "$(llm_origin "$CFG_LLM_BASE_URL")")"
    port="$(neo4j_port "$(llm_origin "$CFG_LLM_BASE_URL")")"
    case "$host" in
        localhost|127.0.0.1|0.0.0.0) : ;;
        *) die "llm.base_url points at $host, which is not this machine, and it is not responding.
Start the endpoint there, or point llm.base_url / LLM_API_BASE_URL at a reachable one." ;;
    esac

    if [ -x "$REPO_ROOT/.venv/bin/vllm" ]; then vllm_bin="$REPO_ROOT/.venv/bin/vllm"
    elif have vllm; then vllm_bin="vllm"
    else
        [ "$DRY_RUN" -eq 1 ] && vllm_bin="vllm" || die "vLLM is not installed and the endpoint at $CFG_LLM_BASE_URL is down.
Install it in the project environment on the GPU server:
  .venv/bin/python -m pip install vllm
It is intentionally not installed automatically: it is large and GPU-specific."
    fi

    local tp="${SCHAIN_VLLM_TENSOR_PARALLEL_SIZE:-$(gpu_count)}"
    [ "${tp:-0}" -ge 1 ] 2>/dev/null || tp=1
    local model="${SCHAIN_VLLM_MODEL:-$CFG_LLM_MODEL}"
    local vllm_log="$LOG_DIR/vllm_${RUN_STAMP}.log"

    # shellcheck disable=SC2206
    local cmd=("$vllm_bin" serve "$model" --served-model-name "$model"
               --port "$port" --max-model-len "$VLLM_MAX_MODEL_LEN"
               --gpu-memory-utilization "$VLLM_GPU_UTIL" --tensor-parallel-size "$tp"
               $VLLM_EXTRA_ARGS)

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  would start vLLM: ${cmd[*]}"
        echo "  would wait up to ${VLLM_STARTUP_TIMEOUT}s for $(llm_origin "$CFG_LLM_BASE_URL")/health"
        return 0
    fi

    [ "$(gpu_count)" -ge 1 ] || warn "no GPU is visible; vLLM will probably fail to start"
    log "starting vLLM: ${cmd[*]}"
    log "vLLM log: $vllm_log"
    nohup "${cmd[@]}" > "$vllm_log" 2>&1 &
    OWNED_VLLM_PID=$!

    log "waiting for the model to load (up to ${VLLM_STARTUP_TIMEOUT}s; a 27B model takes minutes)"
    for _ in $(seq 1 "$VLLM_STARTUP_TIMEOUT"); do
        if ! kill -0 "$OWNED_VLLM_PID" 2>/dev/null; then
            tail -20 "$vllm_log" >&2
            die "vLLM exited during startup. Full log: $vllm_log"
        fi
        llm_healthy "$CFG_LLM_BASE_URL" && { log "LLM endpoint is healthy"; return 0; }
        sleep 1
    done
    tail -20 "$vllm_log" >&2
    die "vLLM did not become healthy within ${VLLM_STARTUP_TIMEOUT}s. Full log: $vllm_log"
}

# --- 7. build the SChain command ---------------------------------------------------

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
RUN_STAMP="$(date +%Y-%m-%d_%H%M%S)"
RUN_LOG="$LOG_DIR/${MODE}_${RUN_STAMP}.log"

SCHAIN_CMD=("$VENV_PY" -m schain "$MODE" --config "$CONFIG_FILE")
[ "$RESTART" -eq 1 ] && SCHAIN_CMD+=(--restart)

if [ "$MODE" = "ingest" ]; then
    [ "$SMOKE" -eq 1 ] && [ -z "$INPUT" ] && INPUT="$SMOKE_INPUT"
    [ -n "$INPUT" ] && SCHAIN_CMD+=(--input "$INPUT")
elif [ "$SMOKE" -eq 1 ]; then
    # Smallest useful real run: one tier, a few documents, normal pipeline code.
    SCHAIN_CMD+=(--max-tier "${SCHAIN_SMOKE_MAX_TIER:-0}"
                 --max-documents "${SCHAIN_SMOKE_MAX_DOCUMENTS:-3}")
fi

ensure_llm

echo
log "mode      : $MODE$([ "$SMOKE" -eq 1 ] && echo ' (smoke test)')"
log "config    : $CONFIG_FILE"
log "command   : ${SCHAIN_CMD[*]}"
log "log file  : $RUN_LOG"
[ "$RESTART" -eq 1 ] && warn "--restart: checkpoint state and generated data will be cleared first"

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    log "dry run: nothing was started or changed"
    exit 0
fi

# --- 8. run the pipeline -----------------------------------------------------------

echo
"${SCHAIN_CMD[@]}" 2>&1 | tee -a "$RUN_LOG"
STATUS="${PIPESTATUS[0]}"
echo

if [ "$STATUS" -eq 0 ]; then
    log "workload finished successfully"
    [ "$SMOKE" -eq 1 ] && log "smoke test passed - run ./scripts/server_run.sh for the full run"
else
    log "workload exited with status $STATUS - rerun this script to resume from the checkpoint"
fi
log "log: $RUN_LOG"
# cleanup() now stops only the services this run started.
exit "$STATUS"
