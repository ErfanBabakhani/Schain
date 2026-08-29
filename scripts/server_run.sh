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
NEO4J_STARTUP_TIMEOUT="${SCHAIN_NEO4J_STARTUP_TIMEOUT:-180}"
# Local user-space Neo4j, used when nothing is reachable and Docker is unavailable.
NEO4J_VERSION="${SCHAIN_NEO4J_VERSION:-5.26.0}"
RUNTIME_DIR="${SCHAIN_RUNTIME_DIR:-$REPO_ROOT/.runtime}"
NEO4J_HOME="$RUNTIME_DIR/neo4j/neo4j-community-$NEO4J_VERSION"
NEO4J_DIST_URL="${SCHAIN_NEO4J_DIST_URL:-https://dist.neo4j.org/neo4j-community-$NEO4J_VERSION-unix.tar.gz}"
VLLM_AUTO_INSTALL="${SCHAIN_VLLM_AUTO_INSTALL:-1}"
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
    if [ "$OWNED_NEO4J" = "local" ]; then
        log "stopping the local Neo4j started by this run (data is kept)"
        "$NEO4J_HOME/bin/neo4j" stop >/dev/null 2>&1
    elif [ -n "$OWNED_NEO4J" ]; then
        log "stopping Neo4j container '$OWNED_NEO4J' started by this run (data is kept)"
        docker stop "$OWNED_NEO4J" >/dev/null 2>&1
    fi
    # Checkpoint state and graph data are never removed here.
    exit $status
}
trap cleanup EXIT INT TERM

# --- 1. tmux: survive an SSH disconnect --------------------------------------------

# On a SLURM cluster the batch job is the detached process, so tmux adds nothing.
if [ "$FOREGROUND" -eq 0 ] && [ -z "${TMUX:-}" ] && [ -z "${SCHAIN_NO_TMUX:-}" ] \
   && ! slurm_available; then
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

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
RUN_STAMP="${SCHAIN_RUN_STAMP:-$(date +%Y-%m-%d_%H%M%S)}"
RUN_LOG="$LOG_DIR/${MODE}_${RUN_STAMP}.log"

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
    JOB_NAME="${SCHAIN_SLURM_JOB_NAME:-schain-$MODE}"
    SLURM_ARGS=(--partition="$SCHAIN_SLURM_PARTITION"
                --gres="${SCHAIN_SLURM_GRES:-gpu:1}"
                --cpus-per-task="${SCHAIN_SLURM_CPUS:-8}"
                --job-name="$JOB_NAME")
    [ -n "${SCHAIN_SLURM_ACCOUNT:-}" ] && SLURM_ARGS+=(--account="$SCHAIN_SLURM_ACCOUNT")
    [ -n "${SCHAIN_SLURM_MEM:-}" ]     && SLURM_ARGS+=(--mem="$SCHAIN_SLURM_MEM")

    if [ "$SMOKE" -eq 1 ]; then
        # A smoke test is short, and this cluster allows interactive allocations for
        # debugging (capped at two hours), so srun keeps the output in front of you.
        SLURM_ARGS+=(--time="${SCHAIN_SLURM_SMOKE_TIME:-02:00:00}")
        log "requesting an interactive GPU allocation: srun ${SLURM_ARGS[*]}"
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "  would exec: srun ${SLURM_ARGS[*]} $0 $* (allocation is released when this script exits)"
        else
            trap - EXIT INT TERM
            exec srun "${SLURM_ARGS[@]}" \
                env SCHAIN_NO_TMUX=1 SCHAIN_IN_ALLOCATION=1 SCHAIN_RUN_STAMP="$RUN_STAMP" \
                "$0" "$@"
        fi
    else
        # Production must not use an interactive allocation: those are capped at two
        # hours here whatever --time asks for. Submit a batch job instead; SLURM keeps
        # it running after logout and releases the GPU when the job exits.
        SLURM_ARGS+=(--time="${SCHAIN_SLURM_TIME:-24:00:00}")
        BATCH_DIR="$RUNTIME_DIR/slurm"
        mkdir -p "$BATCH_DIR"
        BATCH_FILE="$BATCH_DIR/${MODE}_${RUN_STAMP}.sbatch"
        {
            echo "#!/bin/bash"
            for arg in "${SLURM_ARGS[@]}"; do echo "#SBATCH $arg"; done
            echo "#SBATCH --output=$RUN_LOG"
            echo "set -uo pipefail"
            echo "cd $(printf '%q' "$REPO_ROOT")"
            # SCHAIN_LOG_TO_STDOUT: SLURM already writes stdout to RUN_LOG, so the run
            # must not tee into the same file as well.
            echo "export SCHAIN_NO_TMUX=1 SCHAIN_IN_ALLOCATION=1 SCHAIN_LOG_TO_STDOUT=1"
            echo "export SCHAIN_RUN_STAMP=$(printf '%q' "$RUN_STAMP")"
            echo "exec $(printf '%q' "$0") $(printf '%q ' "$@") --foreground"
        } > "$BATCH_FILE"
        chmod +x "$BATCH_FILE"

        log "submitting a batch job (interactive allocations are capped at 2h here)"
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "  would run: sbatch $BATCH_FILE"
            echo "  batch directives: ${SLURM_ARGS[*]} --output=$RUN_LOG"
        else
            submission="$(sbatch "$BATCH_FILE")" || die "sbatch failed: $submission"
            job_id="$(printf '%s' "$submission" | grep -oE '[0-9]+$')"
            mkdir -p "$(dirname "$(slurm_job_file)")"
            printf '%s\n' "$job_id" > "$(slurm_job_file)"
            echo
            echo "  Submitted SChain ${MODE^^}"
            echo "  SLURM job: $job_id"
            echo "  Log:       $RUN_LOG"
            echo "  Status:    ./scripts/server_status.sh"
            echo "  Stop:      scancel $job_id"
            trap - EXIT INT TERM     # nothing was started here to clean up
            exit 0
        fi
    fi
fi

# --- 5. Neo4j: reuse if reachable, otherwise start a container ---------------------

# Generate a password on first local install, store it in .env, never print it.
ensure_neo4j_password() {
    [ -n "${NEO4J_PASSWORD:-}" ] && return 0
    local generated
    generated="$("$VENV_PY" -c 'import secrets; print(secrets.token_urlsafe(24))')"
    [ -n "$generated" ] || die "could not generate a Neo4j password"

    local env_file="$REPO_ROOT/.env"
    [ -f "$env_file" ] || cp "$REPO_ROOT/.env.example" "$env_file"
    if grep -q '^NEO4J_PASSWORD=' "$env_file"; then
        sed -i "s|^NEO4J_PASSWORD=.*|NEO4J_PASSWORD=$generated|" "$env_file"
    else
        printf '\nNEO4J_PASSWORD=%s\n' "$generated" >> "$env_file"
    fi
    chmod 600 "$env_file"
    export NEO4J_PASSWORD="$generated"
    log "NEO4J_PASSWORD: generated and stored in .env"
}

# Install Neo4j Community from the official tarball into a gitignored runtime directory.
# No sudo, no Docker, no systemd. Data and logs live outside the distribution so an
# upgrade never touches the graph.
install_local_neo4j() {
    local data_dir="$RUNTIME_DIR/neo4j/data" logs_dir="$RUNTIME_DIR/neo4j/logs"
    mkdir -p "$RUNTIME_DIR/neo4j" "$data_dir" "$logs_dir"

    if [ ! -x "$NEO4J_HOME/bin/neo4j" ]; then
        log "downloading Neo4j Community $NEO4J_VERSION (one time)"
        local tarball="$RUNTIME_DIR/neo4j/neo4j-$NEO4J_VERSION.tar.gz"
        curl -fsSL "$NEO4J_DIST_URL" -o "$tarball" \
            || die "could not download Neo4j from $NEO4J_DIST_URL"
        tar -xzf "$tarball" -C "$RUNTIME_DIR/neo4j" || die "could not unpack Neo4j"
        rm -f "$tarball"
        [ -x "$NEO4J_HOME/bin/neo4j" ] || die "unexpected Neo4j layout under $NEO4J_HOME"
    fi

    local conf="$NEO4J_HOME/conf/neo4j.conf"
    if ! grep -q "# added by scripts/server_run.sh" "$conf" 2>/dev/null; then
        local port; port="$(neo4j_port "$CFG_NEO4J_URI")"
        {
            echo ""
            echo "# added by scripts/server_run.sh"
            echo "server.directories.data=$data_dir"
            echo "server.directories.logs=$logs_dir"
            # SChain runs on the same node as this instance, so localhost is enough.
            echo "server.default_listen_address=localhost"
            echo "server.bolt.listen_address=localhost:$port"
        } >> "$conf"
        log "configured local Neo4j (data: $data_dir)"
    fi

    # Only before the very first start; afterwards the stored credential is authoritative.
    if [ ! -e "$data_dir/dbms/auth" ]; then
        "$NEO4J_HOME/bin/neo4j-admin" dbms set-initial-password "$NEO4J_PASSWORD" >/dev/null 2>&1 \
            || warn "could not set the initial Neo4j password; it may already be set"
    fi
}

start_local_neo4j() {
    local major; major="$(java_major)"
    if [ -z "$major" ] || [ "$major" -lt 21 ] 2>/dev/null; then
        die "Java 21/25 is required for local Neo4j (found: ${major:-none}).
Load a JDK module or set JAVA_HOME, or point NEO4J_URI at an existing instance."
    fi
    ensure_neo4j_password
    install_local_neo4j

    log "starting local Neo4j $NEO4J_VERSION"
    "$NEO4J_HOME/bin/neo4j" start >/dev/null || die "could not start local Neo4j"
    OWNED_NEO4J="local"
}

ensure_neo4j() {
    if neo4j_reachable "$CFG_NEO4J_URI"; then
        log "Neo4j already reachable at $CFG_NEO4J_URI - reusing it (it will not be stopped)"
        return 0
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        if have docker; then
            echo "  would start Neo4j container '$NEO4J_CONTAINER' from $NEO4J_IMAGE, data in $NEO4J_DATA_DIR"
        else
            echo "  would install and start local Neo4j $NEO4J_VERSION under $RUNTIME_DIR/neo4j"
        fi
        return 0
    fi

    if have docker; then
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
    else
        # No Docker and no root on this cluster: run Neo4j from the official tarball
        # in user space, on this compute node, beside the pipeline.
        start_local_neo4j
    fi

    log "waiting for Neo4j (up to ${NEO4J_STARTUP_TIMEOUT}s)"
    for _ in $(seq 1 "$NEO4J_STARTUP_TIMEOUT"); do
        neo4j_reachable "$CFG_NEO4J_URI" && { log "Neo4j is up"; return 0; }
        sleep 1
    done
    die "Neo4j did not become reachable within ${NEO4J_STARTUP_TIMEOUT}s.
Logs: ${OWNED_NEO4J:-neo4j} ($RUNTIME_DIR/neo4j/logs or docker logs $NEO4J_CONTAINER)"
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
    elif [ "$DRY_RUN" -eq 1 ]; then vllm_bin="vllm"
    elif [ "$VLLM_AUTO_INSTALL" = "1" ] && [ "$(gpu_count)" -ge 1 ]; then
        # Installed once, into the environment the runner already looks in. Large
        # download, so it happens only when a GPU is present and the endpoint is down.
        log "vLLM is not installed; installing it into .venv (one time, large download)"
        "$VENV_PY" -m pip install --quiet vllm || die "vLLM installation failed.
Install it manually:  .venv/bin/python -m pip install vllm"
        [ -x "$REPO_ROOT/.venv/bin/vllm" ] || die "vLLM installed but .venv/bin/vllm is missing"
        vllm_bin="$REPO_ROOT/.venv/bin/vllm"
    else
        die "vLLM is not installed and the endpoint at $CFG_LLM_BASE_URL is down.
Install it in the project environment on the GPU server:
  .venv/bin/python -m pip install vllm
Automatic installation is skipped here (no GPU visible, or SCHAIN_VLLM_AUTO_INSTALL=0)."
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
    if [ -n "${HF_HOME:-}" ]; then
        export HF_HOME
        log "Hugging Face cache: $HF_HOME"
    fi
    log "HF_TOKEN: $(secret_state HF_TOKEN)"
    log "starting vLLM: ${cmd[*]}"
    log "vLLM log: $vllm_log"
    log "the first run downloads the model, which can take a long time - this is not a hang;"
    log "watch progress with: tail -f $vllm_log"
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
if [ -n "${SCHAIN_LOG_TO_STDOUT:-}" ]; then
    # Running under sbatch: SLURM writes stdout to RUN_LOG already.
    "${SCHAIN_CMD[@]}" 2>&1
    STATUS=$?
else
    "${SCHAIN_CMD[@]}" 2>&1 | tee -a "$RUN_LOG"
    STATUS="${PIPESTATUS[0]}"
fi
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
