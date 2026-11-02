#!/usr/bin/env bash
# Report what this machine offers. Read-only: nothing is started, installed or changed.
#
#   ./scripts/server_preflight.sh [config-file]
#
# server_run.sh calls this first. Secrets are reported as "set"/"not set", never printed.

set -uo pipefail
# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

CONFIG_FILE="${1:-config/expansion.yaml}"
load_env

section() { printf '\n%s\n' "== $* =="; }
item()    { printf '  %-26s %s\n' "$1" "$2"; }

section "host"
item "os"            "$(uname -s) $(uname -r)"
item "hostname"      "$(hostname)"
item "repository"    "$REPO_ROOT"
item "disk free"     "$(df -Ph "$REPO_ROOT" | awk 'NR==2 {print $4" free of "$2}')"

section "python"
item "python3"       "$(have python3 && python3 --version 2>&1 || echo 'MISSING')"
if [ -x "$VENV_PY" ]; then
    item ".venv"     "present ($("$VENV_PY" --version 2>&1))"
    item "schain"    "$("$VENV_PY" -c 'import schain; print(schain.__version__)' 2>/dev/null || echo 'not importable')"
else
    item ".venv"     "missing (server_run.sh will create it)"
fi

section "tools"
item "tmux"          "$(have tmux && tmux -V || echo 'not installed (run stays in the foreground)')"
item "docker"        "$(have docker && docker --version 2>/dev/null | cut -d, -f1 || echo 'not installed')"
item "curl"          "$(have curl && echo present || echo 'MISSING (needed for health checks)')"
item "vllm"          "$(have vllm && echo 'command present' || echo 'command not found')"
if [ -x "$VENV_PY" ]; then
    item "vllm (venv)" "$("$VENV_PY" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'not installed in .venv')"
fi

section "gpu"
if have nvidia-smi; then
    item "nvidia-smi" "present"
    item "gpu count"  "$(gpu_count)"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>/dev/null \
        | while IFS= read -r line; do item "  gpu" "$line"; done
    item "cuda (driver)" "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
else
    item "nvidia-smi" "not found (no local GPU visible)"
fi

section "scheduler"
if slurm_available; then
    item "slurm"      "present"
    item "in allocation" "$(in_slurm_allocation && echo "yes (job ${SLURM_JOB_ID})" || echo 'no')"
    if have sinfo; then
        # Partitions that advertise a GRES, i.e. the ones that can serve a GPU job.
        gpu_partitions="$(sinfo -h -o '%P %G' 2>/dev/null | awk '$2 != "(null)" && $2 != "" {print $1}' | sort -u | tr '\n' ' ')"
        item "gpu partitions" "${gpu_partitions:-could not be listed}"
    fi
    item "SCHAIN_SLURM_PARTITION" "${SCHAIN_SLURM_PARTITION:-not set}"
    item "SCHAIN_SLURM_ACCOUNT"   "${SCHAIN_SLURM_ACCOUNT:-not set}"
    item "SCHAIN_SLURM_GRES"      "${SCHAIN_SLURM_GRES:-gpu:1 (default)}"
else
    item "slurm"      "not present (running directly on this host)"
fi

section "configuration ($CONFIG_FILE)"
if values="$(config_values "$CONFIG_FILE" 2>/dev/null)"; then
    eval "$values"
    item "llm provider"  "$CFG_LLM_PROVIDER"
    item "llm model"     "$CFG_LLM_MODEL"
    item "llm base_url"  "$CFG_LLM_BASE_URL"
    item "neo4j uri"     "$CFG_NEO4J_URI"
    item "neo4j database" "$CFG_NEO4J_DATABASE"
    item "state db"      "$CFG_STATE_DB $([ -f "$CFG_STATE_DB" ] && echo '(exists: a run can resume)' || echo '(absent: first run)')"
else
    item "config"        "not readable (Python dependencies missing or config error)"
fi

section "secrets (.env)"
item ".env file"        "$([ -f "$REPO_ROOT/.env" ] && echo present || echo 'missing (copy .env.example)')"
item "NEO4J_URI"        "${NEO4J_URI:-not set (defaults to bolt://localhost:7687)}"
item "NEO4J_USERNAME"   "${NEO4J_USERNAME:-not set (defaults to neo4j)}"
item "NEO4J_PASSWORD"   "$(secret_state NEO4J_PASSWORD)"
item "LLM_API_KEY"      "$(secret_state LLM_API_KEY)"
item "LLM_API_BASE_URL" "${LLM_API_BASE_URL:-not set (config base_url is used)}"
item "HF_TOKEN"         "$(secret_state HF_TOKEN)"
item "HF_HOME"          "${HF_HOME:-not set (default ~/.cache/huggingface)}"

section "services"
if [ -n "${CFG_NEO4J_URI:-}" ]; then
    item "neo4j"        "$(neo4j_reachable "$CFG_NEO4J_URI" && echo "reachable at $CFG_NEO4J_URI (will be reused)" || echo 'not reachable (server_run.sh can start one)')"
    item "llm endpoint" "$(llm_healthy "$CFG_LLM_BASE_URL" && echo "healthy at $CFG_LLM_BASE_URL (will be reused)" || echo 'not healthy (server_run.sh can start vLLM)')"
fi
printf '\n'
