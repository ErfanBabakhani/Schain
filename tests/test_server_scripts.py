"""The server scripts, exercised locally without a GPU, a real model or a real Neo4j.

The scripts only prepare services and launch the existing Python CLI, so what is worth
testing is exactly that: argument handling, command construction, health detection, and
the ownership rule that decides what cleanup is allowed to stop.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

from schain.config import REPO_ROOT

SCRIPTS = REPO_ROOT / "scripts"
RUN = SCRIPTS / "server_run.sh"
PREFLIGHT = SCRIPTS / "server_preflight.sh"
STATUS = SCRIPTS / "server_status.sh"
LIB = SCRIPTS / "lib.sh"


def run(args, env=None, cwd=None, timeout=180):
    environment = {**os.environ, "SCHAIN_NO_TMUX": "1", **(env or {})}
    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout,
        env=environment, cwd=str(cwd or REPO_ROOT),
    )


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# --- syntax and help ----------------------------------------------------------------


@pytest.mark.parametrize("script", [RUN, PREFLIGHT, STATUS, LIB])
def test_scripts_are_syntactically_valid(script):
    assert run(["bash", "-n", str(script)]).returncode == 0


@pytest.mark.parametrize("script", [RUN, PREFLIGHT, STATUS])
def test_scripts_are_executable(script):
    assert os.access(script, os.X_OK), f"{script.name} is not executable"


def test_help_documents_both_modes_and_the_default():
    result = run([str(RUN), "--help"])
    assert result.returncode == 0
    assert "expand" in result.stdout and "ingest" in result.stdout
    assert "(default)" in result.stdout          # the default mode is stated
    assert "--smoke-test" in result.stdout and "--restart" in result.stdout


def test_unknown_option_fails_clearly():
    result = run([str(RUN), "--nonsense"])
    assert result.returncode != 0
    assert "unknown option" in result.stderr


# --- command construction -----------------------------------------------------------


def plan(*args):
    """The dry-run plan, stdout and stderr together (warnings go to stderr)."""
    result = run([str(RUN), "--dry-run", *args])
    assert result.returncode == 0, result.stderr
    return result.stdout + result.stderr


def test_default_mode_is_expand():
    output = plan()
    assert "-m schain expand --config config/expansion.yaml" in output
    assert "mode      : expand" in output


def test_mode_maps_to_the_right_config_file():
    assert "--config config/expansion.yaml" in plan("expand")
    assert "--config config/ingest.yaml" in plan("ingest", "data/corpus")


def test_ingest_passes_its_input():
    assert "-m schain ingest --config config/ingest.yaml --input data/corpus" in plan(
        "ingest", "data/corpus"
    )


def test_smoke_test_limits_the_expansion():
    output = plan("--smoke-test")
    assert "--max-tier 0 --max-documents 3" in output
    assert "(smoke test)" in output


def test_smoke_test_ingest_uses_a_small_default_document():
    assert "--input docs/SupplyChain_KG_V1_Architecture_Brief_Final.pdf" in plan(
        "ingest", "--smoke-test"
    )


def test_restart_is_passed_through_and_warned_about():
    output = plan("--restart")
    assert "-m schain expand --config config/expansion.yaml --restart" in output
    assert "checkpoint state and generated data will be cleared" in output


def test_resume_is_the_default_so_no_flag_is_added():
    """Resume is the CLI's own behaviour; the script must not pass anything for it."""
    assert "--restart" not in plan()


def test_config_override_is_honoured():
    assert "--config config/ingest.yaml" in plan("--config", "config/ingest.yaml")


def test_repo_root_is_found_from_any_directory(tmp_path):
    result = run([str(RUN), "--dry-run"], cwd=tmp_path)
    assert result.returncode == 0
    assert str(REPO_ROOT) in result.stdout


def test_dry_run_changes_nothing(tmp_path):
    before = sorted(p.name for p in (REPO_ROOT / "data" / "state").glob("*"))
    plan()
    assert sorted(p.name for p in (REPO_ROOT / "data" / "state").glob("*")) == before


# --- environment detection ----------------------------------------------------------


def test_preflight_reports_without_leaking_secrets():
    result = run([str(PREFLIGHT)], env={"NEO4J_PASSWORD": "hunter2", "HF_TOKEN": "hf_secret"})
    assert result.returncode == 0
    assert "hunter2" not in result.stdout and "hf_secret" not in result.stdout
    assert "NEO4J_PASSWORD             set" in result.stdout
    for section in ("host", "python", "tools", "gpu", "scheduler", "services"):
        assert f"== {section} ==" in result.stdout


def test_preflight_reports_the_configured_model_and_endpoint():
    result = run([str(PREFLIGHT)])
    assert "Qwen/Qwen3.8-27B" in result.stdout
    assert "bolt://localhost:7687" in result.stdout


def test_status_runs_and_reports_the_sections():
    result = run([str(STATUS)])
    assert result.returncode == 0
    for section in ("== run ==", "== services ==", "== checkpoint state ==", "== logs =="):
        assert section in result.stdout


def test_process_detection_does_not_match_the_checking_shell():
    """An unanchored pgrep -f would match any shell mentioning "-m schain"."""
    script = f'source {LIB}; schain_pids | wc -l'
    result = run(["bash", "-c", script])
    assert result.stdout.strip() == "0"


def test_health_checks_detect_a_live_endpoint():
    port = free_port()
    server = subprocess.Popen(
        ["python3", "-c", textwrap.dedent(f"""
            from http.server import BaseHTTPRequestHandler, HTTPServer
            class H(BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200); self.end_headers(); self.wfile.write(b'{{}}')
                def log_message(self, *a): pass
            HTTPServer(('127.0.0.1', {port}), H).serve_forever()
        """)],
    )
    try:
        time.sleep(1.5)
        script = (
            f'source {LIB}; '
            f'llm_healthy http://127.0.0.1:{port}/v1 && echo LLM_UP; '
            f'neo4j_reachable bolt://127.0.0.1:{port} && echo NEO4J_UP; '
            f'llm_healthy http://127.0.0.1:{free_port()}/v1 || echo LLM_DOWN'
        )
        result = run(["bash", "-c", script])
        assert "LLM_UP" in result.stdout
        assert "NEO4J_UP" in result.stdout
        assert "LLM_DOWN" in result.stdout
    finally:
        server.terminate()
        server.wait(timeout=10)


# --- SLURM: fail early, naming exactly what is missing -------------------------------


def test_slurm_without_configuration_names_the_missing_variable(tmp_path):
    """SLURM present, no GPU visible, no LLM endpoint: the allocation values are needed."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name in ("sbatch", "srun", "sinfo"):
        stub = fake_bin / name
        stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        stub.chmod(0o755)
    # nvidia-smi exists but reports no GPU, as on a scheduler login node.
    nvidia = fake_bin / "nvidia-smi"
    nvidia.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvidia.chmod(0o755)
    minimal_path = f"{fake_bin}:{os.environ['PATH']}"

    result = run([str(RUN), "--dry-run"], env={"PATH": minimal_path, "SCHAIN_SLURM_PARTITION": ""})
    assert result.returncode == 1
    assert "Missing server configuration:" in result.stderr
    assert "SCHAIN_SLURM_PARTITION" in result.stderr
    assert "sinfo -o" in result.stderr        # tells you how to discover it


def test_slurm_with_configuration_builds_the_allocation_command(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name in ("sbatch", "srun", "sinfo", "nvidia-smi"):
        stub = fake_bin / name
        stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        stub.chmod(0o755)

    result = run([str(RUN), "--dry-run"], env={
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SCHAIN_SLURM_PARTITION": "gpu-short",
        "SCHAIN_SLURM_ACCOUNT": "my-account",
        "SCHAIN_SLURM_GRES": "gpu:a100:2",
    })
    assert result.returncode == 0, result.stderr
    assert "--partition=gpu-short" in result.stdout
    assert "--account=my-account" in result.stdout
    assert "--gres=gpu:a100:2" in result.stdout
    assert "allocation is released when this script exits" in result.stdout


def test_no_slurm_means_no_allocation_step():
    assert "srun" not in plan()


# --- service ownership: what may be stopped, and what may not ------------------------


def write_temp_config(tmp_path: Path, llm_port: int) -> Path:
    """A real config that includes config/base.yaml but points at a local dummy endpoint."""
    config = tmp_path / "server_test.yaml"
    config.write_text(textwrap.dedent(f"""
        include:
          - config/base.yaml
        mode: expand
        llm:
          base_url: http://127.0.0.1:{llm_port}/v1
        expansion:
          max_tier: 0
          max_documents: 1
        storage:
          raw_documents: {tmp_path}/raw.jsonl
          normalized_documents: {tmp_path}/docs.parquet
          extracted_facts: {tmp_path}/facts.parquet
          extractions: {tmp_path}/extractions
          search_results: {tmp_path}/search.parquet
          entity_resolution: {tmp_path}/res.parquet
          state_db: {tmp_path}/state.db
    """).strip(), encoding="utf-8")
    return config


def start_http_stub(port: int, pid_file: Path | None = None) -> subprocess.Popen:
    """An OpenAI-compatible-enough endpoint: 200 on /health and /v1/models."""
    code = textwrap.dedent(f"""
        import os, sys
        from http.server import BaseHTTPRequestHandler, HTTPServer
        if {pid_file!r}:
            open({pid_file!r}, 'w').write(str(os.getpid()))
        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers(); self.wfile.write(b'{{}}')
            def log_message(self, *a): pass
        HTTPServer(('127.0.0.1', {port}), H).serve_forever()
    """)
    process = subprocess.Popen(["python3", "-c", code])
    for _ in range(40):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return process
        time.sleep(0.25)
    raise RuntimeError("stub endpoint did not come up")


def start_closing_listener(port: int) -> subprocess.Popen:
    """Accepts a TCP connection and closes it: reachable, but not really Neo4j."""
    code = textwrap.dedent(f"""
        import socket
        s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', {port})); s.listen(8)
        while True:
            conn, _ = s.accept(); conn.close()
    """)
    process = subprocess.Popen(["python3", "-c", code])
    for _ in range(40):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return process
        time.sleep(0.25)
    raise RuntimeError("stub listener did not come up")


def alive(process: subprocess.Popen) -> bool:
    return process.poll() is None


def test_already_running_services_are_reused_and_never_stopped(tmp_path):
    """The safety-critical rule: what was running before must survive the run."""
    llm_port, neo4j_port = free_port(), free_port()
    llm = start_http_stub(llm_port)
    neo4j = start_closing_listener(neo4j_port)
    try:
        config = write_temp_config(tmp_path, llm_port)
        result = run([str(RUN), "--dry-run", "--config", str(config)],
                     env={"NEO4J_URI": f"bolt://127.0.0.1:{neo4j_port}"})
        assert result.returncode == 0, result.stderr
        output = result.stdout + result.stderr

        assert "Neo4j already reachable" in output and "will not be stopped" in output
        assert "LLM endpoint already healthy" in output
        # nothing was started, so nothing is owned and nothing may be killed
        assert "would start vLLM" not in output
        assert alive(llm) and alive(neo4j)
    finally:
        for process in (llm, neo4j):
            process.terminate()
            process.wait(timeout=10)


def test_a_service_started_by_the_run_is_stopped_when_the_workload_finishes(tmp_path):
    """The other half: an owned vLLM must not be left holding the GPU afterwards."""
    llm_port, neo4j_port = free_port(), free_port()
    neo4j = start_closing_listener(neo4j_port)          # reachable, so the run proceeds
    pid_file = tmp_path / "vllm.pid"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_vllm = fake_bin / "vllm"
    fake_vllm.write_text(textwrap.dedent(f"""
        #!/usr/bin/env python3
        import os, sys
        from http.server import BaseHTTPRequestHandler, HTTPServer
        open({str(pid_file)!r}, 'w').write(str(os.getpid()))
        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers(); self.wfile.write(b'{{}}')
            def log_message(self, *a): pass
        HTTPServer(('127.0.0.1', {llm_port}), H).serve_forever()
    """).strip(), encoding="utf-8")
    fake_vllm.chmod(0o755)

    try:
        config = write_temp_config(tmp_path, llm_port)
        result = run(
            [str(RUN), "--foreground", "--config", str(config)],
            env={
                "NEO4J_URI": f"bolt://127.0.0.1:{neo4j_port}",
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "SCHAIN_VLLM_STARTUP_TIMEOUT": "40",
            },
            timeout=300,
        )
        output = result.stdout + result.stderr

        # the run started vLLM itself...
        assert "starting vLLM" in output
        assert pid_file.exists(), "the fake vLLM never started"
        # ...the pipeline then failed against the dummy Neo4j, which is fine here...
        assert result.returncode != 0
        # ...and cleanup stopped the vLLM it owned
        assert "stopping vLLM" in output
        vllm_pid = int(pid_file.read_text())
        for _ in range(40):
            if not _pid_alive(vllm_pid):
                break
            time.sleep(0.25)
        assert not _pid_alive(vllm_pid), "an owned vLLM was left running after the workload"
        # the pre-existing listener was reused and must still be alive
        assert alive(neo4j)
    finally:
        neo4j.terminate()
        neo4j.wait(timeout=10)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def test_checkpoint_state_is_never_deleted_by_cleanup():
    """Cleanup stops services; only --restart may clear state, via the Python CLI."""
    script = (REPO_ROOT / "scripts" / "server_run.sh").read_text(encoding="utf-8")
    cleanup = script[script.index("cleanup() {"):script.index("trap cleanup")]
    for destructive in ("rm ", "rm -", "unlink", "docker rm", "--restart"):
        assert destructive not in cleanup, f"cleanup must not contain {destructive!r}"
