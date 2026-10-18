"""The server scripts, exercised locally without a GPU, a real model or a real Neo4j.

The scripts only prepare services and launch the existing Python CLI, so what is worth
testing is exactly that: argument handling, command construction, health detection, and
the ownership rule that decides what cleanup is allowed to stop.
"""

from __future__ import annotations

import os
import shlex
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
    environment = {
        **os.environ,
        "SCHAIN_NO_TMUX": "1",
        # Point health checks at a closed port by default: no script test should depend
        # on whatever happens to be listening on 7687 on the developer machine, and the
        # Bolt probe would otherwise wait its full timeout on an unrelated listener.
        "NEO4J_URI": "bolt://127.0.0.1:1",
        "SCHAIN_NEO4J_PROBE_TIMEOUT": "2",
        "SCHAIN_HTTP_PROBE_TIMEOUT": "1",
        **(env or {}),
    }
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
    assert "--max-tier 1 --max-documents 3" in output
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
    result = run([str(PREFLIGHT)], env={"NEO4J_URI": "bolt://db.test:7687"})
    assert "Qwen/Qwen3-4B-Instruct-2507" in result.stdout
    assert "bolt://db.test:7687" in result.stdout


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
            f'tcp_open 127.0.0.1 {port} && echo PORT_OPEN; '
            # an HTTP server is listening, but it is not Neo4j, so this must say no
            f'neo4j_reachable bolt://127.0.0.1:{port} || echo NOT_NEO4J; '
            f'llm_healthy http://127.0.0.1:{free_port()}/v1 || echo LLM_DOWN'
        )
        result = run(["bash", "-c", script])
        assert "LLM_UP" in result.stdout
        assert "PORT_OPEN" in result.stdout
        assert "NOT_NEO4J" in result.stdout
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

    result = run([str(RUN), "--dry-run", "--smoke-test"], env={
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
                     env={"NEO4J_URI": f"bolt://127.0.0.1:{neo4j_port}",
                          "SCHAIN_NEO4J_SKIP_BOLT_CHECK": "1"})
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


def test_config_inspection_works_without_a_project_venv(tmp_path):
    """A fresh checkout must report real configured endpoints before installing .venv."""
    config = write_temp_config(tmp_path, 9123)
    command = (
        f"source {shlex.quote(str(LIB))}; "
        f"VENV_PY={shlex.quote(str(tmp_path / 'missing-python'))}; "
        f"config_values {shlex.quote(str(config))}"
    )
    result = run(
        ["bash", "-c", command], cwd=tmp_path,
        env={"NEO4J_URI": "bolt://configured.test:7687"},
    )
    assert result.returncode == 0, result.stderr
    assert "CFG_LLM_BASE_URL=http://127.0.0.1:9123/v1" in result.stdout
    assert "CFG_NEO4J_URI=bolt://configured.test:7687" in result.stdout
    assert "Qwen/Qwen3-4B-Instruct-2507" in result.stdout
    assert not (tmp_path / ".venv").exists()


def test_open_http_port_is_not_neo4j_when_project_venv_is_missing(tmp_path):
    port = free_port()
    listener = start_http_stub(port)
    try:
        command = (
            f"source {shlex.quote(str(LIB))}; "
            f"VENV_PY={shlex.quote(str(tmp_path / 'missing-python'))}; "
            f"neo4j_reachable bolt://127.0.0.1:{port}"
        )
        result = run(
            ["bash", "-c", command], env={"SCHAIN_NEO4J_SKIP_BOLT_CHECK": ""},
        )
        assert result.returncode != 0
        assert alive(listener)
    finally:
        listener.terminate()
        listener.wait(timeout=10)


def test_occupied_unverified_neo4j_port_never_starts_another_instance(tmp_path):
    port = free_port()
    listener = start_http_stub(port)
    try:
        config = write_temp_config(tmp_path, port)
        result = run(
            [str(RUN), "--dry-run", "--config", str(config)],
            env={"NEO4J_URI": f"bolt://127.0.0.1:{port}",
                 "SCHAIN_NEO4J_SKIP_BOLT_CHECK": ""},
        )
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "occupied but Neo4j could not be verified" in output
        assert "would start Neo4j" not in output
        assert "would install and start local Neo4j" not in output
        assert alive(listener)
    finally:
        listener.terminate()
        listener.wait(timeout=10)


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
                "SCHAIN_NEO4J_SKIP_BOLT_CHECK": "1",
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


# --- SLURM: interactive for a smoke test, batch for production -----------------------


def fake_cluster(tmp_path, sbatch_body="#!/bin/sh\necho 'Submitted batch job 66353'\n"):
    """A PATH where SLURM exists and nvidia-smi reports no GPU, as on a login node."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    for name in ("srun", "sinfo", "nvidia-smi"):
        stub = fake_bin / name
        stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        stub.chmod(0o755)
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(sbatch_body, encoding="utf-8")
    sbatch.chmod(0o755)
    return fake_bin


def cluster_env(tmp_path, **extra):
    return {
        "PATH": f"{fake_cluster(tmp_path)}:{os.environ['PATH']}",
        "SCHAIN_SLURM_PARTITION": "ampere",
        "SCHAIN_SLURM_ACCOUNT": "ampere",
        "SCHAIN_SLURM_GRES": "gpu:a100_80gb:1",
        "SCHAIN_RUNTIME_DIR": str(tmp_path / "runtime"),
        # a closed port, so the health checks fail fast and deterministically instead
        # of depending on whatever happens to hold 7687 on the developer machine
        "NEO4J_URI": "bolt://127.0.0.1:1",
        "SCHAIN_NEO4J_PROBE_TIMEOUT": "2",
        **extra,
    }


def test_production_submits_a_batch_job_not_an_interactive_one(tmp_path):
    """Interactive allocations are capped at 2h here, so production must use sbatch."""
    result = run([str(RUN), "--dry-run"], env=cluster_env(tmp_path))
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "would run: sbatch" in output
    assert "exec: srun" not in output
    assert "--partition=ampere" in output and "--account=ampere" in output
    assert "--gres=gpu:a100_80gb:1" in output


def test_the_smoke_test_still_uses_an_interactive_allocation(tmp_path):
    result = run([str(RUN), "--dry-run", "--smoke-test"], env=cluster_env(tmp_path))
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "would exec: srun" in output
    assert "sbatch" not in output
    assert "--time=02:00:00" in output          # within the interactive cap


def test_production_time_limit_is_configurable_and_defaults_to_a_day(tmp_path):
    default = run([str(RUN), "--dry-run"], env=cluster_env(tmp_path))
    assert "--time=24:00:00" in default.stdout + default.stderr

    longer = run([str(RUN), "--dry-run"],
                 env=cluster_env(tmp_path, SCHAIN_SLURM_TIME="48:00:00"))
    assert "--time=48:00:00" in longer.stdout + longer.stderr


def test_the_generated_batch_file_is_complete(tmp_path):
    """The submitted script must carry the directives and re-enter the runner in-job."""
    env = cluster_env(tmp_path)
    result = run([str(RUN)], env=env)          # a real submission against fake sbatch
    assert result.returncode == 0, result.stderr

    batch_files = sorted((tmp_path / "runtime" / "slurm").glob("*.sbatch"))
    assert len(batch_files) == 1
    body = batch_files[0].read_text(encoding="utf-8")

    assert body.startswith("#!/bin/bash")
    for directive in ("--partition=ampere", "--account=ampere",
                      "--gres=gpu:a100_80gb:1", "--cpus-per-task=8", "--time=24:00:00"):
        assert f"#SBATCH {directive}" in body
    assert "#SBATCH --output=" in body and "/logs/expand_" in body
    assert f"cd {REPO_ROOT}" in body
    # inside the job: no tmux, no re-submission, and stdout is SLURM's to capture
    assert "SCHAIN_NO_TMUX=1" in body and "SCHAIN_IN_ALLOCATION=1" in body
    assert "SCHAIN_LOG_TO_STDOUT=1" in body
    assert body.rstrip().endswith("--foreground")


def test_submission_reports_the_job_id_and_records_it(tmp_path):
    env = cluster_env(tmp_path)
    result = run([str(RUN)], env=env)
    output = result.stdout + result.stderr

    assert "Submitted SChain EXPAND" in output
    assert "SLURM job: 66353" in output
    assert "Status:    ./scripts/server_status.sh" in output
    assert "scancel 66353" in output
    assert (tmp_path / "runtime" / "slurm" / "current_job").read_text().strip() == "66353"


def test_a_failed_submission_is_reported(tmp_path):
    env = cluster_env(tmp_path)
    env["PATH"] = f"{fake_cluster(tmp_path, sbatch_body=chr(35) + '!/bin/sh' + chr(10) + 'echo boom >&2' + chr(10) + 'exit 1' + chr(10))}:{os.environ['PATH']}"
    result = run([str(RUN)], env=env)
    assert result.returncode != 0
    assert "sbatch failed" in result.stderr


def test_no_tmux_wrapper_is_used_on_a_slurm_cluster(tmp_path):
    """SLURM owns the detached process; wrapping it in tmux would add nothing."""
    result = run([str(RUN), "--dry-run"], env=cluster_env(tmp_path))
    assert "starting in tmux session" not in result.stdout + result.stderr


def test_inside_an_allocation_no_further_allocation_is_requested(tmp_path):
    env = cluster_env(tmp_path, SLURM_JOB_ID="66353")
    output = run([str(RUN), "--dry-run"], env=env).stdout
    assert "sbatch" not in output and "would exec: srun" not in output
    assert "-m schain expand" in output


# --- Neo4j without Docker -------------------------------------------------------------


def no_docker_env(tmp_path, **extra):
    """A PATH with SLURM and without docker, matching the real cluster."""
    fake_bin = fake_cluster(tmp_path)
    blocker = tmp_path / "nodocker"
    blocker.mkdir(exist_ok=True)
    # a PATH that keeps the usual tools but hides docker
    kept = ":".join(part for part in os.environ["PATH"].split(":") if part)
    env = {
        "PATH": f"{fake_bin}:{blocker}:{kept}",
        "SCHAIN_RUNTIME_DIR": str(tmp_path / "runtime"),
        "SCHAIN_NEO4J_PROBE_TIMEOUT": "2",
        **extra,
    }
    return env


NEEDED_TOOLS = [
    "bash", "sh", "env", "date", "mkdir", "sed", "grep", "tr", "cut", "seq", "sleep",
    "curl", "timeout", "cat", "wc", "head", "tail", "ls", "df", "uname", "hostname",
    "python3", "tar", "awk", "chmod", "cp", "rm", "dirname", "basename",
]


def test_without_docker_the_plan_is_a_local_neo4j(tmp_path):
    """The cluster has no Docker and no root, so the fallback is a user-space install."""
    bin_dir = tmp_path / "curated_bin"
    bin_dir.mkdir()
    for tool in NEEDED_TOOLS:
        found = shutil.which(tool)
        if found is None:
            pytest.skip(f"{tool} is not available, cannot build a docker-free PATH")
        (bin_dir / tool).symlink_to(found)
    assert shutil.which("docker", path=str(bin_dir)) is None

    result = run([str(RUN), "--dry-run"],
                 env={"PATH": str(bin_dir),
                      "SCHAIN_RUNTIME_DIR": str(tmp_path / "runtime")})
    output = result.stdout + result.stderr
    assert "would install and start local Neo4j" in output
    assert "5.26" in output          # the pinned version, not "latest"


def test_local_neo4j_requires_a_supported_java(tmp_path):
    """Neo4j 5 needs Java 21+; an older runtime must fail clearly, before downloading."""
    fake_bin = tmp_path / "oldjava"
    fake_bin.mkdir()
    java = fake_bin / "java"
    java.write_text('#!/bin/sh\necho \'openjdk version "11.0.20" 2023-07-18\' >&2\n',
                    encoding="utf-8")
    java.chmod(0o755)
    result = run(["bash", "-c", f'source {LIB}; java_major'],
                 env={"PATH": f"{fake_bin}:{os.environ['PATH']}"})
    assert result.stdout.strip() == "11"

    script = (REPO_ROOT / "scripts" / "server_run.sh").read_text(encoding="utf-8")
    assert "Java 21/25 is required for local Neo4j" in script
    assert '[ "$major" -lt 21 ]' in script


def test_local_neo4j_keeps_data_outside_the_distribution_and_off_tmp():
    script = (REPO_ROOT / "scripts" / "server_run.sh").read_text(encoding="utf-8")
    assert 'server.directories.data=$data_dir' in script
    assert 'RUNTIME_DIR/neo4j/data' in script
    assert "/tmp" not in script          # graph data never lands in /tmp
    # the version is pinned in one place, not resolved to "latest" per run
    assert 'NEO4J_VERSION="${SCHAIN_NEO4J_VERSION:-' in script


def test_a_generated_neo4j_password_is_never_printed():
    script = (REPO_ROOT / "scripts" / "server_run.sh").read_text(encoding="utf-8")
    assert "NEO4J_PASSWORD: generated and stored in .env" in script
    assert "chmod 600" in script
    # the value itself is only ever written to .env, never echoed
    assert 'log "NEO4J_PASSWORD: $' not in script
    assert 'echo "$NEO4J_PASSWORD"' not in script


def test_cleanup_stops_a_local_neo4j_it_started():
    script = (REPO_ROOT / "scripts" / "server_run.sh").read_text(encoding="utf-8")
    cleanup = script[script.index("cleanup() {"):script.index("trap cleanup")]
    assert '[ "$OWNED_NEO4J" = "local" ]' in cleanup
    assert "bin/neo4j\" stop" in cleanup
    assert "data is kept" in cleanup
    for destructive in ("rm ", "docker rm", "--restart"):
        assert destructive not in cleanup


# --- vLLM installation ----------------------------------------------------------------


def test_vllm_is_installed_once_into_the_existing_venv():
    script = (REPO_ROOT / "scripts" / "server_run.sh").read_text(encoding="utf-8")
    # reuse first
    assert '[ -x "$REPO_ROOT/.venv/bin/vllm" ]' in script
    # install into the same environment the runner checks, not a new one
    assert '"$VENV_PY" -m pip install --quiet vllm' in script
    # only when a GPU is actually present, and only when the endpoint is down
    assert '[ "$VLLM_AUTO_INSTALL" = "1" ] && [ "$(gpu_count)" -ge 1 ]' in script


def test_first_run_model_download_is_announced_not_silent():
    script = (REPO_ROOT / "scripts" / "server_run.sh").read_text(encoding="utf-8")
    assert "the first run downloads the model" in script
    assert "this is not a hang" in script
    assert 'log "HF_TOKEN: $(secret_state HF_TOKEN)"' in script   # presence only


def test_batch_preserves_nondefault_arguments_and_quoting(tmp_path):
    config = write_temp_config(tmp_path, 1)
    input_path = tmp_path / 'documents with spaces'
    result = run([str(RUN), 'ingest', str(input_path), '--config', str(config), '--restart'],
                 env=cluster_env(tmp_path))
    assert result.returncode == 0, result.stderr
    body = next((tmp_path / 'runtime' / 'slurm').glob('*.sbatch')).read_text()
    command = next(line for line in body.splitlines() if line.startswith('exec '))
    tokens = shlex.split(command)[1:]
    assert tokens == [str(RUN), 'ingest', str(input_path), '--config', str(config),
                      '--restart', '--foreground']


def test_interactive_allocation_plan_preserves_nondefault_arguments(tmp_path):
    config = write_temp_config(tmp_path, 1)
    result = run([str(RUN), '--dry-run', '--smoke-test', '--config', str(config), '--restart'],
                 env=cluster_env(tmp_path))
    assert result.returncode == 0, result.stderr
    allocation = next(line for line in result.stdout.splitlines() if 'would exec: srun' in line)
    assert '--smoke-test' in allocation and '--restart' in allocation
    assert str(config) in allocation


def test_weight_override_uses_configured_api_model_name():
    result = run([str(RUN), '--dry-run'], env={'SCHAIN_VLLM_MODEL': 'different/weights'})
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines() if 'would start vLLM:' in line)
    assert 'serve different/weights' in line
    assert '--served-model-name Qwen/Qwen3-4B-Instruct-2507' in line


def test_tmux_preserves_original_mode_config_and_spaced_input(tmp_path):
    fake_bin = tmp_path / 'tmux_bin'
    fake_bin.mkdir()
    capture = tmp_path / 'tmux_command.txt'
    tmux = fake_bin / 'tmux'
    tmux.write_text('#!/bin/bash\n'
                    'if [ "$1" = has-session ]; then exit 1; fi\n'
                    'if [ "$1" = new-session ]; then printf "%s\\n" "${@: -1}" > '
                    + shlex.quote(str(capture)) + '; fi\n')
    tmux.chmod(0o755)
    input_path = tmp_path / 'documents with spaces'
    result = run([str(RUN), 'ingest', str(input_path), '--config', 'config/ingest.yaml', '--restart'],
                 env={'PATH': f"{fake_bin}:{os.environ['PATH']}", 'SCHAIN_NO_TMUX': '', 'TMUX': ''})
    assert result.returncode == 0, result.stderr
    tokens = shlex.split(capture.read_text())
    run_start = tokens.index(str(RUN))
    assert tokens[run_start:] == [str(RUN), 'ingest', str(input_path), '--config',
                                 'config/ingest.yaml', '--restart']
