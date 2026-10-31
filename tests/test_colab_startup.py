"""Exercise notebook startup without downloading a model or requiring a GPU."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest


NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.mark.parametrize("name", ["SChain_Expand.ipynb", "SChain_Ingest.ipynb"])
@pytest.mark.parametrize("exit_code", [None, 1])
def test_startup_reports_crash_or_starts_with_memory_limits(name, exit_code, tmp_path):
    notebook = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "".join(cell["source"]).startswith("import httpx"))
    # Keep the real notebook code, redirect only its Colab log location.
    source = source.replace('/content/vllm.log', str(tmp_path / "vllm.log").replace("\\", "/"))
    response = httpx.Response(200, json={"data": [{"id": "test-model"}]},
                              request=httpx.Request("GET", "http://localhost/models"))
    process = Mock()
    process.poll.return_value = exit_code
    process.returncode = exit_code
    namespace = dict(MODEL="test-model", MAX_MODEL_LEN=16384,
                     MODEL_START_TIMEOUT_SECONDS=1200)
    import pathlib
    import subprocess
    import sys
    namespace.update(pathlib=pathlib, subprocess=subprocess, sys=sys)
    with patch("httpx.get", side_effect=[httpx.ConnectError("not running"), response]), \
         patch("httpx.post", return_value=httpx.Response(200, json={
             "choices": [{"message": {"content": "OK"}}]})), \
         patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="GPU: test", stderr="")), \
         patch("subprocess.Popen", return_value=process) as start:
        if exit_code is not None:
            with pytest.raises(RuntimeError, match="vLLM exited with code 1"):
                exec(compile(source, name, "exec"), namespace)
        else:
            exec(compile(source, name, "exec"), namespace)
        command = start.call_args.args[0]
        assert command[command.index("--dtype") + 1] == "half"
        assert command[command.index("--max-model-len") + 1] == "16384"
        assert "--enforce-eager" in command
