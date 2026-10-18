"""The single place where the LLM provider is chosen.

The pipeline never talks to a model vendor directly; it receives an ``LLMInterface``.
neo4j-graphrag's ``OpenAILLM`` accepts a ``base_url``, so the same class serves both a
local vLLM server (which exposes an OpenAI-compatible API) and a hosted API. That is why
there is a factory here and not an abstraction layer.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from neo4j_graphrag.llm import LLMInterface, OpenAILLM
from neo4j_graphrag.utils.rate_limit import RetryRateLimitHandler

logger = logging.getLogger(__name__)

DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"


class RuntimeDeadlineExceeded(TimeoutError):
    """No new model call may start after the experiment deadline."""


def _public_endpoint(url: str | None) -> str | None:
    if not url:
        return url
    try:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))
    except ValueError:
        return "<configured endpoint>"


class MeasuredLLM:
    """Transparent per-document API-call meter, including failed calls.

    Counts application invocations, not hidden HTTP attempts. The production client
    explicitly disables layered SDK/rate-handler retries by default below.
    """

    def __init__(self, llm, metrics: dict[str, Any], deadline: float | None = None) -> None:
        self._schain_wrapped_llm = llm
        self.metrics = metrics
        self.stage = "extraction"
        self.deadline = deadline

    def __getattr__(self, name):
        return getattr(self._schain_wrapped_llm, name)

    def _begin(self) -> tuple[str, float]:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.metrics["stop_reason"] = "runtime_deadline"
            raise RuntimeDeadlineExceeded("runtime deadline reached before model call")
        stage = self.stage
        self.metrics["llm_calls"] = self.metrics.get("llm_calls", 0) + 1
        counts = self.metrics.setdefault("llm_calls_by_stage", {})
        counts[stage] = counts.get(stage, 0) + 1
        return stage, time.monotonic()

    def _finish(self, stage: str, started: float) -> None:
        times = self.metrics.setdefault("llm_seconds_by_stage", {})
        times[stage] = round(times.get(stage, 0.0) + time.monotonic() - started, 4)

    def invoke(self, *args, **kwargs):
        stage, started = self._begin()
        try:
            return self._schain_wrapped_llm.invoke(*args, **kwargs)
        finally:
            self._finish(stage, started)

    async def ainvoke(self, *args, **kwargs):
        stage, started = self._begin()
        try:
            return await self._schain_wrapped_llm.ainvoke(*args, **kwargs)
        finally:
            self._finish(stage, started)


def llm_is_configured(config: dict[str, Any]) -> bool:
    """True when an endpoint is reachable in principle; used to skip integration tests."""
    llm_config = config.get("llm", {})
    if llm_config.get("provider") == "api":
        return bool(os.environ.get("LLM_API_KEY"))
    return bool(llm_config.get("base_url") or os.environ.get("LLM_API_BASE_URL"))


def build_llm(config: dict[str, Any]) -> LLMInterface:
    """Build the LLM client described by the ``llm:`` config block."""
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "local")
    model = llm_config.get("model")
    if not model:
        raise ValueError("llm.model is not set in the config file")

    model_params: dict[str, Any] = {"temperature": llm_config.get("temperature", 0.0)}
    if llm_config.get("max_tokens") is not None:
        max_tokens = int(llm_config["max_tokens"])
        if max_tokens <= 0:
            raise ValueError("llm.max_tokens must be positive")
        model_params["max_tokens"] = max_tokens
    client_retries = int(llm_config.get("client_max_retries", 0))
    rate_attempts = int(llm_config.get("rate_limit_max_attempts", 1))
    if client_retries < 0 or rate_attempts < 1:
        raise ValueError("LLM retry limits must be non-negative retries / positive attempts")

    # JSON mode makes the server guarantee parseable JSON. It is set independently of
    # structured_output: Prompt 1's schema-constrained path passes its own response_format
    # and drops this one, but Prompt 2 is a plain call and would otherwise be left
    # unconstrained when structured_output is on.
    if llm_config.get("json_mode", True):
        model_params["response_format"] = {"type": "json_object"}

    if provider == "local":
        base_url = (
            llm_config.get("base_url")
            or os.environ.get("LLM_API_BASE_URL")
            or DEFAULT_VLLM_BASE_URL
        )
        # vLLM requires a key to be present but ignores its value.
        api_key = os.environ.get("LLM_API_KEY") or "EMPTY"
        # Qwen instruct/non-thinking behaviour is requested through the chat template,
        # which vLLM exposes via extra_body. This is vLLM-specific, so it is not sent to
        # hosted APIs below.
        if not llm_config.get("enable_thinking", False):
            model_params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    elif provider == "api":
        base_url = llm_config.get("base_url") or os.environ.get("LLM_API_BASE_URL")
        api_key = os.environ.get("LLM_API_KEY")
        if not api_key:
            raise ValueError(
                "llm.provider is 'api' but LLM_API_KEY is not set in the environment"
            )
    else:
        raise ValueError(f"unknown llm.provider {provider!r}; expected 'local' or 'api'")

    logger.info(
        "llm provider=%s model=%s base_url=%s json_mode=%s structured_output=%s",
        provider, model, _public_endpoint(base_url),
        "response_format" in model_params, llm_config.get("structured_output", False),
    )
    logger.info("llm request timeout=%ss max_tokens=%s SDK retries=%d rate attempts=%d",
                llm_config.get("request_timeout_seconds", 120),
                model_params.get("max_tokens", "server default"), client_retries, rate_attempts)
    return OpenAILLM(
        model_name=model,
        model_params=model_params,
        base_url=base_url,
        api_key=api_key,
        timeout=llm_config.get("request_timeout_seconds", 120),
        max_retries=client_retries,
        rate_limit_handler=RetryRateLimitHandler(max_attempts=rate_attempts),
    )
