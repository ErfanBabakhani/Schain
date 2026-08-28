"""The single place where the LLM provider is chosen.

The KG code below never talks to a model vendor directly; it receives an
``LLMInterface``. neo4j-graphrag's ``OpenAILLM`` accepts a ``base_url``, so the same
class serves both a local vLLM server (which exposes an OpenAI-compatible API) and a
hosted API, and no custom adapter is needed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from neo4j_graphrag.llm import LLMInterface, OpenAILLM

logger = logging.getLogger(__name__)

DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"


def build_llm(config: dict[str, Any]) -> LLMInterface:
    """Build the LLM client described by the ``llm:`` config block."""
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "local")
    model = llm_config.get("model")
    if not model:
        raise ValueError("llm.model is not set in the config file")

    model_params: dict[str, Any] = {"temperature": llm_config.get("temperature", 0.0)}

    if provider == "local":
        base_url = (
            llm_config.get("base_url")
            or os.environ.get("LLM_API_BASE_URL")
            or DEFAULT_VLLM_BASE_URL
        )
        # vLLM requires a key to be present but ignores its value.
        api_key = os.environ.get("LLM_API_KEY") or "EMPTY"
        # Qwen instruct/non-thinking behaviour is requested through the chat template,
        # which vLLM exposes via extra_body. This is vLLM-specific, so it is not sent
        # to hosted APIs below.
        if not llm_config.get("enable_thinking", False):
            model_params["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
    elif provider == "api":
        base_url = llm_config.get("base_url") or os.environ.get("LLM_API_BASE_URL")
        api_key = os.environ.get("LLM_API_KEY")
        if not api_key:
            raise ValueError(
                "llm.provider is 'api' but LLM_API_KEY is not set in the environment"
            )
    else:
        raise ValueError(f"unknown llm.provider {provider!r}; expected 'local' or 'api'")

    logger.info("llm provider=%s model=%s base_url=%s", provider, model, base_url)
    return OpenAILLM(
        model_name=model,
        model_params=model_params,
        base_url=base_url,
        api_key=api_key,
    )
