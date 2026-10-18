"""Model-independent semantic task interfaces.

Business logic should depend on these roles, not on Qwen-/OpenAI-specific APIs.
Today and tomorrow both consume the configured LLM provider via MeasuredLLM.
"""

from __future__ import annotations

from typing import Any, Protocol


class Extractor(Protocol):
    def invoke(self, prompt: str) -> Any: ...


class EvidenceJudge(Protocol):
    def invoke(self, prompt: str) -> Any: ...


class DirectionJudge(Protocol):
    def invoke(self, prompt: str) -> Any: ...


class QueryPlanner(Protocol):
    def invoke(self, prompt: str) -> Any: ...


class EntityResolver(Protocol):
    def invoke(self, prompt: str) -> Any: ...


__all__ = [
    "Extractor",
    "EvidenceJudge",
    "DirectionJudge",
    "QueryPlanner",
    "EntityResolver",
]
