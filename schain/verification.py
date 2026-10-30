"""Targeted multi-source verification has been removed.

This module remains as a no-op stub so older imports and checkpoints do not crash.
Independent-source / second-source verification is no longer part of the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def verify_supplies_to_edges(
    config: dict[str, Any],
    connection,
    llm,
    *,
    deadline: Optional[float] = None,
    documents_used: int = 0,
    customer_ids: Optional[set[str]] = None,
) -> dict[str, Any]:
    """No-op: multi-source verification searches are disabled."""
    logger.debug("verification searches disabled; skipping")
    return {
        "enabled": False,
        "edges_checked": 0,
        "documents_checked": documents_used,
        "queries_executed": 0,
        "promoted": 0,
        "stop_reason": "disabled",
    }
