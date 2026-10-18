"""Small deterministic document policy; ceilings are safety limits, never targets."""
from __future__ import annotations


def entity_document_limits(expansion: dict, tier: int = 0) -> tuple[int, int]:
    retrieval = expansion.get("retrieval") or {}
    if tier >= int(expansion.get("max_tier", 3)):
        return 0, 0
    targets = retrieval.get("target_documents_by_tier") or {}
    maxima = retrieval.get("max_documents_by_tier") or {}
    target = int(targets.get(tier, targets.get(str(tier), retrieval.get("target_documents_per_entity", 0))) or 0)
    tier_max = int(maxima.get(tier, maxima.get(str(tier), 0)) or 0)
    scalar_max = int(retrieval.get("max_documents_per_entity") or 0)
    positive = [value for value in (tier_max, scalar_max) if value > 0]
    hard = min(positive) if positive else target
    if hard:
        target = min(target, hard)
    return target, hard


def scheduling_batch_sizes(expansion: dict) -> dict:
    """Soft per-run scheduling batch sizes (ORDER fairness only; never permanent exclusion).

    Config key ``max_expandable_entities_by_tier`` is retained for compatibility but
    no longer means \"only these N entities ever expand\".
    """
    return (
        expansion.get("scheduling_batch_size_by_tier")
        or expansion.get("max_expandable_entities_by_tier")
        or {}
    )


def breadth_caps(expansion: dict) -> dict:
    """Backward-compatible alias for :func:`scheduling_batch_sizes`."""
    return scheduling_batch_sizes(expansion)


def documents_per_run_limit(expansion: dict) -> int:
    """Per-run document safety ceiling. Pauses the run; does not prune the frontier.

    Prefer ``max_documents_per_run`` when set; otherwise ``max_documents`` is treated
    as a per-run allowance (lifetime progress remains in SQLite usage tables).
    """
    for key in ("max_documents_per_run", "max_documents"):
        value = expansion.get(key)
        if value:
            return int(value)
    return 0


def remaining_target_reserve(connection, config: dict) -> int:
    """Protect unmet targets of known pending/processing frontier entities.

    Deeper undiscovered tiers use scheduling-batch sizes only as a soft planning
    estimate so shallow above-target spend does not starve future depth. Finished
    entities release their unmet target. No document is required merely to fill a
    target, and no supplier is permanently excluded by this reserve.
    """
    from ..config import seed_entities
    expansion = config.get("expansion") or {}
    caps = scheduling_batch_sizes(expansion)
    reserve = 0
    for tier in range(int(expansion.get("max_tier", 3))):
        target, _ = entity_document_limits(expansion, tier)
        if not target:
            continue
        rows = connection.execute(
            "SELECT f.entity_key,f.status,COUNT(u.doc_id) AS used FROM frontier f LEFT JOIN "
            "expansion_document_usage u ON u.entity_key=f.entity_key "
            "WHERE f.tier=? AND f.expandable=1 AND (f.attempts>0 OR f.tier=0 OR f.status IN ('PENDING','PROCESSING')) "
            "GROUP BY f.entity_key", (tier,)).fetchall()
        # Unmet targets for entities already on the frontier.
        reserve += sum(max(0, target - int(row["used"] or 0)) for row in rows if row["status"] != "DONE")
        # Soft planning for deeper tiers not yet populated (expected batch breadth).
        if tier > 0 and not rows:
            planned = int(caps.get(tier, caps.get(str(tier), 0)) or 0)
            reserve += planned * target
        elif tier == 0 and not rows:
            reserve += len(seed_entities(config)) * target
    return reserve
