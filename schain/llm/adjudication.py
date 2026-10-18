"""Strong-LLM semantic adjudication for direction and unclear evidence.

Deterministic safety remains authoritative for quote grounding and ontology.
These judges only resolve genuine ambiguity; they never invent unsupported facts.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

from json_repair import repair_json

from ..config import storage_path

logger = logging.getLogger(__name__)

DIRECTION_PROMPT = """\
Decide whether the evidence establishes a directional suppliesTo relation.

Proposed subject (claimed supplier): {subject}
Proposed object (claimed customer): {object}
Evidence quote:
\"\"\"{evidence}\"\"\"
Limited context:
\"\"\"{context}\"\"\"

Return JSON only:
{{
  "supported": true/false,
  "supplier": "name or null",
  "customer": "name or null",
  "status": "resolved|ambiguous|unsupported",
  "correct_relation": "suppliesTo|none",
  "reason": "short grounded reason"
}}

Rules:
- Use ONLY the supplied evidence/context. Do not use outside knowledge.
- suppliesTo means supplier → customer receiving supply inputs.
- Distributor/reseller/aftermarket of a brand's parts is NOT suppliesTo the brand
  unless the brand is clearly the receiving customer.
- Partnership / collaboration / strategic pact without clear supplier roles =
  ambiguous or unsupported.
- If supplier/customer roles are clear for a real supply relationship, status=resolved.
- If a supply deal is mentioned but roles are unclear, status=ambiguous.
- If no supply relationship is established, status=unsupported and correct_relation=none.
"""

EVIDENCE_SECOND_PASS_PROMPT = """\
Does this exact evidence establish the proposed relation between these exact endpoints?

Subject: {subject} ({subject_type})
Relation: {relation}
Object: {object} ({object_type})
Evidence:
\"\"\"{evidence}\"\"\"

Answer JSON only:
{{
  "supported": true/false,
  "decision": "YES"|"NO"|"AMBIGUOUS",
  "correct_relation": "suppliesTo|owns|produces|locatedIn|none",
  "correct_direction": true/false/null,
  "reason": "brief grounded explanation"
}}

Rules:
- Use only the evidence. Do not invent facts from model memory.
- YES only if THIS subject --relation--> object is directly supported.
- Distributor/reseller/aftermarket sales of a brand's parts to shops/consumers
  is NOT suppliesTo the brand unless the brand is clearly the receiving customer.
- Planned acquisition is NOT owns. Co-design/offer without manufacturing is NOT produces.
- If unsupported, set correct_relation to none (or the single relation that IS supported).
"""


def _cache_path(config: dict[str, Any], kind: str, key: str) -> Path:
    root = storage_path(config, "state_db").parent / "llm_adjudication" / kind
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{key}.json"


def _hash_key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


def adjudicate_supply_direction(
    llm,
    config: dict[str, Any],
    *,
    subject: str,
    obj: str,
    evidence: str,
    context: str = "",
) -> dict[str, Any]:
    """LLM direction judge. Returns status resolved|ambiguous|unsupported."""
    retrieval = (config.get("expansion") or {}).get("retrieval") or {}
    if not retrieval.get("semantic_direction_adjudication", True) or llm is None:
        return {"supported": False, "status": "ambiguous", "reason": "adjudication_disabled",
                "supplier": None, "customer": None}
    key = _hash_key("direction", subject, obj, evidence[:500], context[:300])
    path = _cache_path(config, "direction", key)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    prompt = DIRECTION_PROMPT.format(
        subject=subject, object=obj, evidence=evidence or "", context=(context or "")[:1200],
    )
    previous = getattr(llm, "stage", None)
    try:
        if hasattr(llm, "stage"):
            llm.stage = "direction_judge"
        raw = llm.invoke(prompt).content
        payload = json.loads(repair_json(raw))
    except Exception as error:  # noqa: BLE001
        logger.warning("direction adjudication failed: %s", error)
        payload = {
            "supported": False, "supplier": None, "customer": None,
            "status": "ambiguous", "reason": f"adjudication_error:{type(error).__name__}",
        }
    finally:
        if hasattr(llm, "stage") and previous is not None:
            llm.stage = previous
    status = str(payload.get("status") or "ambiguous").casefold()
    if status not in ("resolved", "ambiguous", "unsupported"):
        status = "ambiguous"
    result = {
        "supported": bool(payload.get("supported")) and status == "resolved",
        "supplier": payload.get("supplier"),
        "customer": payload.get("customer"),
        "status": status,
        "reason": str(payload.get("reason") or "")[:300],
    }
    try:
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except OSError:
        pass
    return result


def second_pass_evidence(
    llm,
    config: dict[str, Any],
    row: dict[str, Any],
) -> str:
    """Return YES / NO / AMBIGUOUS for an evidence_unclear candidate.

    Semantic unsupported/ambiguous answers are final decisions — callers must not
    retry them as transport failures. The cached JSON also stores the full judge
    contract (supported, correct_relation, correct_direction, reason).
    """
    retrieval = (config.get("expansion") or {}).get("retrieval") or {}
    if not retrieval.get("evidence_second_pass", True) or llm is None:
        return "AMBIGUOUS"
    key = _hash_key(
        "evidence2",
        str(row.get("subject_id") or ""),
        str(row.get("relation") or ""),
        str(row.get("object_id") or ""),
        str(row.get("evidence") or "")[:500],
    )
    path = _cache_path(config, "evidence", key)
    if path.exists():
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("decision") or "AMBIGUOUS")
        except (ValueError, OSError):
            pass
    prompt = EVIDENCE_SECOND_PASS_PROMPT.format(
        subject=row.get("subject_name"),
        subject_type=row.get("subject_type"),
        relation=row.get("relation"),
        object=row.get("object_name"),
        object_type=row.get("object_type"),
        evidence=row.get("evidence") or "",
    )
    previous = getattr(llm, "stage", None)
    decision = "AMBIGUOUS"
    payload: dict[str, Any] = {
        "supported": False,
        "decision": "AMBIGUOUS",
        "correct_relation": "none",
        "correct_direction": None,
        "reason": "",
    }
    try:
        if hasattr(llm, "stage"):
            llm.stage = "evidence_judge"
        raw = llm.invoke(prompt).content
        parsed = json.loads(repair_json(raw))
        decision = str(parsed.get("decision") or "AMBIGUOUS").upper()
        if decision not in ("YES", "NO", "AMBIGUOUS"):
            decision = "AMBIGUOUS"
        correct_relation = str(parsed.get("correct_relation") or "none")
        if correct_relation not in ("suppliesTo", "owns", "produces", "locatedIn", "none"):
            correct_relation = "none"
        direction = parsed.get("correct_direction", None)
        if direction not in (True, False, None):
            direction = None
        payload = {
            "supported": bool(parsed.get("supported")) if decision == "YES" else False,
            "decision": decision,
            "correct_relation": correct_relation,
            "correct_direction": direction,
            "reason": str(parsed.get("reason") or "")[:300],
        }
    except Exception as error:  # noqa: BLE001
        logger.warning("evidence second-pass failed: %s", error)
        decision = "AMBIGUOUS"
        payload["reason"] = f"adjudication_error:{type(error).__name__}"
    finally:
        if hasattr(llm, "stage") and previous is not None:
            llm.stage = previous
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass
    return decision
