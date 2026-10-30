"""Executable relation policy shared by retrieval and semantic validation.

Templates identify local roles; they never replace evidence entailment. The only
pattern operator in YAML is ``.*``: a bounded, same-clause phrase, not arbitrary regex.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

from ..config import REPO_ROOT, resolve_path
from ..names import normalization_key, strip_accents
from .entity_types import company_type_hint

logger = logging.getLogger(__name__)
DEFAULT_POLICY_PATH = REPO_ROOT / "config" / "relation_evidence.yaml"
UPSTREAM_EXPLICIT = "UPSTREAM_EXPLICIT"
DIRECTION_AMBIGUOUS = "DIRECTION_AMBIGUOUS"
DOWNSTREAM_EXPLICIT = "DOWNSTREAM_EXPLICIT"


@lru_cache(maxsize=8)
def _load_policy_file(path: str, modified_ns: int) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        policy = yaml.safe_load(handle) or {}
    if not isinstance(policy, dict):
        raise ValueError("relation evidence policy must be a mapping")
    return policy


def load_relation_evidence(path: Optional[str] = None) -> dict[str, Any]:
    policy_path = Path(resolve_path(path) if path else DEFAULT_POLICY_PATH)
    if not policy_path.exists():
        raise FileNotFoundError(f"relation evidence policy missing at {policy_path}")
    return _load_policy_file(str(policy_path), policy_path.stat().st_mtime_ns)


def _norm_evidence(text: str) -> str:
    # Never apply company-suffix stripping to prose (e.g. 'selected X as').
    return " ".join(re.sub(r"[^\w\s]", " ", strip_accents(text or "").casefold()).split())


def _clauses(text: str) -> list[str]:
    # Preserve common legal abbreviations so 'Tesla Inc. purchases...' stays local.
    text = re.sub(r"\b(Inc|Ltd|Co|Corp|AG)\.(?=\s)", r"\1", text or "", flags=re.I)
    return [part for raw in re.split(r"[.!?;\n]+|\b(?:but|whereas)\b", text, flags=re.I)
            if (part := _norm_evidence(raw))]


def _phrase_pattern(template: str, subject: str | None = None, obj: str | None = None) -> str:
    """Compile literals plus bounded wildcard slots; user YAML is never raw regex."""
    raw_tokens = re.split(r"(\{subject\}|\{object\}|\.\*)", template)
    pieces: list[str] = []
    for token in raw_tokens:
        if token == "{subject}":
            pieces.append(re.escape(subject) if subject else r"(?:\w+\s+){0,10}\w+")
        elif token == "{object}":
            pieces.append(re.escape(obj) if obj else r"(?:\w+\s+){0,10}\w+")
        elif token == ".*":
            pieces.append("__GAP__")
        elif value := _norm_evidence(token):
            pieces.extend(re.escape(word) for word in value.split())
    pattern = r"\s+".join(pieces)
    # Slot may be empty, but cannot cross a clause boundary or grow unbounded.
    pattern = pattern.replace(r"\s+__GAP__\s+", r"\s+(?:\w+\s+){0,16}")
    pattern = pattern.replace("__GAP__", r"(?:\w+\s+){0,16}")
    return r"(?<!\w)" + pattern + r"(?!\w)"


def _contains_cue(evidence_norm: str, cues: list[str]) -> bool:
    return any(re.search(_phrase_pattern(cue), evidence_norm) for cue in cues or [] if cue)


def has_relation_cues(text: str, relation: str = "suppliesTo", policy=None) -> bool:
    entry = (load_relation_evidence() if policy is None else policy).get(relation) or {}
    return any(_contains_cue(clause, entry.get(key) or [])
               for clause in _clauses(text) for key in ("positive", "ambiguous"))


def lexical_relation_gate(relation: str, evidence: str, *, policy=None) -> tuple[Optional[bool], str]:
    entry = (load_relation_evidence() if policy is None else policy).get(relation) or {}
    if not entry:
        return None, "no_policy"
    evidence_norm = _norm_evidence(evidence)
    if not evidence_norm:
        return False, "empty_evidence"
    positive = _contains_cue(evidence_norm, entry.get("positive") or [])
    insufficient = _contains_cue(evidence_norm, entry.get("insufficient") or [])
    if insufficient and not positive:
        return False, "insufficient_relation_language"
    if positive:
        return True, "positive_relation_language"
    if _contains_cue(evidence_norm, entry.get("ambiguous") or []):
        return None, "ambiguous_supply_language"
    return None, "no_strong_lexical_cue"


def _forms(name: str, aliases: Optional[list[str]]) -> list[str]:
    forms = set()
    for value in [name, *(aliases or [])]:
        if value:
            forms.update(filter(None, (_norm_evidence(value), normalization_key(value))))
    return sorted(forms, key=lambda value: (-len(value), value))


def _template_matches(entry, clause: str, subject: str | None, obj: str | None):
    if subject and obj and subject == obj:
        return None
    for key, reason in (("direction_templates_supplier_first", "direction_supplier_first"),
                        ("direction_templates_customer_first", "direction_customer_first")):
        for template in entry.get(key) or []:
            for match in re.finditer(_phrase_pattern(template, subject, obj), clause):
                # Negation/modal speculation is a conservative deterministic guard.
                # Entailment still decides tense, entity identity and context.
                before = clause[max(0, match.start() - 30):match.start()]
                span = before + match.group()
                if re.search(r"\b(?:not|never|no longer|neither|denies|denied|may|might|could|whether|if)\b", span):
                    continue
                # Coordinated clauses cannot lend an unrelated actor their predicate.
                coordinated = re.split(r"\b(?:and|while)\b", match.group(), maxsplit=1)
                if len(coordinated) > 1 and _contains_cue(coordinated[1], entry.get("positive") or []):
                    continue
                # A company root is not a model customer (BMW != BMW i7).
                if obj and match.group().endswith(obj):
                    tail = clause[match.end():].split()
                    if tail and company_type_hint(obj + " " + tail[0])[0] == "Product":
                        continue
                return reason
    return None


def check_supplies_to_direction(subject_name: str, object_name: str, evidence: str, *,
                                policy=None, subject_aliases=None, object_aliases=None) -> tuple[bool, str]:
    entry = (load_relation_evidence() if policy is None else policy).get("suppliesTo") or {}
    for clause in _clauses(evidence):
        for subject in _forms(subject_name, subject_aliases):
            for obj in _forms(object_name, object_aliases):
                if reason := _template_matches(entry, clause, subject, obj):
                    return True, reason
    return False, "ambiguous_supply_direction"


def classify_supply_direction(origin_name: str, text: str, aliases=None, policy=None) -> str:
    """Soft retrieval classification: an unknown opposite actor is allowed here only."""
    entry = (load_relation_evidence() if policy is None else policy).get("suppliesTo") or {}
    downstream = False
    for clause in _clauses(text):
        for origin in _forms(origin_name, aliases):
            if _template_matches(entry, clause, None, origin):
                return UPSTREAM_EXPLICIT
            if _template_matches(entry, clause, origin, None):
                downstream = True
    return DOWNSTREAM_EXPLICIT if downstream else DIRECTION_AMBIGUOUS


def render_relation_policy(policy: dict[str, Any], relations: list[str]) -> str:
    """Render the active policy, not a second hard-coded copy, for model entailment."""
    output = []
    for relation in sorted(set(relations)):
        entry = policy.get(relation) or {}
        output.append(f"{relation}: {entry.get('description', '')}")
        for key in ("positive", "insufficient", "ambiguous", "direction_templates_supplier_first",
                    "direction_templates_customer_first"):
            if entry.get(key):
                output.append(f"  {key}: " + "; ".join(entry[key]))
    return "\n".join(output)
