"""Cheap conservative type plausibility; no model-name blacklist or web knowledge."""
from __future__ import annotations

import re

from ..names import normalization_key

# Branded alphanumeric model identifiers: BMW i7 / iX3, Rivian R1S.
_MODEL_CODE = re.compile(r"^[A-Za-z][\w&.-]*(?:\s+[A-Za-z][\w&.-]*){0,3}\s+[A-Za-z]{1,3}\d{1,3}[A-Za-z]{0,2}$")
_MODEL_WORD = re.compile(r"\bmodel\s+[A-Za-z0-9-]+$", re.I)
_FACILITY_WORDS = re.compile(r"\b(?:plant|factory|facility|campus|smelter|refinery|warehouse)\b", re.I)


def company_type_hint(name: str) -> tuple[str | None, str | None]:
    """Return a safe repair, or an uncertain Company reason to reject downstream.

    Only names with explicit model structure are repaired. Facilities have no
    general-purpose ontology type; reject their Company role rather than invent one.
    """
    name = " ".join((name or "").split()).strip()
    if _MODEL_CODE.fullmatch(name) or (_MODEL_WORD.search(name) and len(name.split()) >= 3):
        return "Product", "branded_model_structure"
    if _FACILITY_WORDS.search(normalization_key(name)):
        return None, "facility_not_company"
    return None, None
