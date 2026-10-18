"""Structured supplier-list / table extraction (deterministic, evidence-grounded).

Detects customer/origin headings with named supplier rows. Does not invent a
customer from model memory — the expansion origin must appear in the document.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ..names import canonical_id, normalization_key
from .validate import is_generic_company_phrase

_LIST_HEADINGS = (
    r"key\s+suppliers?",
    r"main\s+suppliers?",
    r"major\s+suppliers?",
    r"top\s+suppliers?",
    r"list\s+of\s+suppliers?",
    r"suppliers?\s+to",
    r"supplier\s+list",
    r"supplier\s+awards?",
    r"awarded\s+suppliers?",
    r"preferred\s+suppliers?",
    r"strategic\s+suppliers?",
)

_BULLET = re.compile(
    r"^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$",
    re.MULTILINE,
)


def _heading_customer(block: str, origin_name: str, aliases: Optional[list[str]] = None) -> Optional[str]:
    """Return the customer name if the heading names the origin (or alias)."""
    names = [origin_name, *(aliases or [])]
    lowered = block.casefold()
    for name in names:
        key = normalization_key(name)
        if key and key in normalization_key(block):
            return name
        if name and name.casefold() in lowered:
            return name
    # Heading like "Key suppliers:" without customer — caller must supply origin
    # only when the surrounding document already grounds that customer.
    return None


def extract_supplier_list_facts(
    text: str,
    *,
    doc_id: str,
    origin_name: str,
    origin_id: Optional[str] = None,
    aliases: Optional[list[str]] = None,
    source_url: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Yield candidate suppliesTo facts from structured lists.

    Customer must be grounded: either the list heading names the origin, or a
    nearby heading does. Never invent the customer.
    """
    if not text or not origin_name:
        return []
    facts: list[dict[str, Any]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        heading_match = None
        for pattern in _LIST_HEADINGS:
            if re.search(pattern, line, re.I):
                heading_match = line
                break
        if not heading_match:
            i += 1
            continue
        # Window: heading + following lines until blank/blankish break.
        window_lines = [heading_match]
        j = i + 1
        while j < len(lines) and j < i + 40:
            nxt = lines[j]
            if not nxt.strip():
                if len(window_lines) > 1:
                    break
                j += 1
                continue
            if re.search(r"^[A-Z][A-Za-z ].{8,}:\s*$", nxt) and not _BULLET.match(nxt):
                break
            window_lines.append(nxt)
            j += 1
        window = "\n".join(window_lines)
        customer = _heading_customer(heading_match, origin_name, aliases)
        if customer is None:
            # Require origin mention in the local window before attributing the list.
            blob = normalization_key(window)
            if not any(normalization_key(n) in blob for n in [origin_name, *(aliases or [])] if n):
                i = j
                continue
            customer = origin_name
        customer_id = origin_id or canonical_id("Company", customer)
        for match in _BULLET.finditer(window):
            raw = match.group(1).strip()
            # Truncate trailing clause noise.
            supplier = re.split(r"[–—:;,(]", raw, maxsplit=1)[0].strip()
            supplier = re.sub(r"\s+", " ", supplier)
            if len(supplier) < 2 or len(supplier) > 80:
                continue
            if is_generic_company_phrase(supplier):
                continue
            if normalization_key(supplier) == normalization_key(customer):
                continue
            evidence = f"{heading_match.strip()} {raw}".strip()
            if evidence not in text and raw not in text:
                continue
            facts.append({
                "doc_id": doc_id,
                "chunk_index": -1,
                "subject_id": canonical_id("Company", supplier),
                "subject_name": supplier,
                "subject_type": "Company",
                "subject_mention": supplier,
                "relation": "suppliesTo",
                "object_id": customer_id,
                "object_name": customer,
                "object_type": "Company",
                "object_mention": customer,
                "evidence": evidence if evidence in text else raw,
                "source_url": source_url,
                "extraction_path": "structured_supplier_list",
            })
        i = j
    return facts
