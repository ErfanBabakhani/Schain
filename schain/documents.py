"""Load one input file into a normalized document record.

This is the front of the INGEST path. It is deliberately dull: read the file,
normalize the text, compute the identifiers that later deduplication depends on,
and append to the raw/normalized data layers.

Identifiers (PROJECT_DECISIONS.md section 9):

* ``content_hash`` - sha256 of the normalized text; catches the same text arriving twice.
* ``source_url``   - normalized URL; catches the same location arriving twice.
* ``doc_id``       - short prefix of the content hash, used as the join key everywhere else.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".text"}
DOC_ID_LENGTH = 16
_TRACKING_PREFIXES = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid")


def normalize_text(raw: str) -> str:
    """Unicode-normalize, drop a BOM, and make line endings and trailing space uniform.

    Text normalization runs before hashing, so trivial formatting differences do not
    produce two different documents.
    """
    text = unicodedata.normalize("NFC", raw.replace("﻿", ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def normalize_url(url: Optional[str]) -> Optional[str]:
    """Lower-case scheme/host, drop the fragment and tracking parameters, tidy the path."""
    if not url:
        return None
    parts = urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith(_TRACKING_PREFIXES)
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), "")
    )


def content_hash(text: str) -> str:
    """sha256 of the normalized text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_document_text(path: Path) -> str:
    """Read a .txt/.md file directly, or a PDF through the neo4j-graphrag PdfLoader."""
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        from neo4j_graphrag.components.pdf_loader import PdfLoader

        loaded = asyncio.run(PdfLoader().run(filepath=path))
        return loaded.text
    raise ValueError(
        f"unsupported input type {suffix!r}; expected one of "
        f"{sorted(TEXT_SUFFIXES)} or .pdf"
    )


def make_document(
    text: str,
    source_url: Optional[str] = None,
    title: Optional[str] = None,
    source_path: Optional[str] = None,
    content_source: str = "file",
) -> dict[str, Any]:
    """Build the normalized document record from raw text.

    Used both by INGEST (text read from a file) and by EXPAND (text from retrieval).

    ``content_source`` records where the text came from - "file" for INGEST,
    "full_text" or "title_snippet" for retrieved documents - so it is visible later
    whether a fact was extracted from an article body or only from a headline.
    """
    text = normalize_text(text)
    if not text:
        raise ValueError("document is empty after normalization")

    digest = content_hash(text)
    return {
        "doc_id": digest[:DOC_ID_LENGTH],
        "content_hash": digest,
        "source_path": source_path,
        "source_url": normalize_url(source_url),
        "title": title or (Path(source_path).stem if source_path else ""),
        "text": text,
        "char_count": len(text),
        "content_source": content_source,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def load_document(
    path: Path,
    source_url: Optional[str] = None,
    title: Optional[str] = None,
) -> dict[str, Any]:
    """Build the normalized document record for one input file."""
    path = Path(path)
    record = make_document(
        read_document_text(path),
        source_url=source_url,
        title=title or path.stem,
        source_path=str(path),
    )
    logger.info(
        "loaded document %s (%s, %d chars)", record["doc_id"], path.name, record["char_count"]
    )
    return record


def iter_input_files(target: Path, patterns: list[str]) -> list[Path]:
    """Input files for INGEST: the file itself, or every match inside a directory."""
    target = Path(target)
    if target.is_file():
        return [target]
    if not target.exists():
        raise FileNotFoundError(f"input path does not exist: {target}")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(target.rglob(pattern)))
    return sorted(set(files))


def append_raw_jsonl(record: dict[str, Any], path: Path) -> None:
    """Append the record to the raw JSONL layer, skipping a doc_id already present."""
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip() and json.loads(line).get("doc_id") == record["doc_id"]:
                    logger.info("doc_id %s already in %s", record["doc_id"], path.name)
                    return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def upsert_normalized_parquet(record: dict[str, Any], path: Path) -> None:
    """Write the record to the normalized Parquet layer, replacing any earlier copy."""
    frame = pd.DataFrame([record])
    if path.exists():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
        frame = frame.drop_duplicates(subset="doc_id", keep="last").reset_index(drop=True)
    frame.to_parquet(path, index=False)
