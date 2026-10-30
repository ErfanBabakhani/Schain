"""Small atomic writes for expensive intermediate results."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: Any) -> None:
    """Replace a checkpoint only after the complete JSON is flushed to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_parquet(frame: Any, path: Path) -> None:
    """Replace a Parquet artifact only after the complete file is durable.

    A killed Colab kernel must not truncate the previous complete facts/history file.
    The temporary file is on the same filesystem so replacement is atomic.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    replaced = False
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
        frame.to_parquet(temporary, index=False)
        try:
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
        except OSError:
            # Some Windows environments cannot fsync this descriptor; the file is
            # still complete on disk before the atomic replace.
            pass
        os.replace(temporary, path)
        replaced = True
    finally:
        if temporary is not None and not replaced:
            temporary.unlink(missing_ok=True)
