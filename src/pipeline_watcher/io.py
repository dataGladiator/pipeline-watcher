"""
I/O utilities for pipeline-watcher.

This module provides lightweight, atomic-safe JSON writing helpers used by
the watcher to persist pipeline reports and other structured artifacts.
Writes are performed atomically by first writing to a temporary file in the
target directory and then renaming it into place, ensuring that partially
written or corrupted files are never observed by readers.

Functions
---------
atomic_write_json(path, data, *, indent=2, encoding='utf-8')
    Write a JSON object atomically to disk.
dump_report(path, report)
    Serialize a Pydantic model (with `.model_dump_json()`) to JSON atomically.
"""

from __future__ import annotations
from pathlib import Path
import json
import os
import tempfile


def atomic_write_json(
    path: Path,
    data: dict,
    *,
    indent: int = 2,
    encoding: str = "utf-8"
) -> None:
    """
    Write a JSON object atomically to disk.

    A temporary file is created in the same directory as the target file,
    written to in full, and then atomically renamed to the final path.
    This guarantees that the output file is either the **old complete file**
    or the **new complete file**, never a partial write.

    Parameters
    ----------
    path : pathlib.Path
        Destination path for the JSON file. Parent directories are created
        automatically if they do not exist.
    data : dict
        Python dictionary (or JSON-serializable mapping) to write.
    indent : int, optional
        Indentation level for the output JSON, by default 2.
    encoding : str, optional
        File encoding for the output, by default ``'utf-8'``.

    Notes
    -----
    - The temporary file is created in the same directory as `path` to ensure
      atomicity across filesystems.
    - Existing files at `path` are replaced without truncation races.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, encoding=encoding
    ) as tmp:
        json.dump(data, tmp, indent=indent, ensure_ascii=False, default=str)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def dump_report(path: Path, report) -> None:
    """
    Serialize a Pydantic model to JSON (atomically).

    This function serializes any Pydantic v2 model exposing
    :meth:`model_dump_json` into a JSON file using
    :func:`atomic_write_json` for safe, atomic replacement.

    Parameters
    ----------
    path : pathlib.Path
        Destination file path for the serialized report.
    report : pydantic.BaseModel
        Pydantic model instance to serialize. Must implement
        ``.model_dump_json()``.

    Notes
    -----
    - Ensures the output file is always valid JSON (never half-written).
    - Uses ``default=str`` when serializing nested objects (e.g., datetimes,
      paths).
    """
    atomic_write_json(path, json.loads(report.model_dump_json()))

