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
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


def atomic_write_json(
    path: Path,
    data: Mapping[str, Any],
    *,
    indent: int = 2,
    encoding: str = "utf-8",
) -> None:
    """
    Write a JSON object atomically to disk.

    A temporary file is created in the same directory as the target file,
    written to in full, flushed + fsynced, and then atomically renamed to
    the final path via os.replace.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=path.parent,
            encoding=encoding,
        ) as tmp:
            tmp_path = Path(tmp.name)
            json.dump(data, tmp, indent=indent, ensure_ascii=False, default=str)
            tmp.flush()
            os.fsync(tmp.fileno())

        os.replace(tmp_path, path)

        # Optional: fsync directory entry for extra durability on POSIX
        # (Windows doesn't support fsync on directories)
        try:
            dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
        except (AttributeError, NotImplementedError, OSError):
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    finally:
        # If something failed before os.replace, clean up temp file
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass



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

