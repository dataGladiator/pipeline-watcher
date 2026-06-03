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

import errno
import json
import os
import secrets
from pathlib import Path
from typing import Any, Mapping, TextIO


def _create_replace_temp_file(
    directory: Path,
    *,
    encoding: str,
) -> tuple[TextIO, Path]:
    """
    Create a uniquely named temporary file in directory for atomic
    replacement.

    Unlike tempfile.NamedTemporaryFile, this requests mode 0o666 so the OS
    applies the process umask exactly as it would for normal file creation via
    open(path, "w"). The O_EXCL flag prevents accidental reuse of an existing
    path.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL

    for _ in range(100):
        tmp_path = directory / f".tmp-{secrets.token_hex(16)}"
        try:
            fd = os.open(tmp_path, flags, 0o666)
        except FileExistsError:
            continue

        return os.fdopen(fd, "w", encoding=encoding), tmp_path

    raise FileExistsError(
        errno.EEXIST,
        "could not create a unique temporary file",
        str(directory),
    )


def _fsync_parent_directory(path: Path) -> None:
    """
    Best-effort fsync of the parent directory.

    This improves durability of the rename on POSIX filesystems. Some
    platforms
    do not support opening or fsyncing directories, so failures are ignored.
    """
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    except (AttributeError, NotImplementedError, OSError):
        return

    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


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
    written fully, flushed, fsynced, and then atomically renamed to the final
    path via os.replace.

    For newly-created output files, permissions match normal open(path, "w")
    behavior: the file is created with requested mode 0o666, filtered by the
    process umask. This avoids tempfile.NamedTemporaryFile's usual private
    0o600 permissions, which can break shared-filesystem readers.

    Existing target files are replaced by a new inode. Permission bits are not
    copied from an existing target file; this function intentionally uses
    normal
    new-file creation permissions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path: Path | None = None

    try:
        tmp, tmp_path = _create_replace_temp_file(
            path.parent,
            encoding=encoding,
        )

        with tmp:
            json.dump(data, tmp, indent=indent, ensure_ascii=False,
                      default=str)
            tmp.flush()
            os.fsync(tmp.fileno())

        os.replace(tmp_path, path)
        _fsync_parent_directory(path)

    finally:
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

