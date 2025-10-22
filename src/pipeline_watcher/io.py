from __future__ import annotations
from pathlib import Path
import json, os, tempfile
from .clocks import now_utc  # exported for convenience

def atomic_write_json(path: Path, data: dict, *, indent: int = 2, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding=encoding) as tmp:
        json.dump(data, tmp, indent=indent, ensure_ascii=False, default=str)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)

def dump_report(path: Path, report) -> None:
    """Serialize any pydantic model with .model_dump_json() to JSON (atomically)."""
    atomic_write_json(path, json.loads(report.model_dump_json()))

