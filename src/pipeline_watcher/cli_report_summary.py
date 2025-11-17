#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, List, Tuple

from pipeline_watcher import PipelineReport  # adjust import if needed


# ------------------------
# Loading
# ------------------------

def load_pipeline_report(path: Path) -> PipelineReport:
    text = path.read_text(encoding="utf-8")
    return PipelineReport.model_validate_json(text)


# ------------------------
# Counting
# ------------------------

def count_files_and_steps(report: PipelineReport) -> tuple[int, int]:
    files = report.files or []
    pipeline_steps = report.steps or []
    num_files = len(files)
    num_steps = len(pipeline_steps) + sum(len(f.steps or []) for f in files)
    return num_steps, num_files


# ------------------------
# Row building helpers
# ------------------------

# Columns: Type, Parent, Name/ID, Label, Status, Succeeded, ms, Review
Row = Tuple[str, str, str, str, str, str, str, str]


def _format_status(obj: Any) -> str:
    status = getattr(obj, "status", None)
    if hasattr(status, "name"):
        return status.name
    return str(status) if status is not None else ""


def _format_duration(obj: Any) -> str:
    dur = getattr(obj, "duration_ms", None)
    return f"{dur:.0f}" if dur is not None else ""


def _format_file_name(f: Any) -> str:
    path = getattr(f, "path", None)
    if path:
        try:
            return str(Path(path).name)
        except TypeError:
            return str(path)
    file_id = getattr(f, "file_id", None)
    if file_id:
        return str(file_id)
    name = getattr(f, "name", None)
    if name:
        return str(name)
    return "<file>"


def _format_file_review(f: Any) -> str:
    # FileReport has computed properties:
    #   requires_human_review (bool)
    #   human_review_reason (str)
    rhr = getattr(f, "requires_human_review", False)
    if not rhr:
        return ""
    reason = getattr(f, "human_review_reason", "") or ""
    return f"YES: {reason}" if reason else "YES"


def _format_step_review(st: Any) -> str:
    review = getattr(st, "review", None)
    if review is None:
        return ""
    flagged = getattr(review, "flagged", None)
    if not flagged:
        return ""
    reason = getattr(review, "reason", "") or ""
    return f"YES: {reason}" if reason else "YES"


def build_rows(report: PipelineReport) -> List[Row]:
    rows: List[Row] = []

    # ---------- pipeline-level steps ----------
    for st in report.steps or []:
        rows.append((
            "pipeline-step",
            "pipeline",
            st.id or "",
            st.label or "",
            _format_status(st),
            str(st.succeeded),
            _format_duration(st),
            _format_step_review(st),
        ))

    # ---------- files + file steps ----------
    for f in report.files or []:
        file_name = _format_file_name(f)

        # File row itself
        rows.append((
            "file",
            "-",
            file_name,
            "",
            _format_status(f),
            str(f.succeeded),
            _format_duration(f),
            _format_file_review(f),
        ))

        # File-level steps
        for st in f.steps or []:
            rows.append((
                "file-step",
                file_name,
                st.id or "",
                st.label or "",
                _format_status(st),
                str(st.succeeded),
                _format_duration(st),
                _format_step_review(st),
            ))

    return rows


# ------------------------
# Tracebacks
# ------------------------

def collect_tracebacks(report: PipelineReport) -> List[Tuple[str, str, str]]:
    """
    Collect (scope, ident, traceback_text) from:
      - files
      - file-level steps
      - pipeline-level steps
    """

    def tb_for(obj: Any) -> str | None:
        meta = getattr(obj, "metadata", {}) or {}
        if not isinstance(meta, dict):
            return None
        tb = meta.get("traceback")
        if tb:
            return str(tb)
        return None

    tbs: List[Tuple[str, str, str]] = []

    # Files and file steps
    for f in report.files or []:
        file_name = _format_file_name(f)

        tb = tb_for(f)
        if tb:
            tbs.append(("file", file_name, tb))

        for st in f.steps or []:
            ident = st.id or st.label or "<unnamed>"
            tb = tb_for(st)
            if tb:
                tbs.append(("file-step", f"{file_name}:{ident}", tb))

    # Pipeline-level steps
    for st in report.steps or []:
        ident = st.id or st.label or "<unnamed>"
        tb = tb_for(st)
        if tb:
            tbs.append(("pipeline-step", ident, tb))

    return tbs


# ------------------------
# Printing
# ------------------------

def print_summary(report: PipelineReport) -> None:
    kind = report.kind.capitalize()
    succeeded_str = str(getattr(report, "succeeded", None))
    total_steps, total_files = count_files_and_steps(report)

    print(f"{kind} {succeeded_str}")
    print(f"Total number of steps: {total_steps}")
    print(f"Total number of files: {total_files}")
    print()


def print_table(rows: List[Row]) -> None:
    if not rows:
        print("No steps or files recorded.")
        print()
        return

    headers = ("Type", "Parent", "ID/Name", "Label", "Status", "Succeeded", "ms", "Review")
    all_rows = [headers] + rows

    col_widths = [
        max(len(str(row[i])) for row in all_rows)
        for i in range(len(headers))
    ]

    def fmt(row: Iterable[str]) -> str:
        return "  ".join(
            str(col).ljust(col_widths[i])
            for i, col in enumerate(row)
        )

    print(fmt(headers))
    print("  ".join("-" * w for w in col_widths))

    for r in rows:
        print(fmt(r))

    print()


def print_tracebacks(tracebacks: List[Tuple[str, str, str]]) -> None:
    if not tracebacks:
        return

    print("Tracebacks:")
    print()

    for scope, ident, tb in tracebacks:
        print(f"[{scope} {ident}]")
        print(tb.rstrip())
        print()


# ------------------------
# CLI entrypoint
# ------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a PipelineReport JSON file."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to PipelineReport JSON produced by pipeline-watcher.",
    )
    args = parser.parse_args(argv)

    report = load_pipeline_report(args.path)

    print_summary(report)

    rows = build_rows(report)
    print_table(rows)

    tbs = collect_tracebacks(report)
    print_tracebacks(tbs)


if __name__ == "__main__":
    main()
