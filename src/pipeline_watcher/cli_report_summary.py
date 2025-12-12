#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from pipeline_watcher import PipelineReport  # adjust import as needed


# ------------------------
# Loading
# ------------------------

def load_pipeline_report(path: Path | str) -> PipelineReport:
    """
    Temporary, strict-ish loader.
    You said you've changed it to this; keeping it verbatim for now.
    """
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    return PipelineReport.model_validate(data)


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

# Each row is a dict; possible keys:
#   type, parent, id, name, label, status, succeeded, ms, review

def _status_str(obj: Any) -> str:
    status = getattr(obj, "status", None)
    if hasattr(status, "name"):
        return status.name
    return str(status) if status is not None else ""


def _duration_ms_str(obj: Any) -> str:
    dur = getattr(obj, "duration_ms", None)
    return f"{dur:.0f}" if dur is not None else ""


def _file_name(f: Any) -> str:
    path = getattr(f, "path", None)
    if path:
        try:
            return Path(path).name
        except TypeError:
            return str(path)
    file_id = getattr(f, "file_id", None)
    if file_id:
        return str(file_id)
    name = getattr(f, "name", None)
    if name:
        return str(name)
    return "<file>"


def _file_review_str(f: Any) -> str:
    rhr = getattr(f, "requires_human_review", False)
    if not rhr:
        return ""
    reason = getattr(f, "human_review_reason", "") or ""
    return f"YES: {reason}" if reason else "YES"


def _step_review_str(st: Any) -> str:
    review = getattr(st, "review", None)
    if review is None:
        return ""
    flagged = getattr(review, "flagged", None)
    if not flagged:
        return ""
    reason = getattr(review, "reason", "") or ""
    return f"YES: {reason}" if reason else "YES"


def build_rows(report: PipelineReport) -> List[Dict[str, str]]:
    """
    Build a list of dict rows.

    Keys:
      - type:        pipeline-step | file | file-step
      - parent:      pipeline | <file name> | "-"
      - id:          step id (if any)
      - name:        file name or pipeline kind
      - label:       step label
      - status:      Status enum name/string
      - succeeded:   "True"/"False"
      - ms:          duration in milliseconds (string)
      - review:      HITL review info, if any
    """
    rows: List[Dict[str, str]] = []

    # ---- pipeline-level steps ----
    for st in report.steps or []:
        rows.append({
            "type": "pipeline-step",
            "parent": "pipeline",
            "id": st.id or "",
            "name": report.label or report.kind or "pipeline",
            "label": st.label or "",
            "status": _status_str(st),
            "succeeded": str(st.succeeded),
            "ms": _duration_ms_str(st),
            "review": _step_review_str(st),
        })

    # ---- files + file steps ----
    for f in report.files or []:
        fname = _file_name(f)

        # File row
        rows.append({
            "type": "file",
            "parent": "-",
            "id": getattr(f, "file_id", "") or "",
            "name": fname,
            "label": f.label,
            "status": _status_str(f),
            "succeeded": str(f.succeeded),
            "ms": _duration_ms_str(f),
            "review": _file_review_str(f),
        })

        # Steps within file
        for st in f.steps or []:
            rows.append({
                "type": "file-step",
                "parent": fname,
                "id": st.id or "",
                "name": fname,
                "label": st.label or "",
                "status": _status_str(st),
                "succeeded": str(st.succeeded),
                "ms": _duration_ms_str(st),
                "review": _step_review_str(st),
            })

    return rows


# ------------------------
# Tracebacks
# ------------------------

def collect_tracebacks(report: PipelineReport) -> List[tuple[str, str, str]]:
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
        if not tb:
            return None
        return str(tb)

    tbs: List[tuple[str, str, str]] = []

    # Files + file steps
    for f in report.files or []:
        fname = _file_name(f)

        tb = tb_for(f)
        if tb:
            tbs.append(("file", fname, tb))

        for st in f.steps or []:
            ident = st.id or st.label or "<unnamed>"
            tb = tb_for(st)
            if tb:
                tbs.append(("file-step", f"{fname}:{ident}", tb))

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
    # Kind is e.g. "validation" | "process" | "test"
    kind = (report.kind or "pipeline").capitalize()

    # Use the new roll-up status field (computed_field on PipelineReport)
    status_str = _status_str(report)

    total_steps, total_files = count_files_and_steps(report)

    print(f"{kind} status: {status_str}")
    print(f"Total number of steps: {total_steps}")
    print(f"Total number of files: {total_files}")
    print()



# Map column keys → human-friendly header labels
COLUMN_HEADERS = {
    "type": "Type",
    "parent": "Parent",
    "id": "ID",
    "name": "Name",
    "label": "Label",
    "status": "Status",
    "succeeded": "Succeeded",
    "ms": "ms",
    "review": "Review",
}


def print_table(rows: List[Dict[str, str]], cols_to_show: List[str]) -> None:
    if not rows:
        print("No steps or files recorded.")
        print()
        return

    # Only use supported keys
    cols = [c for c in cols_to_show if c in COLUMN_HEADERS]
    if not cols:
        print("No columns selected to display (cols_to_show was empty).")
        print()
        return

    # Build a "view" of each row with only selected keys
    projected_rows = [
        {k: (row.get(k, "") or "") for k in cols}
        for row in rows
    ]

    # Prepare headers and width calculation
    headers = [COLUMN_HEADERS[k] for k in cols]
    all_rows_for_width = [headers] + [
        [r[k] for k in cols] for r in projected_rows
    ]

    col_widths = [
        max(len(str(row[i])) for row in all_rows_for_width)
        for i in range(len(cols))
    ]

    def fmt_vals(values: List[str]) -> str:
        return "  ".join(
            str(values[i]).ljust(col_widths[i])
            for i in range(len(cols))
        )

    # Print header
    print(fmt_vals(headers))
    print("  ".join("-" * w for w in col_widths))

    # Print rows
    for r in projected_rows:
        values = [r[k] for k in cols]
        print(fmt_vals(values))

    print()


def print_tracebacks(tracebacks: List[tuple[str, str, str]]) -> None:
    if not tracebacks:
        return

    print("Tracebacks:")
    print()
    has_tracebacks = False
    for scope, ident, tb in tracebacks:
        has_tracebacks = True
        print(f"[{scope} {ident}]")
        print(tb.rstrip())
        print()

    if has_tracebacks:
        print('These tracebacks are from the pipeline report.')

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

    # Column toggles (False by default as requested)
    parser.add_argument("--show-type", action="store_true", help="Show the row type.")
    parser.add_argument("--show-parent", action="store_true", help="Show the parent (e.g. file for file-steps).")
    parser.add_argument("--show-id", action="store_true", help="Show the step/file ID.")
    parser.add_argument("--show-name", action="store_true", help="Show the name (e.g. file name or pipeline label).")
    parser.add_argument("--show-label", action="store_true", help="Show the step label.")
    parser.add_argument("--show-status", action="store_true", help="Show status (PENDING/RUNNING/SUCCEEDED/etc).")
    parser.add_argument("--show-succeeded", action="store_true", help="Show succeeded boolean.")
    parser.add_argument("--show-ms", action="store_true", help="Show duration in milliseconds.")
    parser.add_argument("--show-review", action="store_true", help="Show review/HITL info.")

    args = parser.parse_args(argv)

    report = load_pipeline_report(args.path)
    print_summary(report)

    # Decide which columns to show
    # If the user specified any --show-* flags, use only those.
    # Otherwise, fall back to a sensible default set.
    flag_to_col = {
        "show_type": "type",
        "show_parent": "parent",
        "show_id": "id",
        "show_name": "name",
        "show_label": "label",
        "show_status": "status",
        "show_succeeded": "succeeded",
        "show_ms": "ms",
        "show_review": "review",
    }

    selected_cols = [
        col for attr, col in flag_to_col.items()
        if getattr(args, attr)
    ]

    if not selected_cols:
        # Default columns when no --show-* flags are used
        selected_cols = ["type", "name", "label", "status", "succeeded"]

    rows = build_rows(report)
    print_table(rows, selected_cols)

    tbs = collect_tracebacks(report)
    print_tracebacks(tbs)
    print("\n\npipeline-watcher summary tool completed successfully.")


if __name__ == "__main__":
    main()
