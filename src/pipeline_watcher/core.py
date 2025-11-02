from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Literal, Mapping
from abc import ABC, abstractmethod
from contextlib import contextmanager
import time, traceback, contextvars, re
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, computed_field
from pathlib import Path

from .clocks import now_utc as _now


SCHEMA_VERSION = "v2"


def _norm_text(s: str | None) -> str:
    return "" if s is None else " ".join(str(s).strip().split()).lower()


def _norm_key(x) -> str:
    """Accepts str/Path/Any with .name; returns a normalized key."""
    if isinstance(x, Path):
        return _norm_text(str(x))
    try:
        return _norm_text(str(x))
    except Exception:
        return _norm_text(repr(x))


def _file_keys(fr: "FileReport") -> set[str]:
    """All comparable keys for a FileReport: id, name, path, and basename."""
    keys = set()
    if fr.file_id:
        keys.add(_norm_key(fr.file_id))
    if fr.name:
        keys.add(_norm_key(fr.name))
    if fr.path:
        p = Path(fr.path)
        keys.add(_norm_key(str(p)))
        keys.add(_norm_key(p.name))
    return keys


def _slugify(s: str) -> str:
    s = re.sub(r"\s+", "-", s.strip().lower())
    s = re.sub(r"[^a-z0-9\-]+", "", s)
    return s


class StepStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED  = "FAILED"
    SKIPPED = "SKIPPED"


_TERMINAL = {StepStatus.SUCCESS, StepStatus.FAILED, StepStatus.SKIPPED}


class Check(BaseModel):
    name: str
    ok: bool
    detail: Optional[str] = None


class ReviewFlag(BaseModel):
    flagged: bool = False
    reason: Optional[str] = None


class ReportBase(BaseModel, ABC):
    status: StepStatus = StepStatus.PENDING
    percent: int = 0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    notes: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    review: ReviewFlag = Field(default_factory=ReviewFlag)
    report_version: str = SCHEMA_VERSION

    def start(self) -> "ReportBase":
        self.status = StepStatus.RUNNING
        if not self.started_at:
            self.started_at = _now()
        return self

    def succeed(self) -> "ReportBase":
        self.status = StepStatus.SUCCESS
        self.percent = 100
        self.finished_at = _now()
        return self

    def fail(self, message: Optional[str] = None) -> "ReportBase":
        self.status = StepStatus.FAILED
        if message:
            self.errors.append(message)
        self.finished_at = _now()
        return self

    def skip(self, reason: Optional[str] = None) -> "ReportBase":
        self.status = StepStatus.SKIPPED
        if reason:
            self.notes.append(f"Skipped: {reason}")
        self.finished_at = _now()
        return self

    def request_review(self, reason: str | None = None) -> "ReportBase":
        self.review = ReviewFlag(flagged=True, reason=reason)
        return self

    def clear_review(self) -> "ReportBase":
        self.review = ReviewFlag()
        return self

    def note(self, msg: str) -> "ReportBase":
        self.notes.append(msg)
        return self

    def warn(self, msg: str) -> "ReportBase":
        self.warnings.append(msg)
        return self

    def error(self, msg: str) -> "ReportBase":
        self.errors.append(msg)
        return self

    @property
    def requires_human_review(self) -> bool:
        return bool(self.review.flagged)

    @property
    @abstractmethod
    def ok(self) -> bool:
        """Concrete subclasses decide what 'ok' means."""

    def end(self) -> "ReportBase":
        if self.status in (StepStatus.SUCCESS, StepStatus.FAILED, StepStatus.SKIPPED):
            if not self.finished_at:
                self.finished_at = _now()
            return self
        return self.succeed() if self.ok else self.fail("One or more checks failed" if self.errors or hasattr(self, "checks") else "Step failed")


class StepReport(ReportBase):
    """
    One unit of work (e.g., 'parse', 'analyze'). Designed to be JSON-serialized.
    """
    id: str
    label: Optional[str] = None
    checks: List[Check] = Field(default_factory=list)

    @classmethod
    def begin(cls, id: str, *, label: str | None = None, **meta) -> "StepReport":
        return cls(id=id, label=label, metadata=dict(meta)).start()

    @property
    def ok(self) -> bool:
        if self.status == StepStatus.FAILED or self.errors:
            return False
        if self.status == StepStatus.SUCCESS:
            return True
        return all(c.ok for c in self.checks) if self.checks else True

    def add_check(self, name: str, ok: bool, detail: Optional[str] = None) -> None:
        self.checks.append(Check(name=name, ok=ok, detail=detail))


class FileReport(ReportBase):
    """
    Per-document aggregation with its own ordered sequence of steps.
    """
    file_id: str
    path: str | None = None
    name: str | None = None
    size_bytes: int | None = None
    mime_type: str | None = None
    steps: List[StepReport] = Field(default_factory=list)
    review: ReviewFlag = Field(default_factory=ReviewFlag)

    @classmethod
    def begin(cls, file_id: str, **meta) -> "FileReport":
        return cls(file_id=file_id, **meta).start()

    def append_step(self, step: StepReport) -> "FileReport":
        """Finalize the step via end(), append, recompute percent, and return self (chainable)."""
        step.end()
        self.steps.append(step)
        self._recompute_percent()
        # roll-up HITL review if you added ReviewFlag earlier
        if step.review.flagged and not self.review.flagged:
            self.review = ReviewFlag(flagged=True,
                                     reason=step.review.reason or f"Step '{step.id}' requested review")
        return self

    def add_step(self, id: str, *, label: str | None = None, **meta) -> StepReport:
        """
        Convenience: construct StepReport.begin(...), append it, and return the StepReport.
        Great when you need the step object to update metadata mid-callers.
        """
        step = StepReport.begin(id, label=label, **meta)
        step.end()
        self.steps.append(step)
        self._recompute_percent()
        return step

    def last_step(self) -> StepReport | None:
        return self.steps[-1] if self.steps else None

    def end(self) -> "FileReport":
        if self.status in (StepStatus.SUCCESS, StepStatus.FAILED, StepStatus.SKIPPED):
            if not self.finished_at:
                self.finished_at = _now()
            return self
        return self.succeed() if self.ok else self.fail("One or more file steps failed")

    @property
    def ok(self) -> bool:
        # Failure wins
        if self.status == StepStatus.FAILED or self.errors:
            return False
        # Explicit success wins
        if self.status == StepStatus.SUCCESS:
            return True
        # Otherwise roll up from steps (if any)
        return all(s.ok for s in self.steps) if self.steps else True

    def _recompute_percent(self) -> None:
        if self.steps:
            self.percent = int(round(sum(s.percent for s in self.steps) / len(self.steps)))

    def _make_unique_step_id(self, label: str) -> str:
        base = _slugify(label) or "step"
        existing = {s.id for s in self.steps}
        if base not in existing:
            return base
        i = 2
        while f"{base}-{i}" in existing:
            i += 1
        return f"{base}-{i}"

    def add_completed_step(
        self,
        label: str,
        *,
        id: str | None = None,
        note: str | None = None,
        metadata: dict | None = None,
    ) -> "FileReport":
        """Create a SUCCESS step with `label`, append via append_step(), and return self."""
        sid = id or self._make_unique_step_id(label)
        step = StepReport.begin(sid, label=label)
        if metadata:
            step.metadata.update(metadata)
        if note:
            step.notes.append(note)
        step.succeed()  # terminal; append_step will end() again idempotently
        return self.append_step(step)

    def add_failed_step(
        self,
        label: str,
        *,
        id: str | None = None,
        reason: str | None = None,
        metadata: dict | None = None,
    ) -> "FileReport":
        """Create a FAILED step and append it (chainable)."""
        sid = id or self._make_unique_step_id(label)
        step = StepReport.begin(sid, label=label)
        if metadata:
            step.metadata.update(metadata)
        step.fail(reason)
        return self.append_step(step)

    def add_skipped_step(
        self,
        label: str,
        *,
        id: str | None = None,
        reason: str | None = None,
        metadata: dict | None = None,
    ) -> "FileReport":
        """Create a SKIPPED step and append it (chainable)."""
        sid = id or self._make_unique_step_id(label)
        step = StepReport.begin(sid, label=label)
        if metadata:
            step.metadata.update(metadata)
        step.skip(reason)
        return self.append_step(step)

    def add_review_step(
        self,
        label: str,
        *,
        id: str | None = None,
        reason: str | None = None,
        metadata: dict | None = None,
        mark_success: bool = True,
    ) -> "FileReport":
        """
        Create a step that flags Human-In-The-Loop review, then append.
        By default we mark it SUCCESS (common pattern: 'passed but needs review').
        """
        sid = id or self._make_unique_step_id(label)
        step = StepReport.begin(sid, label=label).request_review(reason)
        if metadata:
            step.metadata.update(metadata)
        if mark_success:
            step.succeed()
        # If your append_step rolls review up to the FileReport, that'll happen there.
        return self.append_step(step)

    def _flagged_steps(self) -> List[StepReport]:
        """Internal: steps that requested human review."""
        out = []
        for s in self.steps:
            rf = getattr(s, "review", None)
            if rf and getattr(rf, "flagged", False):
                out.append(s)
        return out

    @computed_field(return_type=bool)  # included in model_dump / JSON
    def requires_human_review(self) -> bool:
        """
        True if either:
          - the file itself is flagged for review, or
          - any contained step is flagged for review.
        """
        if self.review and self.review.flagged:
            return True
        return any(True for _ in self._flagged_steps())

    @computed_field(return_type=Optional[str])  # included in model_dump / JSON
    def human_review_reason(self) -> Optional[str]:
        """
        A compact summary of why this file needs human review.
        Includes a file-level reason (if present) and a rollup from steps.
        """
        parts: List[str] = []

        # File-level reason first (if you use it)
        if self.review and self.review.flagged:
            parts.append(f"File-level review: {self.review.reason or 'Review requested'}")

        flagged = self._flagged_steps()
        if not flagged and not parts:
            return None

        if flagged:
            count = len(flagged)
            # show up to 5 step names; add a "+N more" if needed
            def step_name(st: StepReport) -> str:
                return st.label or st.id

            names = [step_name(s) for s in flagged[:5]]
            more = count - len(names)
            if more > 0:
                names.append(f"+{more} more")

            first_reason = getattr(flagged[0].review, "reason", None) or "Review requested"
            step_word = "step" if count == 1 else f"{count} steps"
            parts.append(f"{step_word} flagged human review: {', '.join(names)}. First reason: {first_reason}.")

        return " ".join(parts) if parts else None


class PipelineReport(BaseModel):
    """
    Batch-level container with an ordered list of steps and per-file reports.
    """
    label: str
    kind: Literal["validation", "process", "test"]
    stage: str = ""
    percent: int = 0
    message: str = ""
    updated_at: datetime = Field(default_factory=_now)
    report_version: str = SCHEMA_VERSION

    steps: List[StepReport] = Field(default_factory=list)
    files: List[FileReport] = Field(default_factory=list)

    output_path: Optional[Path] = None

    def save(
        self,
        path: str | Path | None = None,
        *,
        indent: int = 2,
        ensure_dir: bool = True,
        encoding: str = "utf-8",
    ) -> None:
        """
        Persist this report as JSON directly to disk (no temp file).
        """
        target = Path(path or self.output_path or "reports/progress.json")
        if ensure_dir:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=indent), encoding=encoding)

    def set_progress(self, stage: str, percent: int, message: str = "") -> None:
        self.stage = stage
        self.percent = max(0, min(100, percent))
        self.message = message
        self.updated_at = _now()

    def append_step(self, step: StepReport) -> "PipelineReport":
        step.end()
        self.steps.append(step)
        self.updated_at = _now()
        return self

    def add_step(self, id: str, *, label: str | None = None, **meta) -> StepReport:
        step = StepReport.begin(id, label=label, **meta).end()
        self.steps.append(step)
        self.updated_at = _now()
        return step

    def append_file(self, fr: FileReport) -> "PipelineReport":
        fr.end()
        self.files.append(fr)
        self.updated_at = _now()
        return self

    def last_step(self) -> StepReport | None:
        return self.steps[-1] if self.steps else None

    def iter_steps(self, *, status: StepStatus | None = None) -> Iterable[StepReport]:
        for s in self.steps:
            if status is None or s.status == status:
                yield s

    def recompute_overall_from_steps(self) -> None:
        if not self.steps:
            return
        pct = int(round(sum(s.percent for s in self.steps) / len(self.steps)))
        self.set_progress(self.stage or "steps", pct, self.message or "")

    def files_for(self, key) -> list["FileReport"]:
        """Return all FileReports that match file_id/name/path/basename."""
        k = _norm_key(key)
        if not k:
            return []
        out = []
        for fr in self.files:
            if k in _file_keys(fr):
                out.append(fr)
        return out

    def get_file(self, key) -> "FileReport | None":
        """Return the first matching FileReport (or None)."""
        matches = self.files_for(key)
        return matches[0] if matches else None

    def file_seen(self, key) -> bool:
        """Did this file appear in the report at all?"""
        return self.get_file(key) is not None

    def file_processed(self, key, *, require_success: bool = False) -> bool:
        """
        Has the file reached a terminal state?
          - default: SUCCESS/FAILED/SKIPPED counts as processed
          - require_success=True: only SUCCESS counts
        """
        fr = self.get_file(key)
        if fr is None:
            return False
        if require_success:
            return fr.status == StepStatus.SUCCESS
        # terminal = finished state OR explicit 100% OR finished_at set
        return (
            fr.status in _TERMINAL
            or getattr(fr, "finished_at", None) is not None
            or (getattr(fr, "percent", None) == 100)
        )

    def unseen_expected(self, expected_iterable) -> list[str]:
        """
        Given filenames/ids/paths, return the ones not present in the report.
        """
        have = set()
        for fr in self.files:
            have |= _file_keys(fr)
        missing = []
        for x in expected_iterable:
            k = _norm_key(x)
            if k and k not in have:
                missing.append(str(x))
        return missing

    def table_rows_for_files_map(self, expected: Mapping[str, Any]) -> list[dict]:
        """
        Produce Django/Jinja-friendly row dicts from a mapping of
        {filename_or_id_or_path: other_properties}.

        Row fields:
          - filename: str
          - seen: bool
          - status: str ("SUCCESS"/"FAILED"/"SKIPPED"/"RUNNING"/"PENDING" or "MISSING")
          - percent: int | None
          - flagged_human_review: bool
          - human_review_reason: str ("" if none)
          - file_id: str | None
          - path: str | None
          - other: Any (the value from the mapping; often a dict for template access)
        """
        def _norm(s: str | None) -> str:
            return "" if s is None else " ".join(str(s).strip().split()).lower()

        def _keys(fr: "FileReport") -> set[str]:
            ks = set()
            if fr.file_id: ks.add(_norm(fr.file_id))
            if fr.name:    ks.add(_norm(fr.name))
            if fr.path:
                p = Path(fr.path)
                ks.add(_norm(str(p)))
                ks.add(_norm(p.name))
            return ks

        # index by all comparable keys (first match wins)
        index: dict[str, "FileReport"] = {}
        for fr in self.files:
            for k in _keys(fr):
                index.setdefault(k, fr)

        rows: list[dict] = []
        for filename, other in expected.items():
            display_name = str(filename)
            key = _norm(display_name)
            fr = index.get(key)

            if fr is None:
                rows.append({
                    "filename": display_name,
                    "seen": False,
                    "status": "MISSING",
                    "percent": None,
                    "flagged_human_review": False,
                    "human_review_reason": "",
                    "file_id": None,
                    "path": None,
                    "other": other,
                })
            else:
                flagged = getattr(fr, "requires_human_review", False)
                reason  = getattr(fr, "human_review_reason", None) or ""
                rows.append({
                    "filename": display_name,
                    "seen": True,
                    "status": fr.status,
                    "percent": getattr(fr, "percent", None),
                    "flagged_human_review": bool(flagged),
                    "human_review_reason": reason,
                    "file_id": getattr(fr, "file_id", None),
                    "path": getattr(fr, "path", None),
                    "other": other,
                })

        return rows

# --- context var to hold the "current" PipelineReport (thread/async-safe) ---
_current_pipeline_report: contextvars.ContextVar[Optional["PipelineReport"]] = contextvars.ContextVar(
    "_current_pipeline_report", default=None
)

@contextmanager
def bind_pipeline(pr: "PipelineReport"):
    """
    Bind a PipelineReport to the current context so helpers (pipeline_step)
    can auto-append to it without passing `pr` explicitly.

    Usage:
        with bind_pipeline(report):
            # any pipeline_step(...) inside uses `report` by default
            ...
    """
    token = _current_pipeline_report.set(pr)
    try:
        yield pr
    finally:
        _current_pipeline_report.reset(token)


@contextmanager
def pipeline_step(
    pr: Optional["PipelineReport"],
    id: str,
    *,
    label: str | None = None,
    banner_stage: str | None = None,
    banner_percent: int | None = None,
    banner_message: str | None = None,
    set_stage_on_enter: bool = False,
    raise_on_exception: bool = False,
    save_on_exception: bool = True,
    output_path_override: str | Path | None = None,
):
    pr = pr or _current_pipeline_report.get()
    st = StepReport.begin(id, label=label)
    t0 = time.perf_counter()

    if pr is not None and set_stage_on_enter:
        pr.set_progress(stage=banner_stage or id, percent=pr.percent, message=banner_message or pr.message)

    exc: BaseException | None = None
    try:
        yield st
        st.end()
    except BaseException as e:
        exc = e
        st.errors.append(f"{type(e).__name__}: {e}")
        st.metadata["traceback"] = traceback.format_exc()
        st.fail("Unhandled exception")
    finally:
        st.metadata["duration_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        if pr is not None:
            try:
                pr.append_step(st)
                if any(v is not None for v in (banner_stage, banner_percent, banner_message)):
                    pr.set_progress(
                        stage=banner_stage or (pr.stage or id),
                        percent=pr.percent if banner_percent is None else banner_percent,
                        message=banner_message or pr.message,
                    )
            finally:
                if exc and save_on_exception:
                    try:
                        print(f"trying to save to {output_path_override or pr.output_path}")
                        pr.save(output_path_override or pr.output_path)
                    except Exception as save_err:
                        st.warnings.append(f"save failed: {save_err!r}")

        if exc and raise_on_exception:
            raise exc


@contextmanager
def bind_pipeline(pr: "PipelineReport"):
    """
    Bind a PipelineReport to the current context so helpers can auto-append to it.
    """
    token = _current_pipeline_report.set(pr)
    try:
        yield pr
    finally:
        _current_pipeline_report.reset(token)


@contextmanager
def pipeline_file(
    pr: Optional["PipelineReport"],
    *,
    file_id: str,
    path: str | None = None,
    name: str | None = None,
    size_bytes: int | None = None,
    mime_type: str | None = None,
    metadata: dict | None = None,
    set_stage_on_enter: bool = False,
    banner_stage: str | None = None,
    banner_percent_on_exit: int | None = None,
    banner_message_on_exit: str | None = None,
    raise_on_exception: bool = False,
    save_on_exception: bool = True,
    output_path_override: str | Path | None = None
):
    pr = pr or _current_pipeline_report.get()

    fr = FileReport.begin(file_id=file_id, path=path, name=name)
    if size_bytes is not None: fr.size_bytes = size_bytes
    if mime_type  is not None: fr.mime_type  = mime_type
    if metadata:               fr.metadata.update(metadata)

    t0 = time.perf_counter()

    if pr is not None and set_stage_on_enter:
        pr.set_progress(stage=banner_stage or (name or file_id),
                        percent=pr.percent,
                        message=pr.message)

    exc: BaseException | None = None
    try:
        yield fr
        fr.succeed() # only runs if when completes successfully.
    except BaseException as e:
        exc = e
        fr.errors.append(f"{type(e).__name__}: {e}")
        fr.metadata["traceback"] = traceback.format_exc()
        fr.fail("Unhandled exception while processing file")
    finally:
        fr.metadata["duration_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        fr.end()
        if pr is not None:
            try:
                pr.append_file(fr)
                if banner_stage or banner_percent_on_exit is not None or banner_message_on_exit:
                    pr.set_progress(
                        stage=banner_stage or pr.stage,
                        percent=pr.percent if banner_percent_on_exit is None else banner_percent_on_exit,
                        message=banner_message_on_exit or pr.message,
                    )
            finally:
                if exc and save_on_exception:
                    try:
                        pr.save(output_path_override or pr.output_path)
                    except Exception as save_err:
                        fr.warnings.append(f"save failed: {save_err!r}")

        if exc and raise_on_exception:
            raise exc


@contextmanager
def file_step(file_report, id: str, *, label: str | None = None):
    st = StepReport.begin(id, label=label)
    t0 = time.perf_counter()
    try:
        yield st
        st.end()
    except BaseException as e:
        st.errors.append(f"{type(e).__name__}: {e}")
        st.metadata["traceback"] = traceback.format_exc()
        st.fail("Unhandled exception in file step")
        # By default, do not re-raise so the file continues recording
    finally:
        st.metadata["duration_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        file_report.append_step(st)
