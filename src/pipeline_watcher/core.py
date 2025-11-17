"""Core helpers and base definitions for pipeline-watcher.

This module holds lightweight utilities (normalization, slugging),
status enums, and Pydantic models used across the reporting layer.

The intent is to keep these pieces framework-agnostic and JSON-friendly.
"""
# std lib imports
from __future__ import annotations
import os
from typing import Any, Dict, Iterable, List, Optional, Literal, Mapping, Protocol, TypeVar
from abc import ABC, abstractmethod
from contextlib import contextmanager
import time, traceback, contextvars, warnings
from io import StringIO
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from enum import Enum, auto
from datetime import datetime
from pathlib import Path
from dataclasses import fields
import mimetypes

# third party import
from pydantic import BaseModel, Field, computed_field, model_validator, field_validator
from pydantic import ConfigDict, PrivateAttr

# local imports
from .clocks import now_utc as _now
from .utilities import _slugify, _file_keys, _norm_key
from .settings import WatcherSettings, use_settings


#: Schema version written to JSON artifacts.
SCHEMA_VERSION = "v2"


class LowerStrEnum(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


class Status(LowerStrEnum):
    """Lifecycle status for a unit of work."""
    PENDING = auto()     # Declared but not yet started.
    RUNNING = auto()     # In progress.
    SUCCEEDED = auto()   # Completed successfully.
    FAILED  = auto()     # Completed with an error or failing condition.
    SKIPPED = auto()     # Intentionally not executed (precondition not met, cached, etc.).

    @property
    def running(self) -> bool:
        return self is Status.RUNNING

    @property
    def pending(self) -> bool:
        return self is Status.PENDING

    @property
    def succeeded(self) -> bool:
        return self is Status.SUCCEEDED

    @property
    def failed(self) -> bool:
        return self is Status.FAILED

    @property
    def skipped(self) -> bool:
        return self is Status.SKIPPED

    @property
    def terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.SKIPPED}


class Check(BaseModel):
    """Small data class to hold the result of a check.

    Parameters
    ----------
    name: str
        Identifier for the check (e.g., ``"manifest_present"``).
    ok: bool
        Result of the check.
    detail: str
        Optional human-readable context (e.g., counts, reason).

    Examples
    --------
    >>> Check(name="ids_unique", ok=False, detail="3 duplicates")
    Check(name='ids_unique', ok=False, detail='3 duplicates')
    """
    name: str
    ok: bool
    detail: Optional[str] = None


class ReviewFlag(BaseModel):
    """Small data class to hold a Human-in-the-loop (HITL) review request.

    Parameters
    ----------
    flagged: bool
        Whether review is requested.
    reason: str
        Optional short reason shown in the UI.
    """

    flagged: bool = False
    reason: Optional[str] = None


# Static typing help: any model that has `.review: ReviewFlag`
class HasReview(Protocol):
    review: ReviewFlag


HasReviewTV = TypeVar("HasReviewTV", bound="HasReview")

class ReviewHelpers:
    """Behavior-only mixin: no fields, no BaseModel inheritance."""
    @property
    def requires_human_review(self: HasReview) -> bool:
        """Whether the unit is flagged for human review.

        Returns
        -------
        bool
            ``True`` if :attr:`review.flagged` is set, else ``False``.
        """
        return self.review.flagged

    def request_review(self: HasReviewTV, reason: str | None = None) -> HasReviewTV:
        """Set the HITL review flag.

        Parameters
        ----------
        reason : str, optional
            Short UI-visible reason for requesting review.

        Returns
        -------
        ReportBase
            Self.
        """
        self.review = ReviewFlag(flagged=True, reason=reason)
        return self

    def clear_review(self: HasReviewTV) -> HasReviewTV:
        """Clear the HITL review flag.

        Returns
        -------
        ReportBase
            Self.
        """
        self.review = ReviewFlag()
        return self


def construct_unique_step_id_from_label(label: str, steps: list) -> str:
    """Generate a slugified, unique step id based on a label.

    If the slug already exists among current steps, appends ``-2``, ``-3``,
    etc., until unique.

    Parameters
    ----------
    label : str
        Human-readable label to slugify.
    steps: list
        List of StepReport objects.

    Returns
    -------
    str
        Unique step identifier.
    """
    base = _slugify(label) or "step"
    existing = {s.id for s in steps}
    if base not in existing:
        return base
    i = 2
    while f"{base}-{i}" in existing:
        i += 1
    return f"{base}-{i}"


def make_step_id_unique(id: str, steps: list, max_steps: int = 10_000) -> str:
    if not id:
        id = f'step-{1+len(steps)}'
    if any(s.id == id for s in steps):
        j = 1
        while any(s.id == f"{id}-{j}" for s in steps):
            j += 1
            if j > max_steps:
                raise ValueError(f"Reached maximum number of steps: {max_steps}.")
        id = f"{id}-{j}"
    return id


class PipelineReport(BaseModel):
    """Batch-level container with ordered batch steps and per-file reports.

    Designed to be JSON-serializable (Pydantic v2) and UI-friendly. Maintains
    a progress banner (``stage``, ``percent``, ``message``, ``updated_at``),
    an ordered list of batch-level [StepReport](file_step.md#StepReport) items, and a list of
    [FileReport](file_step.md#FileReport) records.

    Attributes
    ----------
    label : str
        Human-friendly batch label (e.g., project/run name).
    kind : {"validation", "process", "test"}
        Category of run for UI filtering and routing.
    stage : str
        Short machine-friendly stage label for the progress banner.
    percent : int
        Overall progress ``0..100`` (informational; not enforced).
    message : str
        Human-readable progress note for dashboards.
    updated_at : datetime
        Last update timestamp (UTC). Defaults to ``now`` on construction.
    report_version : str
        Schema version emitted to artifacts.
    steps : list[StepReport]
        Ordered sequence of batch-level steps (rare compared to file steps,
        but useful for pre/post stages like discovery/validation).
    files : list[FileReport]
        Per-file timelines collected in this batch.
    output_path : Path or None
        Default output path for :meth:`save`.

    Notes
    -----
    - The model is append-only for auditability.
    - Use :meth:`set_progress` to update the banner; it stamps ``updated_at``.
    """
    label: str
    output_path: Optional[Path] = None
    kind: Literal["validation", "process", "test"] = "process"
    stage: str = ""
    percent: int = 0
    message: str = ""
    updated_at: datetime = Field(default_factory=_now)
    report_version: str = SCHEMA_VERSION
    steps: List[StepReport] = Field(default_factory=list)
    files: List[FileReport] = Field(default_factory=list)

    def save(
        self,
        path: str | Path | None = None,
        *,
        indent: int = 2,
        ensure_dir: bool = True,
        encoding: str = "utf-8",
    ) -> None:
        """Persist the report to JSON on disk.

        Writes directly to the target (no temp/atomic swap in this helper).

        Parameters
        ----------
        path : str or Path or None, default None
            Destination path. If ``None``, uses :attr:`output_path` or
            ``reports/progress.json``.
        indent : int, default 2
            JSON indentation (passed to :meth:`BaseModel.model_dump_json`).
        ensure_dir : bool, default True
            If ``True``, creates parent directories as needed.
        encoding : str, default "utf-8"
            Text encoding for the output file.

        Examples
        --------
        >>> report = PipelineReport(...)
        >>> report.output_path = Path("reports/run-42.json")
        >>> report.save()
        """
        target = Path(path or self.output_path or "reports/progress.json")
        if ensure_dir:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=indent), encoding=encoding)

    def add_completed_step(self, label: str, *, id: str | None = None) -> StepReport:
        """Construct, finalize, append, and return a batch-level step. Ensures id uniqueness.

        Parameters
        ----------
        label : str
            Human-friendly label.
        id : str, optional
            Step identifier.
        Returns
        -------
        StepReport
            The created (and ended) step.
        """
        if not id:
            id = construct_unique_step_id_from_label(label, self.steps)
        step = StepReport.begin(label, id=id)
        self.append_step(step)
        return step

    def append_file(self, fr: FileReport) -> "PipelineReport":
        """Finalize and append a :class:`FileReport`; update ``updated_at``.

        Parameters
        ----------
        fr : FileReport
            File report to finalize via :meth:`FileReport.end` and append.

        Returns
        -------
        PipelineReport
            Self (chainable).
        """
        fr.end()
        self.files.append(fr)
        self.updated_at = _now()
        return self

    def append_step(self, step: StepReport) -> "PipelineReport":
        """Finalize and append a batch-level step; update ``updated_at``.

        Parameters
        ----------
        step : StepReport
            Step to finalize via :meth:`StepReport.end` and append.

        Returns
        -------
        PipelineReport
            Self (chainable).
        """
        step.end()
        if not step.id:
            step.id = construct_unique_step_id_from_label(step.label, self.steps)
        step.id = make_step_id_unique(step.id, self.steps)
        self.steps.append(step)
        self.updated_at = _now()
        return self

    def files_for(self, key) -> list["FileReport"]:
        """Return all file reports matching an id/name/path/basename.

        Matching is case- and whitespace-insensitive and accepts
        identifiers, filenames, full paths, and basenames.

        Parameters
        ----------
        key : Any
            A value convertible to a normalized comparison key.

        Returns
        -------
        list[FileReport]
            All matching file reports (may be empty).
        """
        k = _norm_key(key)
        if not k:
            return []
        out = []
        for fr in self.files:
            if k in _file_keys(fr):
                out.append(fr)
        return out

    def file_processed(self, key, *, require_success: bool = False) -> bool:
        """Whether a file has reached a terminal/finished state.

        By default, any terminal status counts (``SUCCESS``, ``FAILED``,
        ``SKIPPED``). When ``require_success=True``, only ``SUCCESS`` counts.

        Parameters
        ----------
        key : Any
            File identifier/name/path/basename.
        require_success : bool, default False
            If ``True``, require ``SUCCESS`` only.

        Returns
        -------
        bool
            ``True`` if processed under the selected rule.

        Notes
        -----
        As a additional heuristic, a file is considered processed if it
        has ``finished_at`` set or ``percent == 100``.
        """
        fr = self.get_file(key)
        if fr is None:
            return False
        if require_success:
            return fr.status.succeeded
        # terminal = finished state OR explicit 100% OR finished_at set
        return (
                fr.status.terminal
                or getattr(fr, "finished_at", None) is not None
                or (getattr(fr, "percent", None) == 100)
        )

    def file_seen(self, key) -> bool:
        """Whether a file with the given key appears in the report at all.

        Parameters
        ----------
        key : Any

        Returns
        -------
        bool
        """
        return self.get_file(key) is not None

    def get_file(self, key) -> "FileReport | None":
        """Return the first matching file report or ``None`` if absent.

        Parameters
        ----------
        key : Any
            See :meth:`files_for`.

        Returns
        -------
        FileReport or None
        """
        matches = self.files_for(key)
        return matches[0] if matches else None

    def iter_steps(self, *, status: Status | None = None) -> Iterable[StepReport]:
        """Iterate over batch-level steps, optionally filtering by status.

        Parameters
        ----------
        status : StepStatus or None, default None
            If provided, only yield steps whose status equals ``status``.

        Yields
        ------
        StepReport
            Matching steps in append order.
        """
        for s in self.steps:
            if status is None or s.status == status:
                yield s

    def last_step(self) -> StepReport | None:
        """Return the last batch-level step or ``None`` if empty.

        Returns
        -------
        StepReport or None
        """
        return self.steps[-1] if self.steps else None

    def recompute_overall_from_steps(self) -> None:
        """Recalculate overall ``percent`` from batch-level steps.

        Sets the banner percent to the arithmetic mean of child step
        percents and preserves the current ``stage``/``message`` (or
        uses defaults if unset). Does nothing if there are no steps.
        """
        if not self.steps:
            return
        pct = int(round(sum(s.percent for s in self.steps) / len(self.steps)))
        self.set_progress(self.stage or "steps", pct, self.message or "")

    def set_progress(self, stage: str, percent: int, message: str = "") -> None:
        """Update the progress banner and timestamp.

        Parameters
        ----------
        stage : str
            Short stage label (e.g., ``"discover"``).
        percent : int
            Clamped to ``0..100``.
        message : str, default ""
            UI-visible note.

        Examples
        --------
        >>> report = PipelineReport(...)
        >>> report.set_progress("discover", 5, "Scanning directory…")
        """
        self.stage = stage
        self.percent = max(0, min(100, percent))
        self.message = message
        self.updated_at = _now()

    def table_rows_for_files_map(self, expected: Mapping[str, Any]) -> list[dict]:
        """Produce Django/Jinja-friendly summary rows for a files mapping.

        Accepts a mapping of ``{filename_or_id_or_path: other_properties}``
        and returns table rows with normalized status fields for templating.

        Parameters
        ----------
        expected : Mapping[str, Any]
            Keys are identifiers/names/paths; values are attached as ``other``.

        Returns
        -------
        list[dict]
            Rows with fields:
            - ``filename`` : str
            - ``seen`` : bool
            - ``status`` : str (``"SUCCESS"|"FAILED"|"SKIPPED"|"RUNNING"|"PENDING"|"MISSING"``)
            - ``percent`` : int or None
            - ``flagged_human_review`` : bool
            - ``human_review_reason`` : str
            - ``file_id`` : str or None
            - ``path`` : str or None
            - ``other`` : Any (value from ``expected``)
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

    def unseen_expected(self, expected_iterable) -> list[str]:
        """Return expected filenames/ids/paths that are **not** present.

        Parameters
        ----------
        expected_iterable : Iterable[Any]
            Filenames/ids/paths expected to appear in :attr:`files`.

        Returns
        -------
        list[str]
            Items from ``expected_iterable`` that were not matched.
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


class ReportBase(BaseModel, ABC, ReviewHelpers):
    """Abstract base for report units (steps/files/batches).

    Note:
    ----
    Requires implementation of ok method in order to be instantiated.

    Provides a common lifecycle (``start → {succeed|fail|skip} → end``),
    structured messaging (``notes``, ``warnings``, ``errors``), metadata,
    and HITL review flags. Subclasses must define :attr:`ok` to indicate
    success when :meth:`end` infers a terminal status.

    Attributes
    ----------
    status : Status
        Current lifecycle status (defaults to ``PENDING``).
    percent : int
        Progress percentage ``0..100`` (informational; not enforced).
    started_at : datetime | None
        UTC timestamp when processing started.
    finished_at : datetime | None
        UTC timestamp when processing finalized.
    notes : list[str]
        Freeform narrative messages intended for UI display.
    errors : list[str]
        Fatal issues; presence typically implies failure.
    warnings : list[str]
        Non-fatal issues worth surfacing to users.
    metadata : dict[str, Any]
        Arbitrary structured context for search/analytics/UI.
    review : ReviewFlag
        Human-in-the-loop flag (``flagged`` + optional ``reason``).
    report_version : str
        Schema version written to JSON artifacts.
    defer_start: bool
        Initialization flag to skip .begin call upon construction.
    duration_ms : float|None
        Elapsed time in milliseconds (property).
    failed : bool
        Return `True` if the unit has failed (property).
    pending : bool
        Return `True` if the unit is pending (property).
    running : bool
        Return `True` if the unit is running (property).
    skipped : bool
        Return `True` if the unit was skipped (property).
    succeeded : bool
        Return `True` if the unit has succeeded (property).


    See Also
    --------
    StepReport
        Unit of work inside a file/batch.
    FileReport
        Ordered collection of steps for a single file.
    """

    status: Status = Status.PENDING
    percent: int = 0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    notes: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    review: ReviewFlag = Field(default_factory=ReviewFlag)
    report_version: str = SCHEMA_VERSION
    defer_start: bool = Field(default=False, exclude=True, repr=False)

    def error(self, msg: str) -> "ReportBase":
        """Append an error message (does not change status).

        Note
        ----
            Fail is the preferred method. error does not change status for design consistency.

        Parameters
        ----------
        msg : str
            Message to append to :attr:`errors`.

        Returns
        -------
        ReportBase
            Self.

        Notes
        -----
        Use :meth:`fail` to change terminal status to ``FAILED``. This method
        only records text.
        """
        self.errors.append(msg)
        return self

    def end(self) -> "ReportBase":
        """Finalize the unit if not already terminal.

        If already terminal (``SUCCESS``, ``FAILED``, ``SKIPPED``), this stamps
        :attr:`finished_at` if missing and returns. Otherwise, infers success
        from :attr:`ok` (roll-up of step outcomes) and calls :meth:`succeed`
        or :meth:`fail` accordingly.

        Returns
        -------
        ReportBase
            Self.
        """
        if self.status.terminal:
            if not self.finished_at:
                self.finished_at = _now()
            return self
        return self.succeed() if self.ok else self.fail("One or more file steps failed")

    def fail(self, message: Optional[str] = None) -> "ReportBase":
        """Finalize as failed (``FAILED``).

        Parameters
        ----------
        message : str, optional
            Error text to append to :attr:`errors`.

        Returns
        -------
        ReportBase
            Self.
        """
        self.status = Status.FAILED
        if message:
            self.error(message)
        self.finished_at = _now()
        return self

    def model_post_init(self, __context) -> None:
        # auto-start unless caller explicitly defers
        if not self.defer_start and self.pending:
            self.start()

    def note(self, msg: str) -> "ReportBase":
        """Append a user-facing note.

        Parameters
        ----------
        msg : str
            Message to append to :attr:`notes`.

        Returns
        -------
        ReportBase
            Self.
        """
        self.notes.append(msg)
        return self

    def skip(self, reason: Optional[str] = None) -> "ReportBase":
        """Finalize as skipped (``SKIPPED``).

        Parameters
        ----------
        reason : str, optional
            Rationale appended to :attr:`notes` as ``"Skipped: {reason}"``.

        Returns
        -------
        ReportBase
            Self.
        """
        self.status = Status.SKIPPED
        if reason:
            self.notes.append(f"Skipped: {reason}")
        self.finished_at = _now()
        return self

    def start(self) -> "ReportBase":
        """Mark the unit as running and stamp ``started_at`` if missing.

        Returns
        -------
        ReportBase
            Self (for fluent chaining).

        Examples
        --------
        >>> step = StepReport(...)
        >>> step.start().note("begin").percent == 0
        True
        """
        self.status = Status.RUNNING
        if not self.started_at:
            self.started_at = _now()
        return self

    def succeed(self) -> "ReportBase":
        """Finalize successfully (``SUCCEEDED``) and set ``percent=100``.

        Also stamps ``finished_at`` to the current time.

        Returns
        -------
        ReportBase
            Self.
        """
        self.status = Status.SUCCEEDED
        self.percent = 100
        self.finished_at = _now()
        return self

    def warn(self, msg: str) -> "ReportBase":
        """Append a non-fatal warning.

        Parameters
        ----------
        msg : str
            Message to append to :attr:`warnings`.

        Returns
        -------
        ReportBase
            Self.
        """
        self.warnings.append(msg)
        return self

    @property
    @abstractmethod
    def ok(self) -> bool:
        """Truthiness of success when inferring status.

        Subclasses define the success condition used by :meth:`end`
        when the unit is not already in a terminal status. Typical
        implementations derive this from fields like ``errors``,
        ``checks`` (for steps), or child statuses (for files/batches).

        Returns
        -------
        bool
            ``True`` if the unit should be considered successful.
        """
        raise NotImplementedError

    @computed_field
    @property
    def duration_ms(self) -> float | None:
        """
        Elapsed time in milliseconds.

        Returns None if timing cannot be determined (e.g., no started_at).
        Uses finished_at when present; otherwise uses 'now' to reflect
        in-flight duration. Clamped at >= 0 and rounded to 3 decimals.
        """
        if not self.started_at:
            return None
        end = self.finished_at or _now()
        delta_ms = (end - self.started_at).total_seconds() * 1000.0
        return round(max(delta_ms, 0.0), 3)

    @property
    def failed(self) -> bool:
        """Return ``True`` if the unit has failed."""
        return self.status.failed

    @property
    def pending(self):
        """Return ``True`` if the unit is pending."""
        return self.status.pending

    @property
    def running(self):
        """Return ``True`` if the unit is running."""
        return self.status.running

    @property
    def skipped(self):
        """Return ``True`` if the unit was skipped."""
        return self.status.skipped

    @property
    def succeeded(self):
        """Return ``True`` if the unit has succeeded."""
        return self.status.succeeded

    @property
    def terminal(self):
        """Return ``True`` if the unit is terminal."""
        return self.status.terminal


class FileReport(ReportBase):
    """Per-file processing timeline composed of ordered steps.

    Aggregates :class:`StepReport` items and provides convenience methods
    for appending terminal steps (success/failed/skipped) and requesting
    human-in-the-loop (HITL) review. Designed to be JSON-serialized and
    consumed by UIs.

    Attributes
    ----------
    path (required) : Path or None
        Source path or URI for display/debugging.
    file_id : str
        Stable identifier for the file (preferably unique within a batch).
    steps : list[StepReport]
        Ordered step sequence (normally appended via helpers).
    n_steps : int
        The number of steps in the process (used for computing percent).
    requires_human_review (property): bool
        Whether this file needs human review (computed)
    human_review_reason (property): str
        Compact human-readable reason summarizing review needs (computed).
    size_bytes (property): Optional[int]
        Best-effort determination of file's size in bytes.

    Notes
    -----
    See ReportBase for additional attributes, properties and methods
    The auto-generated initializer accepts the same fields as attributes.
    """
    model_config = ConfigDict(extra='forbid')
    path: Path
    file_id: Optional[str] = None
    steps: List[StepReport] = Field(default_factory=list)
    n_steps: int = 1
    _pipeline: Optional["PipelineReport"] = PrivateAttr(default=None)

    def add_completed_step(
        self,
        label: str,
        *,
        id: str | None = None,
        note: str | None = None,
        metadata: dict | None = None,
    ) -> "FileReport":
        """Create a ``SUCCEEDED`` step and append it (chainable).

        Parameters
        ----------
        label : str
            Step label for UI.
        id : str, optional
            Explicit step id; if omitted, a unique id is derived from ``label``.
        note (optional): str
            Optional note appended to the step.
        metadata : dict, optional
            Metadata merged into the step.

        Returns
        -------
        FileReport
            Self.
        """
        sid = id or self._make_unique_step_id(label)
        step = StepReport.begin(label, id=sid)
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
        """Create a ``FAILED`` step and append it (chainable).

        Parameters
        ----------
        label : str
            Step label.
        id : str, optional
            Explicit id; if omitted, derived from ``label``.
        reason : str, optional
            Failure reason recorded on the step.
        metadata : dict, optional
            Metadata merged into the step.

        Returns
        -------
        FileReport
            Self.
        """
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
        """Create a ``SKIPPED`` step and append it (chainable).

        Parameters
        ----------
        label : str
            Step label.
        id : str, optional
            Explicit id; if omitted, derived from ``label``.
        reason : str, optional
            Skip rationale (added to notes).
        metadata : dict, optional
            Metadata merged into the step.

        Returns
        -------
        FileReport
            Self.
        """
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
        """Create a step that requests HITL review, then append it.

        By default the step is marked ``SUCCESS`` (common pattern: “passed
        but needs review”). The file-level review flag will be set when
        appended if the file isn’t already flagged.

        Parameters
        ----------
        label : str
            Step label.
        id : str, optional
            Explicit id; if omitted, derived from ``label``.
        reason : str, optional
            UI-visible reason for review.
        metadata : dict, optional
            Extra context for reviewers.
        mark_success : bool, default True
            If ``True``, mark the step ``SUCCESS``; otherwise leave status as-is.

        Returns
        -------
        FileReport
            Self.
        """
        sid = id or self._make_unique_step_id(label)
        step = StepReport.begin(sid, label=label).request_review(reason)
        if metadata:
            step.metadata.update(metadata)
        if mark_success:
            step.succeed()
        # If your append_step rolls review up to the FileReport, that'll happen there.
        return self.append_step(step)

    def append_step(self,
                    step: StepReport,
                    max_steps: int = 10_000) -> "FileReport":
        """Finalize and append a step; recompute aggregate percent.

        The step is finalized via [StepReport.end](.#pipeline_watcher.StepReport.end), appended to
        [steps](.#pipeline_watcher.FileReport.steps), and the file percent is updated as the arithmetic
        mean of child step percents. If the step requests HITL review and
        the file is not already flagged, the file's [review](.#pipeline_watcher.FileReport.review) is set.

        Parameters
        ----------
        step : StepReport
            Step to finalize and append.
        max_steps : int, optional
            The maximum number of steps allowed.

        Returns
        -------
        FileReport
            Self (chainable).
        """
        step.end()
        if not step.id:
            step.id = construct_unique_step_id_from_label(step.label, self.steps)
        step.id = make_step_id_unique(step.id, self.steps)
        self.steps.append(step)
        self._recompute_percent()
        # roll-up HITL review if you added ReviewFlag earlier
        if step.review.flagged and not self.review.flagged:
            self.review = ReviewFlag(flagged=True,
                                     reason=step.review.reason or f"Step '{step.id}' requested review")
        return self

    @classmethod
    def begin(cls,
              path: Path | str,
              file_id: str | None = None,
              n_steps: int = 1,
              metadata: dict | None = None
    ) -> "FileReport":
        """Construct and mark the file report as running.

        Parameters
        ----------
        path : Path
            The path to file.
        file_id (optional) : str
            Stable file identifier.
        n_steps : int, optional
            Number of steps in process.
        metadata (optional): dict
            Dictionary of metadata about the file.

        Returns
        -------
        FileReport
            Started file report (``status=RUNNING``).
        """
        return cls(path=Path(path),
                   file_id=file_id,
                   n_steps=n_steps,
                   metadata=dict(metadata) if metadata else {}).start()


    def last_step(self) -> StepReport | None:
        """Return the most recently appended step or ``None`` if empty.

        Returns
        -------
        StepReport or None
            Last step in :attr:`steps`, if any.
        """
        return self.steps[-1] if self.steps else None

    @property
    def ok(self) -> bool:
        """Truthiness of success used by :meth:`end`.

        Returns
        -------
        bool
            ``False`` if status is ``FAILED`` or any errors exist.
            ``True`` if status is ``SUCCEEDED``.
            Otherwise, ``all(step.ok for step in steps)`` (or ``True`` if no steps).
        """
        # Failure wins
        if self.failed or self.errors:
            return False
        # Explicit success wins
        if self.succeeded:
            return True
        # Otherwise roll up from steps and surface errors
        all_steps_ok = all(s.ok for s in self.steps) if self.steps else True
        if not all_steps_ok:
            errors = ["; ".join(step.errors) for step in self.steps if not step.ok]
            for error in errors:
                self.errors.append(error)
        return all_steps_ok

    @computed_field
    @property
    def name(self) -> str:
        # read-only, derived; not persisted unless you include computed fields explicitly
        return self.path.name

    @computed_field
    @property
    def mime_type(self) -> Optional[str]:
        # Extension-based guess; no filesystem touch
        mt, _ = mimetypes.guess_type(self.path.as_posix())
        return mt

    @computed_field
    @property
    def size_bytes(self) -> Optional[int]:
        """Best-effort determination of file's size in bytes.
        Avoid raising on missing/inaccessible paths."""
        try:
            # Use os.path.getsize for slightly cheaper syscall than full stat attr unpack
            return os.path.getsize(self.path)
        except Exception:
            return None

    @computed_field
    @property
    def requires_human_review(self) -> bool:
        """Whether this file needs human review (computed).

        True if the file itself is flagged or any contained step is flagged.

        Returns
        -------
        bool
            ``True`` if review is required; otherwise ``False``.
        """
        if self.review and self.review.flagged:
            return True
        return any(True for _ in self._flagged_steps())

    @computed_field
    @property
    def human_review_reason(self) -> Optional[str]:
        """Compact human-readable reason summarizing review needs (computed).

        Combines file-level reason (if present) with a roll-up of flagged steps.
        Shows up to five step names and a “+N more” suffix if necessary.

        Returns
        -------
        str or None
            Summary text or ``None`` if no review is requested.
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

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, v):
        if isinstance(v, (str, Path)):
            p = Path(v)
            if str(p) == "":
                raise ValueError("path cannot be empty")
            return p
        raise TypeError("path must be str or Path")

    def _flagged_steps(self) -> List[StepReport]:
        """Return steps that have requested human review.

        Returns
        -------
        list[StepReport]
            Subset of :attr:`steps` whose ``review.flagged`` is ``True``.
        """
        out = []
        for s in self.steps:
            rf = getattr(s, "review", None)
            if rf and getattr(rf, "flagged", False):
                out.append(s)
        return out

    def _recompute_percent(self) -> None:
        """Recompute :attr:`percent` as the fraction of completed steps.
        """
        if self.n_steps and self.steps:
            self.percent = int(100 * sum(1 for s in self.steps if s.terminal) / self.n_steps + 0.5)
        else:
            self.percent = 0

    def _make_unique_step_id(self, label: str) -> str:
        """Generate a slugified, unique step id based on a label.

        If the slug already exists among current steps, appends ``-2``, ``-3``,
        etc., until unique.

        Parameters
        ----------
        label : str
            Human-readable label to slugify.

        Returns
        -------
        str
            Unique step identifier.
        """
        base = _slugify(label) or "step"
        existing = {s.id for s in self.steps}
        if base not in existing:
            return base
        i = 2
        while f"{base}-{i}" in existing:
            i += 1
        return f"{base}-{i}"

class StepReport(ReportBase):
    """Single unit of work within a file or batch.

    A step succeeds if
    it is explicitly marked ``SUCCESS`` or, when not terminal, if all
    recorded checks pass and no errors are present.

    Attributes
    ----------
    label : str or None
        Human-readable label for UI display.
    id (optional) : str
        Machine-friendly identifier (e.g., ``"parse"``, ``"analyze"``).
    checks : list[Check]
        Recorded boolean validations for this step.
    status : Status
        Lifecycle state inherited from :class:`ReportBase`.
    percent : int
        Optional progress indicator (0..100).
    notes : list[str]
        Narrative messages intended for UI.
    warnings : list[str]
        Non-fatal issues.
    errors : list[str]
        Fatal issues; presence typically implies failure.
    metadata : dict[str, Any]
        Arbitrary structured context.
    review : ReviewFlag
        HITL review flag attached to the step.
    report_version : str
        Schema version for persisted artifacts.

    Notes
    -----
    `id` is optional because in some cases the burdern to construct a unique id
        is placed on the container (e.g. FileReport). Therefore, the unique id
        construction is by default deferred to the container.
    The auto-generated initializer accepts the same fields as attributes.

    Examples
    --------
    >>> st = StepReport.begin("Extract text (OCR)")
    ... st.add_check("ocr_quality>=0.9", ok=True)
    ... st.end() # note end is typically called via e.g. FileReport.append_step where other finalization occurs.
    ... st.succeeded
    True
    >>> st.terminal
    True
    >>> st.duration_ms
    314
    """
    label: str
    id: Optional[str] = None
    checks: List[Check] = Field(default_factory=list)

    @model_validator(mode="after")
    def _default_id(self):
        # Treat None/"" as unset; drop the `or self.label == ""` part if "" is meaningful
        if self.id is None or self.id == "":
            # When frozen, use object.__setattr__ during construction-time adjustments
            object.__setattr__(self, "id", _slugify(self.label))
        return self

    @classmethod
    def begin(cls, label: str, *, id: str | None = None) -> "StepReport":
        """Construct and mark the step as started.

        Parameters
        ----------
        label : str
            Human-friendly label.
        id : str | None, optional
            Step identifier.

        Returns
        -------
        StepReport
            Started step report (``status=RUNNING``).
        """
        if id is None or id == "":
            id = _slugify(label)
        return cls(label=label, id=id).start()

    @property
    def ok(self) -> bool:
        """Reviews checks and surfaces errors.

        Returns
        -------
        bool
            ``False`` if status is ``FAILED`` or any errors exist.
            ``True`` if status is ``SUCCESS``.
            Otherwise, ``all(check.ok for check in checks)`` (or ``True`` if no checks).
        """
        if self.status.failed or self.errors:
            return False
        if self.status.succeeded:
            return True
        all_checks_passed = all(c.ok for c in self.checks) if self.checks else True
        if not all_checks_passed:
            for check in self.checks:
                if not check.ok:
                    self.errors.append(check.detail)
        return all_checks_passed

    def add_check(self, name: str, ok: bool, detail: Optional[str] = None) -> None:
        """Record a boolean validation result.

        Parameters
        ----------
        name : str
            Check identifier (e.g., ``"manifest_present"``).
        ok : bool
            Outcome of the check.
        detail : str, optional
            Additional context for UI/debugging.

        Examples
        --------
        >>> st = StepReport.begin("validate")
        >>> st.add_check("ids_unique", ok=False, detail="3 duplicates")
        >>> st.ok
        False
        """
        self.checks.append(Check(name=name, ok=ok, detail=detail))



# Thread/async-safe context variable holding the bound PipelineReport.
# Used by helpers (e.g., pipeline_step) to discover the active pipeline
# without passing it explicitly. Prefer the bind_pipeline() context manager
# to set/reset this value safely.
_current_pipeline_report: contextvars.ContextVar[Optional["PipelineReport"]] = contextvars.ContextVar(
    "_current_pipeline_report", default=None
)


@contextmanager
def bind_pipeline(pr: "PipelineReport"):
    """Bind a :class:`PipelineReport` to the current context.

    Makes nested helpers (e.g., :func:`pipeline_step`) discover the active
    pipeline implicitly via a context variable, avoiding the need to pass
    ``pr`` on every call. Safe across threads/async tasks.

    Parameters
    ----------
    pr : PipelineReport
        The pipeline report to bind in this context.

    Yields
    ------
    PipelineReport
        The same report passed in, for convenience.

    Examples
    --------
    >>> report = PipelineReport(...)
    >>> with bind_pipeline(report):
    ...     # Inside this block, pipeline_step(None, ...) will use `report`
    ...     with pipeline_step(None, "discover", label="Discover inputs") as st:
    ...         st.note("Scanning directory")
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
    """Context manager for a **batch-level** step.

    Creates a :class:`StepReport`, times it, captures exceptions (optional
    re-raise), and appends it to the provided or bound pipeline. Also supports
    updating the pipeline's progress banner on enter/exit.

    Parameters
    ----------
    pr : PipelineReport or None
        Pipeline to append the step to. If ``None``, uses the pipeline bound
        via :func:`bind_pipeline`. If neither is available, the step is still
        yielded but not appended.
    id : str
        Machine-friendly step id (e.g., ``"validate_manifest"``).
    label : str, optional
        Human-friendly label for UI display.
    banner_stage : str, optional
        Stage label to set on the pipeline banner after appending the step
        (or on enter if ``set_stage_on_enter=True``). Defaults to ``id`` when
        provided as ``None``.
    banner_percent : int, optional
        Percent to set on the banner after appending. If ``None``, leaves the
        current percent unchanged.
    banner_message : str, optional
        Message to set on the banner (falls back to existing message if ``None``).
    set_stage_on_enter : bool, default False
        If ``True``, set the banner's ``stage``/``message`` when entering the
        context (percent remains unchanged on enter).
    raise_on_exception : bool, default False
        If ``True``, re-raise the exception after recording it on the step.
        If ``False``, swallow after recording so the pipeline can continue.
    save_on_exception : bool, default True
        If an exception occurs and a pipeline is available, attempt to save
        the pipeline JSON immediately (best-effort).
    output_path_override : str or Path or None, optional
        When saving on exception, write to this path instead of
        ``pr.output_path`` if provided.

    Yields
    ------
    StepReport
        The live step report to populate within the ``with`` block.

    Notes
    -----
    - Exceptions inside the block are recorded as:
      - ``errors += [\"{Type}: {message}\"]``
      - ``metadata['traceback'] = traceback.format_exc()``
      - status set to ``FAILED`` via ``st.fail(...)``
    - If a pipeline is available, the step is finalized via ``st.end()``,
      appended with :meth:`PipelineReport.append_step`, and the banner is
      optionally updated.

    Examples
    --------
    Minimal with bound pipeline:
    >>> report = PipelineReport(...)
    >>> with bind_pipeline(report):
    ...     with pipeline_step(None, "index", label="Index batch") as st:
    ...         st.add_check("manifest_present", ok=True)

    Update the banner and save immediately on error:

    >>> with pipeline_step(
    ...     report, "extract",
    ...     banner_stage="extract",
    ...     banner_percent=40,
    ...     banner_message="Extracting data…",
    ...     save_on_exception=True,
    ... ):
    ...     # proceed with calculation...
    """
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
    """Bind a :class:`PipelineReport` to the current context.

    Lets nested helpers (e.g., :func:`pipeline_step`, :func:`pipeline_file`)
    discover the active pipeline via a thread/async-safe context variable,
    so you don’t have to pass ``pr`` explicitly.

    Parameters
    ----------
    pr : PipelineReport
        The pipeline to bind for the duration of the context.

    Yields
    ------
    PipelineReport
        The same report object (convenience).

    Examples
    --------
    >>> report = PipelineReport(...)
    ... with bind_pipeline(report):
    ...     # inside, helpers can omit `pr`
    ...     ...
    """
    token = _current_pipeline_report.set(pr)
    try:
        yield pr
    finally:
        _current_pipeline_report.reset(token)


_SETTINGS_KEYS: set[str] = {f.name for f in fields(WatcherSettings)}


@contextmanager
def pipeline_file(
    pr: Optional["PipelineReport"],
    path: Path | str,
    *,
    file_id: str | None = None,
    metadata: dict | None = None,
    set_stage_on_enter: bool = False,
    banner_stage: str | None = None,
    banner_percent_on_exit: int | None = None,
    banner_message_on_exit: str | None = None,
    **other_options,
):
    """
    Per-file processing block using WatcherSettings as the source of truth.

    Pass any WatcherSettings fields as kwargs (e.g., raise_on_exception=True)
    and they will apply only within this context; otherwise the current
    context settings are used.
    """
    settings_overrides = {k: v for k, v in other_options.items() if k in _SETTINGS_KEYS}

    # Bind to a pipeline if not explicitly provided
    pr = pr or _current_pipeline_report.get(None)
    if pr is None:
        raise RuntimeError(
            "pipeline_file requires a PipelineReport: pass `pr=` or call within `with bind_pipeline(pr):`"
        )

    if not isinstance(path, (str, os.PathLike)):
        raise ValueError(f"path must be str or os.PathLike, not {type(path)!r}")

    # Apply settings overrides (if any) only for this block
    with use_settings(**settings_overrides) as settings:
        fr = FileReport.begin(
            path=Path(path),
            file_id=file_id,
            metadata=dict(metadata) if metadata else {},
        )
        fr._pipeline = pr

        # Optional banner update on enter
        if set_stage_on_enter:
            pr.set_progress(
                stage=banner_stage or fr.name,   # name always present from path
                percent=pr.percent,
                message=pr.message,
            )

        # Optional capture (settings-driven)
        stdout_buf = StringIO() if settings.capture_streams else None
        stderr_buf = StringIO() if settings.capture_streams else None
        warn_list: list[warnings.WarningMessage] | None = None

        with ExitStack() as stack:
            if settings.capture_warnings:
                warn_list = stack.enter_context(warnings.catch_warnings(record=True))
                warnings.simplefilter("default")
            if settings.capture_streams:
                if stdout_buf:
                    stack.enter_context(redirect_stdout(stdout_buf))
                if stderr_buf:
                    stack.enter_context(redirect_stderr(stderr_buf))

            encountered_exception = False
            try:
                yield fr
                fr.succeed()
            except BaseException as e:
                # --- Always record first ---
                encountered_exception = True
                fr.errors.append(f"{type(e).__name__}: {e}")
                if settings.store_traceback:
                    tb = "".join(
                        traceback.format_exception(type(e), e, e.__traceback__, limit=settings.traceback_limit)
                    )
                    if tb:
                        fr.metadata["traceback"] = tb
                fr.fail("Unhandled exception while processing file")

                # --- Decide raising via settings helpers ---
                if settings.should_raise(e):
                    raise
                if (msg := settings.suppression_breadcrumb(e)):
                    fr.warnings.append(msg)
            finally:
                # Persist diagnostics
                if stdout_buf is not None:
                    fr.metadata["stdout"] = stdout_buf.getvalue()
                if stderr_buf is not None:
                    fr.metadata["stderr"] = stderr_buf.getvalue()
                if warn_list is not None:
                    fr.metadata["warnings"] = [
                        f"{w.category.__name__}: {w.message}"  # type: ignore[attr-defined]
                        for w in warn_list
                    ]
                fr.end()

                # Append to pipeline and (optionally) update banner on exit
                try:
                    pr.append_file(fr)
                    if (banner_stage is not None
                        or banner_percent_on_exit is not None
                        or banner_message_on_exit is not None):
                        pr.set_progress(
                            stage=banner_stage or pr.stage,
                            percent=pr.percent if banner_percent_on_exit is None else banner_percent_on_exit,
                            message=banner_message_on_exit or pr.message,
                        )
                finally:
                    # Best-effort save-on-exception
                    if encountered_exception and settings.save_on_exception:
                        try:
                            save_path = settings.exception_save_path_override or pr.output_path
                            if save_path:
                                pr.save(save_path)
                            else:
                                fr.warnings.append("auto-save skipped: no output path configured")
                        except Exception as save_err:
                            fr.warnings.append(f"save failed: {save_err!r}")


@contextmanager
def file_step(
    file_report: FileReport,
    id: str,
    *,
    label: str | None = None,
    **other_options,
):
    """
    Context manager for a step inside a FileReport, governed by WatcherSettings.

    Pass any WatcherSettings fields as kwargs (e.g., raise_on_exception=True).
    They apply only within this context; otherwise current settings are used.
    """
    settings_overrides = {
        k: v for k, v in other_options.items() if k in _SETTINGS_KEYS
    }

    with use_settings(**settings_overrides) as settings:
        st = StepReport.begin(id=id, label=label)

        stdout_buf = StringIO() if settings.capture_streams else None
        stderr_buf = StringIO() if settings.capture_streams else None
        warn_list: list[warnings.WarningMessage] | None = None
        encountered_exception = False

        with ExitStack() as stack:
            if settings.capture_warnings:
                warn_list = stack.enter_context(
                    warnings.catch_warnings(record=True)
                )
                warnings.simplefilter("default")

            if settings.capture_streams:
                if stdout_buf is not None:
                    stack.enter_context(redirect_stdout(stdout_buf))
                if stderr_buf is not None:
                    stack.enter_context(redirect_stderr(stderr_buf))

            try:
                yield st
                st.succeed()

            except BaseException as e:
                encountered_exception = True
                exc_type_name = type(e).__name__

                st.errors.append(f"{exc_type_name}: {e}")

                if settings.store_traceback:
                    tb = "".join(
                        traceback.format_exception(
                            type(e), e, e.__traceback__,
                            limit=settings.traceback_limit,
                        )
                    )
                    if tb:
                        st.metadata["traceback"] = tb

                st.fail(f"Unhandled {exc_type_name} in file step")

                if settings.should_raise(e):
                    raise

                if msg := settings.suppression_breadcrumb(e):
                    st.warnings.append(msg)

            finally:
                if stdout_buf is not None:
                    st.metadata["stdout"] = stdout_buf.getvalue()
                if stderr_buf is not None:
                    st.metadata["stderr"] = stderr_buf.getvalue()
                if warn_list is not None:
                    st.metadata["warnings"] = [
                        f"{w.category.__name__}: {w.message}"
                        for w in warn_list
                    ]

                # Finalize step via FileReport: append_step calls st.end(),
                # enforces id uniqueness, updates timestamps, etc.
                file_report.append_step(st)

                # Auto-save via owning pipeline, if available
                if encountered_exception and settings.save_on_exception:
                    pipeline = getattr(file_report, "pipeline", None)
                    if pipeline is not None:
                        try:
                            save_path = (
                                settings.exception_save_path_override
                                or pipeline.output_path
                            )
                            if save_path:
                                pipeline.save(save_path)
                            else:
                                st.warnings.append(
                                    "auto-save skipped: no pipeline output path configured"
                                )
                        except Exception as save_err:
                            st.warnings.append(f"save failed: {save_err!r}")
                    else:
                        # Invariant *should* be: file_report is always tied to a pipeline.
                        # But this keeps the behavior graceful if that’s ever relaxed.
                        st.warnings.append(
                            "auto-save skipped: file_report not attached to a pipeline"
                        )


@contextmanager
def pipeline_step(
    pr: Optional[PipelineReport],
    label: str,
    *,
    id: str | None = None,
    **other_options,
):
    """
    Context manager for a step inside a PipelineReport, governed by WatcherSettings.

    Pass any WatcherSettings fields as kwargs (e.g., raise_on_exception=True).
    They apply only within this context; otherwise current settings are used.

    The associated PipelineReport is either passed explicitly via `pr=`
    or discovered from the current bind_pipeline() context.
    """
    # Filter only valid WatcherSettings keys
    settings_overrides = {k: v for k, v in other_options.items() if k in _SETTINGS_KEYS}

    # Bind to a pipeline if not explicitly provided
    pr = pr or _current_pipeline_report.get(None)
    if pr is None:
        raise RuntimeError(
            "pipeline_step requires a PipelineReport: pass `pr=` or call within `with bind_pipeline(pr):`"
        )

    # Apply overrides (if any) for just this block
    with use_settings(**settings_overrides) as settings:
        st = StepReport.begin(label, id=id)

        # Optional capture (settings-driven)
        stdout_buf = StringIO() if settings.capture_streams else None
        stderr_buf = StringIO() if settings.capture_streams else None
        warn_list: list[warnings.WarningMessage] | None = None
        encountered_exception = False

        with ExitStack() as stack:
            if settings.capture_warnings:
                warn_list = stack.enter_context(warnings.catch_warnings(record=True))
                warnings.simplefilter("default")

            if settings.capture_streams:
                if stdout_buf is not None:
                    stack.enter_context(redirect_stdout(stdout_buf))
                if stderr_buf is not None:
                    stack.enter_context(redirect_stderr(stderr_buf))

            try:
                # User code runs here
                yield st
                st.succeed()
            except BaseException as e:
                encountered_exception = True

                exc_type_name = type(e).__name__

                # Record error + optional detail
                st.errors.append(f"{exc_type_name}: {e}")

                if settings.store_traceback:
                    tb = "".join(
                        traceback.format_exception(
                            type(e), e, e.__traceback__, limit=settings.traceback_limit
                        )
                    )
                    if tb:
                        st.metadata["traceback"] = tb

                # Status line surfaces exception *name* even if traceback is suppressed
                st.fail(f"Unhandled {exc_type_name} in pipeline step")

                if settings.should_raise(e):
                    raise

                if msg := settings.suppression_breadcrumb(e):
                    st.warnings.append(msg)

            finally:
                # Persist diagnostics
                if stdout_buf is not None:
                    st.metadata["stdout"] = stdout_buf.getvalue()
                if stderr_buf is not None:
                    st.metadata["stderr"] = stderr_buf.getvalue()
                if warn_list is not None:
                    st.metadata["warnings"] = [
                        f"{w.category.__name__}: {w.message}"  # type: ignore[attr-defined]
                        for w in warn_list
                    ]

                # Let the pipeline/file own finalization:
                # append_step(st) will call st.end(), enforce id uniqueness,
                # and update the FileReport's update time.
                pr.append_step(st)

                # Optional: mirror pipeline_file best-effort auto-save
                if encountered_exception and settings.save_on_exception:
                    try:
                        save_path = settings.exception_save_path_override or pr.output_path
                        if save_path:
                            pr.save(save_path)
                        else:
                            st.warnings.append("auto-save skipped: no output path configured")
                    except Exception as save_err:
                        st.warnings.append(f"save failed: {save_err!r}")
