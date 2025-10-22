from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Literal
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from .clocks import now_utc as _now

SCHEMA_VERSION = "v2"  # steps are lists; append_* auto-finalizes via .end()

class StepStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED  = "FAILED"
    SKIPPED = "SKIPPED"

class Check(BaseModel):
    name: str
    ok: bool
    detail: Optional[str] = None

class StepReport(BaseModel):
    """
    One unit of work (e.g., 'parse', 'analyze'). Designed to be JSON-serialized.
    """
    id: str
    label: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    percent: int = 0  # 0..100
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    checks: List[Check] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    report_version: str = SCHEMA_VERSION

    def start(self) -> "StepReport":
        self.status = StepStatus.RUNNING
        if not self.started_at:
            self.started_at = _now()
        return self

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

    def succeed(self) -> "StepReport":
        self.status = StepStatus.SUCCESS
        self.percent = 100
        self.finished_at = _now()
        return self

    def fail(self, message: Optional[str] = None) -> "StepReport":
        self.status = StepStatus.FAILED
        if message:
            self.errors.append(message)
        self.finished_at = _now()
        return self

    def skip(self, reason: Optional[str] = None) -> "StepReport":
        self.status = StepStatus.SKIPPED
        if reason:
            self.notes.append(f"Skipped: {reason}")
        self.finished_at = _now()
        return self

    def end(self) -> "StepReport":
        """
        Idempotent finalize:
        - Terminal → ensure finished_at set; return self.
        - Otherwise → succeed if ok else fail with a generic message.
        """
        if self.status in (StepStatus.SUCCESS, StepStatus.FAILED, StepStatus.SKIPPED):
            if not self.finished_at:
                self.finished_at = _now()
            return self
        return self.succeed() if self.ok else self.fail(
            "One or more checks failed" if self.checks else "Step failed"
        )

class FileReport(BaseModel):
    """
    Per-document aggregation with its own ordered sequence of steps.
    """
    file_id: str
    path: str | None = None
    name: str | None = None
    size_bytes: int | None = None
    mime_type: str | None = None

    status: StepStatus = StepStatus.PENDING
    percent: int = 0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    steps: List[StepReport] = Field(default_factory=list)

    notes: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    report_version: str = SCHEMA_VERSION

    def start(self) -> "FileReport":
        self.status = StepStatus.RUNNING
        if not self.started_at:
            self.started_at = _now()
        return self

    @classmethod
    def begin(cls, file_id: str, **meta) -> "FileReport":
        return cls(file_id=file_id, **meta).start()

    @property
    def ok(self) -> bool:
        if self.status == StepStatus.FAILED or self.errors:
            return False
        if self.status == StepStatus.SUCCESS:
            return True
        return all(s.ok for s in self.steps) if self.steps else True

    def append_step(self, step: StepReport) -> StepReport:
        step.end()                     # auto-finalize
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

    def succeed(self) -> "FileReport":
        self.status = StepStatus.SUCCESS
        self.percent = 100
        self.finished_at = _now()
        return self

    def fail(self, message: Optional[str] = None) -> "FileReport":
        self.status = StepStatus.FAILED
        if message:
            self.errors.append(message)
        self.finished_at = _now()
        return self

    def skip(self, reason: Optional[str] = None) -> "FileReport":
        self.status = StepStatus.SKIPPED
        if reason:
            self.notes.append(f"Skipped: {reason}")
        self.finished_at = _now()
        return self

    def _recompute_percent(self) -> None:
        if self.steps:
            self.percent = int(round(sum(s.percent for s in self.steps) / len(self.steps)))

class PipelineReport(BaseModel):
    """
    Batch-level container with an ordered list of steps and per-file reports.
    """
    batch_id: int
    kind: Literal["validation", "process"]
    stage: str = ""
    percent: int = 0
    message: str = ""
    updated_at: datetime = Field(default_factory=_now)
    report_version: str = SCHEMA_VERSION

    steps: List[StepReport] = Field(default_factory=list)
    files: List[FileReport] = Field(default_factory=list)

    def set_progress(self, stage: str, percent: int, message: str = "") -> None:
        self.stage = stage
        self.percent = max(0, min(100, percent))
        self.message = message
        self.updated_at = _now()

    def append_step(self, step: StepReport) -> StepReport:
        step.end()                     # auto-finalize
        self.steps.append(step)
        self.updated_at = _now()
        return step

    def append_file(self, fr: FileReport) -> FileReport:
        fr.end()                       # auto-finalize
        self.files.append(fr)
        self.updated_at = _now()
        return fr

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

