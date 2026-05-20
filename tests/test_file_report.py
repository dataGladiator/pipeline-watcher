from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline_watcher import FileReport, Status, StepReport


def _step(
    label: str,
    status: Status = Status.SUCCEEDED,
    *,
    percent: int | None = None,
    id: str | None = None,
) -> StepReport:
    if status is Status.PENDING:
        step = StepReport(label=label, id=id, defer_start=True)
    else:
        step = StepReport.begin(label, id=id)

    if status is Status.SUCCEEDED:
        step.succeed()
    elif status is Status.FAILED:
        step.fail("step failed")
    elif status is Status.SKIPPED:
        step.skip("not needed")
    elif status is Status.RUNNING:
        pass

    if percent is not None:
        step.percent = percent
    return step


def test_begin_sets_path_identity_metadata_and_running_lifecycle():
    metadata = {"source": "fixture", "item_count": 3}

    file_report = FileReport.begin(
        path="outputs/result.json",
        file_id="result",
        metadata=metadata,
    )

    assert file_report.path == Path("outputs/result.json")
    assert file_report.file_id == "result"
    assert file_report.metadata == metadata
    assert file_report.metadata is not metadata
    assert file_report.running
    assert file_report.percent == 0
    assert file_report.started_at is not None
    assert file_report.finished_at is None
    assert file_report.steps == []


def test_path_derived_fields_are_available_without_touching_missing_file():
    file_report = FileReport.begin("outputs/result.json")

    assert file_report.label == "result.json"
    assert file_report.name == "result.json"
    assert file_report.n_steps == 0
    assert file_report.mime_type == "application/json"
    assert file_report.size_bytes is None


def test_size_bytes_reports_existing_file_size(tmp_path: Path):
    output = tmp_path / "result.txt"
    output.write_bytes(b"pipeline")

    file_report = FileReport.begin(output)

    assert file_report.size_bytes == len(b"pipeline")


def test_append_step_finalizes_step_assigns_unique_id_and_updates_percent():
    file_report = FileReport.begin("outputs/result.json")
    first = StepReport.begin("Load inputs")
    second = StepReport.begin("Load inputs")

    file_report.append_step(first)
    file_report.append_step(second)

    assert file_report.steps == [first, second]
    assert [step.id for step in file_report.steps] == ["load-inputs", "load-inputs-1"]
    assert all(step.succeeded for step in file_report.steps)
    assert all(step.terminal for step in file_report.steps)
    assert file_report.percent == 100


def test_completed_failed_skipped_step_helpers_create_terminal_steps():
    file_report = FileReport.begin("outputs/result.json")

    assert file_report.add_completed_step(
        "Load inputs",
        note="Loaded source data",
        metadata={"records": 10},
    ) is file_report
    assert file_report.add_failed_step(
        "Compute result",
        reason="Computation failed",
        metadata={"attempt": 2},
    ) is file_report
    assert file_report.add_skipped_step(
        "Render output",
        reason="No result",
        metadata={"cached": False},
    ) is file_report

    completed, failed, skipped = file_report.steps

    assert completed.succeeded
    assert completed.notes == ["Loaded source data"]
    assert completed.metadata == {"records": 10}

    assert failed.failed
    assert failed.errors == ["Computation failed"]
    assert failed.metadata == {"attempt": 2}

    assert skipped.skipped
    assert skipped.notes == ["Skipped: No result"]
    assert skipped.metadata == {"cached": False}


def test_last_step_returns_none_then_latest_step():
    file_report = FileReport.begin("outputs/result.json")
    first = _step("Load inputs")
    second = _step("Compute result")

    assert file_report.last_step() is None

    file_report.steps.extend([first, second])

    assert file_report.last_step() is second


def test_ok_rolls_up_child_step_failures_and_end_marks_file_failed():
    file_report = FileReport.begin("outputs/result.json")
    file_report.append_step(_step("Load inputs"))
    file_report.append_step(_step("Compute result", Status.FAILED))

    file_report.end()

    assert file_report.failed
    assert file_report.errors == ["step failed", "One or more file steps failed"]
    assert file_report.finished_at is not None


def test_end_succeeds_when_all_steps_are_ok():
    file_report = FileReport.begin("outputs/result.json")
    file_report.append_step(_step("Load inputs"))
    file_report.append_step(_step("Use cached output", Status.SKIPPED))

    file_report.end()

    assert file_report.succeeded
    assert file_report.percent == 100
    assert file_report.finished_at is not None


def test_review_step_rolls_up_human_review_to_file():
    file_report = FileReport.begin("outputs/result.json")

    file_report.add_review_step(
        "Validate result",
        reason="Confidence below threshold",
        metadata={"confidence": 0.72},
    )

    step = file_report.steps[0]
    assert step.succeeded
    assert step.review.flagged
    assert step.metadata == {"confidence": 0.72}
    assert file_report.review.flagged
    assert file_report.review.reason == "Confidence below threshold"
    assert file_report.requires_human_review
    assert file_report.human_review_reason == (
        "File-level review: Confidence below threshold "
        "step flagged human review: Validate result. "
        "First reason: Confidence below threshold."
    )


def test_human_review_reason_summarizes_multiple_flagged_steps():
    file_report = FileReport.begin("outputs/result.json")

    for index in range(6):
        file_report.add_review_step(
            f"Review check {index + 1}",
            reason=f"Reason {index + 1}",
        )

    assert file_report.human_review_reason == (
        "File-level review: Reason 1 "
        "6 steps flagged human review: Review check 1, Review check 2, "
        "Review check 3, Review check 4, Review check 5, +1 more. "
        "First reason: Reason 1."
    )


def test_invalid_path_is_rejected():
    with pytest.raises(ValidationError, match="path cannot be empty"):
        FileReport(path="")

    with pytest.raises(TypeError, match="path must be str or Path"):
        FileReport(path=123)
