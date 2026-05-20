import json
import warnings

import pytest

from pipeline_watcher import FileReport, PipelineReport, Status, file_step


class UnitException(Exception):
    """A single file unit failed, but the process may continue."""


def _attached_file_report(report: PipelineReport) -> FileReport:
    file_report = FileReport.begin("outputs/result.json", file_id="result")
    file_report._pipeline = report
    report.files.append(file_report)
    return file_report


def test_file_step_success_appends_finalized_step_with_runtime_details():
    file_report = FileReport.begin("outputs/result.json", file_id="result")

    with file_step(file_report, "Load inputs") as step:
        step.note("Using cached input")
        step.metadata["source_path"] = "inputs/source.json"

    assert len(file_report.steps) == 1
    step = file_report.steps[0]
    assert step.label == "Load inputs"
    assert step.id == "load-inputs"
    assert step.succeeded
    assert step.terminal
    assert step.percent == 100
    assert step.started_at is not None
    assert step.finished_at is not None
    assert step.notes == ["Using cached input"]
    assert step.metadata["source_path"] == "inputs/source.json"
    assert file_report.percent == 100


def test_file_step_recomputes_file_percent_from_appended_terminal_steps():
    file_report = FileReport.begin("outputs/result.json", file_id="result")

    with file_step(file_report, "Load inputs"):
        pass

    with file_step(file_report, "Compute result") as step:
        step.note("Computed result")

    assert [step.terminal for step in file_report.steps] == [True, True]
    assert file_report.percent == 100


def test_file_step_failed_checks_make_step_fail_when_finalized():
    file_report = FileReport.begin("outputs/result.json", file_id="result")

    with file_step(file_report, "Validate result") as step:
        step.add_check("manifest_present", ok=True)
        step.add_check("ids_unique", ok=False, detail="3 duplicate ids")

    step = file_report.steps[0]
    assert step.failed
    assert step.errors == [
        "3 duplicate ids",
        "One or more file steps failed",
    ]


def test_file_step_review_request_rolls_up_to_file_report():
    file_report = FileReport.begin("outputs/result.json", file_id="result")

    with file_step(file_report, "Validate result") as step:
        step.metadata["confidence"] = 0.82
        step.request_review("Result confidence below threshold: 0.82")

    assert file_report.review.flagged
    assert file_report.review.reason == "Result confidence below threshold: 0.82"
    assert file_report.requires_human_review
    assert "Validate result" in file_report.human_review_reason


def test_file_step_records_nonfatal_exception_without_reraising():
    file_report = FileReport.begin("outputs/result.json", file_id="result")

    with file_step(file_report, "Compute result"):
        raise ValueError("invalid input")

    assert len(file_report.steps) == 1
    step = file_report.steps[0]
    assert step.failed
    assert "ValueError: invalid input" in step.errors
    assert "Unhandled exception while running file step" in step.errors
    assert "traceback" in step.metadata


def test_file_step_raise_on_exception_reraises_after_recording_step():
    file_report = FileReport.begin("outputs/result.json", file_id="result")

    with pytest.raises(ValueError, match="invalid input"):
        with file_step(file_report, "Compute result", raise_on_exception=True):
            raise ValueError("invalid input")

    assert len(file_report.steps) == 1
    assert file_report.steps[0].failed
    assert "ValueError: invalid input" in file_report.steps[0].errors


def test_file_step_suppressed_exception_records_breadcrumb_without_reraising():
    file_report = FileReport.begin("outputs/result.json", file_id="result")

    with file_step(
        file_report,
        "Compute result",
        raise_on_exception=True,
        suppressed_exceptions=(UnitException,),
    ):
        raise UnitException("known fail point")

    step = file_report.steps[0]
    assert step.failed
    assert "UnitException: known fail point" in step.errors
    assert step.warnings == [
        "suppressed raise_on_exception for UnitException via suppressed_exceptions",
    ]


def test_file_step_unrelated_exception_reraises_when_raise_on_exception_is_enabled():
    file_report = FileReport.begin("outputs/result.json", file_id="result")

    with pytest.raises(ValueError, match="invalid input"):
        with file_step(
            file_report,
            "Compute result",
            raise_on_exception=True,
            suppressed_exceptions=(UnitException,),
        ):
            raise ValueError("invalid input")

    assert len(file_report.steps) == 1
    assert file_report.steps[0].failed


def test_file_step_can_capture_streams():
    file_report = FileReport.begin("outputs/result.json", file_id="result")

    with file_step(file_report, "Load inputs", capture_streams=True):
        print("loaded inputs")
        print("warning line", file=__import__("sys").stderr)

    step = file_report.steps[0]
    assert step.metadata["stdout"] == "loaded inputs\n"
    assert step.metadata["stderr"] == "warning line\n"


def test_file_step_can_capture_warnings():
    file_report = FileReport.begin("outputs/result.json", file_id="result")

    with file_step(file_report, "Validate result", capture_warnings=True):
        warnings.warn("low confidence result", UserWarning)

    assert file_report.steps[0].metadata["warnings"] == [
        "UserWarning: low confidence result",
    ]


def test_file_step_autosaves_attached_pipeline_output_path(tmp_path):
    output_path = tmp_path / "reports" / "process-report.json"
    report = PipelineReport(label="process-report", output_path=output_path)
    file_report = _attached_file_report(report)

    with file_step(file_report, "Render output") as step:
        step.metadata["output_path"] = "outputs/result.json"

    data = json.loads(output_path.read_text())
    assert data["files"][0]["file_id"] == "result"
    assert data["files"][0]["steps"][0]["label"] == "Render output"
    assert data["files"][0]["steps"][0]["status"] == Status.SUCCEEDED


def test_file_step_writes_finalized_step_snapshot(tmp_path):
    file_report = FileReport.begin("outputs/result.json", file_id="result")
    step_save_to = tmp_path / "snapshots" / "render-output.json"

    with file_step(file_report, "Render output", step_save_to=step_save_to):
        pass

    data = json.loads(step_save_to.read_text())
    assert data["label"] == "Render output"
    assert data["id"] == "render-output"
    assert data["status"] == Status.SUCCEEDED
    assert data["percent"] == 100
    assert data["finished_at"] is not None


def test_file_step_writes_additional_pipeline_snapshot_for_attached_file(tmp_path):
    report = PipelineReport(label="process-report")
    file_report = _attached_file_report(report)
    pipeline_save_to = tmp_path / "snapshots" / "process-report.json"

    with file_step(file_report, "Render output", pipeline_save_to=pipeline_save_to):
        pass

    data = json.loads(pipeline_save_to.read_text())
    assert data["label"] == "process-report"
    assert data["files"][0]["steps"][0]["label"] == "Render output"
