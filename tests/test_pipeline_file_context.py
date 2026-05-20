import json
import sys
import warnings

import pytest

from pipeline_watcher import PipelineReport, Status, file_step, pipeline_file
from pipeline_watcher.core import bind_pipeline


class UnitException(Exception):
    """A single file failed, but the process may continue."""


def test_pipeline_file_success_appends_finalized_file_report_with_steps():
    report = PipelineReport(label="process-report")

    with pipeline_file(
        report,
        "outputs/result.json",
        file_id="result",
        metadata={"source_path": "inputs/source.json"},
    ) as file_report:
        with file_step(file_report, "Load inputs") as step:
            step.note("Using cached input")
        with file_step(file_report, "Render output") as step:
            step.metadata["output_path"] = "outputs/result.json"

    assert len(report.files) == 1
    file_report = report.files[0]
    assert file_report.path.name == "result.json"
    assert file_report.file_id == "result"
    assert file_report.metadata["source_path"] == "inputs/source.json"
    assert file_report.succeeded
    assert file_report.terminal
    assert file_report.percent == 100
    assert file_report.started_at is not None
    assert file_report.finished_at is not None
    assert [step.label for step in file_report.steps] == [
        "Load inputs",
        "Render output",
    ]
    assert file_report.steps[0].notes == ["Using cached input"]
    assert file_report.steps[1].metadata["output_path"] == "outputs/result.json"
    assert report.status is Status.SUCCEEDED


def test_pipeline_file_uses_bound_pipeline_when_report_is_none():
    report = PipelineReport(label="process-report")

    with bind_pipeline(report):
        with pipeline_file(None, "outputs/result.json", file_id="result") as file_report:
            with file_step(file_report, "Validate result"):
                pass

    assert len(report.files) == 1
    assert report.files[0].file_id == "result"
    assert report.files[0].steps[0].label == "Validate result"


def test_pipeline_file_requires_explicit_or_bound_pipeline():
    with pytest.raises(RuntimeError, match="pipeline_file requires a PipelineReport"):
        with pipeline_file(None, "outputs/result.json"):
            pass


def test_pipeline_file_rejects_invalid_path_before_appending_file():
    report = PipelineReport(label="process-report")

    with pytest.raises(ValueError, match="path must be str or os.PathLike"):
        with pipeline_file(report, object()):
            pass

    assert report.files == []


def test_pipeline_file_updates_banner_on_enter_and_exit():
    report = PipelineReport(label="process-report")

    with pipeline_file(
        report,
        "outputs/result.json",
        file_id="result",
        set_stage_on_enter=True,
        banner_stage="process-result",
        banner_percent_on_exit=75,
        banner_message_on_exit="processed result",
    ):
        assert report.stage == "process-result"
        assert report.percent == 0

    assert report.stage == "process-result"
    assert report.percent == 75
    assert report.message == "processed result"


def test_pipeline_file_records_nonfatal_exception_without_reraising():
    report = PipelineReport(label="process-report")

    with pipeline_file(report, "outputs/result.json", file_id="result"):
        raise ValueError("invalid input")

    assert len(report.files) == 1
    file_report = report.files[0]
    assert file_report.failed
    assert "ValueError: invalid input" in file_report.errors
    assert "Unhandled exception while processing file" in file_report.errors
    assert "traceback" in file_report.metadata
    assert report.status is Status.FAILED


def test_pipeline_file_raise_on_exception_reraises_after_recording_file():
    report = PipelineReport(label="process-report")

    with pytest.raises(ValueError, match="invalid input"):
        with pipeline_file(
            report,
            "outputs/result.json",
            file_id="result",
            raise_on_exception=True,
        ):
            raise ValueError("invalid input")

    assert len(report.files) == 1
    file_report = report.files[0]
    assert file_report.failed
    assert "ValueError: invalid input" in file_report.errors


def test_pipeline_file_suppressed_exception_records_breadcrumb_without_reraising():
    report = PipelineReport(label="process-report")

    with pipeline_file(
        report,
        "outputs/result.json",
        file_id="result",
        raise_on_exception=True,
        suppressed_exceptions=(UnitException,),
    ):
        raise UnitException("known fail point")

    file_report = report.files[0]
    assert file_report.failed
    assert "UnitException: known fail point" in file_report.errors
    assert "Handled exception (suppressed)" in file_report.errors
    assert file_report.warnings == [
        "suppressed raise_on_exception for UnitException via suppressed_exceptions",
    ]


def test_pipeline_file_unrelated_exception_reraises_with_raise_on_exception_enabled():
    report = PipelineReport(label="process-report")

    with pytest.raises(ValueError, match="invalid input"):
        with pipeline_file(
            report,
            "outputs/result.json",
            file_id="result",
            raise_on_exception=True,
            suppressed_exceptions=(UnitException,),
        ):
            raise ValueError("invalid input")

    assert len(report.files) == 1
    assert report.files[0].failed


def test_pipeline_file_fatal_exceptions_are_reraised_after_recording_file():
    report = PipelineReport(label="process-report")

    with pytest.raises(SystemExit):
        with pipeline_file(report, "outputs/result.json", file_id="result"):
            raise SystemExit("stop process")

    assert len(report.files) == 1
    assert report.files[0].failed
    assert "SystemExit: stop process" in report.files[0].errors


def test_pipeline_file_can_capture_streams():
    report = PipelineReport(label="process-report")

    with pipeline_file(
        report,
        "outputs/result.json",
        file_id="result",
        capture_streams=True,
    ):
        print("loaded inputs")
        print("warning line", file=sys.stderr)

    file_report = report.files[0]
    assert file_report.metadata["stdout"] == "loaded inputs\n"
    assert file_report.metadata["stderr"] == "warning line\n"


def test_pipeline_file_can_capture_warnings():
    report = PipelineReport(label="process-report")

    with pipeline_file(
        report,
        "outputs/result.json",
        file_id="result",
        capture_warnings=True,
    ):
        warnings.warn("low confidence result", UserWarning)

    assert report.files[0].metadata["warnings"] == [
        "UserWarning: low confidence result",
    ]


def test_pipeline_file_autosaves_pipeline_output_path_on_success(tmp_path):
    output_path = tmp_path / "reports" / "process-report.json"
    report = PipelineReport(label="process-report", output_path=output_path)

    with pipeline_file(report, "outputs/result.json", file_id="result") as file_report:
        with file_step(file_report, "Render output"):
            pass

    data = json.loads(output_path.read_text())
    assert data["label"] == "process-report"
    assert data["files"][0]["file_id"] == "result"
    assert data["files"][0]["status"] == Status.SUCCEEDED
    assert data["files"][0]["steps"][0]["label"] == "Render output"


def test_pipeline_file_writes_finalized_file_snapshot(tmp_path):
    report = PipelineReport(label="process-report")
    file_save_to = tmp_path / "snapshots" / "result-file.json"

    with pipeline_file(
        report,
        "outputs/result.json",
        file_id="result",
        file_save_to=file_save_to,
    ) as file_report:
        with file_step(file_report, "Render output"):
            pass

    data = json.loads(file_save_to.read_text())
    assert data["file_id"] == "result"
    assert data["status"] == Status.SUCCEEDED
    assert data["percent"] == 100
    assert data["finished_at"] is not None
    assert data["steps"][0]["label"] == "Render output"


def test_pipeline_file_writes_additional_pipeline_snapshot(tmp_path):
    report = PipelineReport(label="process-report")
    pipeline_save_to = tmp_path / "snapshots" / "process-report.json"

    with pipeline_file(
        report,
        "outputs/result.json",
        file_id="result",
        pipeline_save_to=pipeline_save_to,
    ):
        pass

    data = json.loads(pipeline_save_to.read_text())
    assert data["label"] == "process-report"
    assert data["files"][0]["file_id"] == "result"


def test_pipeline_file_writes_distinct_exception_snapshot_override(tmp_path):
    output_path = tmp_path / "reports" / "process-report.json"
    exception_path = tmp_path / "exceptions" / "process-report.json"
    report = PipelineReport(label="process-report", output_path=output_path)

    with pipeline_file(
        report,
        "outputs/result.json",
        file_id="result",
        exception_save_path_override=str(exception_path),
    ):
        raise ValueError("invalid input")

    output_data = json.loads(output_path.read_text())
    exception_data = json.loads(exception_path.read_text())
    assert output_data["files"][0]["status"] == Status.FAILED
    assert exception_data["files"][0]["status"] == Status.FAILED
    assert exception_data["files"][0]["errors"][:2] == [
        "ValueError: invalid input",
        "Unhandled exception while processing file",
    ]
