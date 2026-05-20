import json
import warnings

import pytest

from pipeline_watcher import PipelineReport, Status, pipeline_step
from pipeline_watcher.core import bind_pipeline


class UnitException(Exception):
    """A single unit failed, but the process may continue."""


def test_pipeline_step_success_appends_finalized_step_with_runtime_details():
    report = PipelineReport(label="process-report")

    with pipeline_step(report, "Initialize orchestrator") as step:
        step.note("Loaded process plan")
        step.metadata["item_count"] = 3

    assert len(report.steps) == 1
    step = report.steps[0]
    assert step.label == "Initialize orchestrator"
    assert step.id == "initialize-orchestrator"
    assert step.succeeded
    assert step.terminal
    assert step.percent == 100
    assert step.started_at is not None
    assert step.finished_at is not None
    assert step.notes == ["Loaded process plan"]
    assert step.metadata["item_count"] == 3
    assert report.status is Status.SUCCEEDED


def test_pipeline_step_deduplicates_label_derived_ids():
    report = PipelineReport(label="process-report")

    with pipeline_step(report, "Load static configuration"):
        pass
    with pipeline_step(report, "Load static configuration"):
        pass

    assert [step.id for step in report.steps] == [
        "load-static-configuration",
        "load-static-configuration-1",
    ]


def test_pipeline_step_uses_bound_pipeline_when_report_is_none():
    report = PipelineReport(label="process-report")

    with bind_pipeline(report):
        with pipeline_step(None, "Discover inputs") as step:
            step.note("Scanning directory")

    assert len(report.steps) == 1
    assert report.steps[0].label == "Discover inputs"
    assert report.steps[0].notes == ["Scanning directory"]


def test_pipeline_step_requires_explicit_or_bound_pipeline():
    with pytest.raises(RuntimeError, match="pipeline_step requires a PipelineReport"):
        with pipeline_step(None, "Discover inputs"):
            pass


def test_pipeline_step_records_nonfatal_exception_without_reraising():
    report = PipelineReport(label="process-report")

    with pipeline_step(report, "Validate batch"):
        raise ValueError("manifest missing")

    assert len(report.steps) == 1
    step = report.steps[0]
    assert step.failed
    assert "ValueError: manifest missing" in step.errors
    assert "Unhandled ValueError in pipeline step" in step.errors
    assert "traceback" in step.metadata
    assert "auto-save skipped: no output path configured" in step.warnings
    assert report.status is Status.FAILED


def test_pipeline_step_raise_on_exception_reraises_after_recording_step():
    report = PipelineReport(label="process-report")

    with pytest.raises(ValueError, match="manifest missing"):
        with pipeline_step(report, "Validate batch", raise_on_exception=True):
            raise ValueError("manifest missing")

    assert len(report.steps) == 1
    assert report.steps[0].failed
    assert "ValueError: manifest missing" in report.steps[0].errors


def test_pipeline_step_suppressed_exception_records_breadcrumb_without_reraising():
    report = PipelineReport(label="process-report")

    with pipeline_step(
        report,
        "Compute result",
        raise_on_exception=True,
        suppressed_exceptions=(UnitException,),
    ):
        raise UnitException("known fail point")

    step = report.steps[0]
    assert step.failed
    assert "UnitException: known fail point" in step.errors
    assert step.warnings == [
        "suppressed raise_on_exception for UnitException via suppressed_exceptions",
        "auto-save skipped: no output path configured",
    ]


def test_pipeline_step_fatal_exceptions_are_reraised_after_recording_step():
    report = PipelineReport(label="process-report")

    with pytest.raises(SystemExit):
        with pipeline_step(report, "Validate batch"):
            raise SystemExit("stop process")

    assert len(report.steps) == 1
    assert report.steps[0].failed
    assert "SystemExit: stop process" in report.steps[0].errors


def test_pipeline_step_can_capture_streams():
    report = PipelineReport(label="process-report")

    with pipeline_step(report, "Initialize orchestrator", capture_streams=True):
        print("loaded plan")
        print("warning line", file=__import__("sys").stderr)

    step = report.steps[0]
    assert step.metadata["stdout"] == "loaded plan\n"
    assert step.metadata["stderr"] == "warning line\n"


def test_pipeline_step_can_capture_warnings():
    report = PipelineReport(label="process-report")

    with pipeline_step(report, "Validate batch", capture_warnings=True):
        warnings.warn("deprecated input shape", UserWarning)

    assert report.steps[0].metadata["warnings"] == [
        "UserWarning: deprecated input shape",
    ]


def test_pipeline_step_saves_pipeline_json_on_exception_when_output_path_is_set(
    tmp_path,
):
    output_path = tmp_path / "reports" / "process-report.json"
    report = PipelineReport(label="process-report", output_path=output_path)

    with pipeline_step(report, "Validate batch"):
        raise ValueError("manifest missing")

    data = json.loads(output_path.read_text())
    assert data["label"] == "process-report"
    assert data["steps"][0]["label"] == "Validate batch"
    assert data["steps"][0]["status"] == Status.FAILED
    assert data["steps"][0]["errors"][:2] == [
        "ValueError: manifest missing",
        "Unhandled ValueError in pipeline step",
    ]
