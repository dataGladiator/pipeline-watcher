import pytest
from pydantic import ValidationError

from pipeline_watcher import Check, Status, StepReport


def test_begin_sets_label_id_and_running_lifecycle():
    step = StepReport.begin("Load inputs")

    assert step.label == "Load inputs"
    assert step.id == "load-inputs"
    assert step.running
    assert step.percent == 0
    assert step.started_at is not None
    assert step.finished_at is None
    assert step.checks == []
    assert step.notes == []
    assert step.errors == []
    assert step.warnings == []
    assert step.metadata == {}
    assert step.report_version == "v2"


def test_begin_preserves_explicit_id_and_empty_id_uses_label_slug():
    explicit = StepReport.begin("Load inputs", id="load")
    empty = StepReport.begin("Load inputs", id="")

    assert explicit.id == "load"
    assert empty.id == "load-inputs"


def test_constructor_auto_starts_unless_deferred():
    auto_started = StepReport(label="Compute result")
    deferred = StepReport(label="Compute result", defer_start=True)

    assert auto_started.running
    assert auto_started.started_at is not None

    assert deferred.pending
    assert deferred.started_at is None
    assert deferred.finished_at is None
    assert deferred.duration_ms is None


def test_lifecycle_methods_are_chainable_and_update_terminal_state():
    step = StepReport.begin("Compute result")

    assert step.note("Using cached input") is step
    assert step.warn("Cache is stale") is step
    assert step.error("Recoverable issue") is step

    assert step.notes == ["Using cached input"]
    assert step.warnings == ["Cache is stale"]
    assert step.errors == ["Recoverable issue"]
    assert step.running

    assert step.fail("Computation failed") is step

    assert step.failed
    assert step.terminal
    assert step.errors == ["Recoverable issue", "Computation failed"]
    assert step.finished_at is not None


def test_succeed_sets_terminal_success_and_percent():
    step = StepReport.begin("Render output")

    assert step.succeed() is step

    assert step.succeeded
    assert step.terminal
    assert step.percent == 100
    assert step.finished_at is not None


def test_skip_sets_terminal_status_and_records_reason():
    step = StepReport.begin("Render output")

    assert step.skip("output already exists") is step

    assert step.skipped
    assert step.terminal
    assert step.notes == ["Skipped: output already exists"]
    assert step.finished_at is not None


def test_end_succeeds_when_checks_pass_or_no_checks_exist():
    no_checks = StepReport.begin("Load inputs")
    passing_checks = StepReport.begin("Validate result")
    passing_checks.add_check("manifest_present", ok=True)
    passing_checks.add_check("ids_unique", ok=True)

    no_checks.end()
    passing_checks.end()

    assert no_checks.succeeded
    assert passing_checks.succeeded
    assert no_checks.percent == 100
    assert passing_checks.percent == 100


def test_failed_checks_make_ok_false_and_surface_check_details():
    step = StepReport.begin("Validate result")
    step.add_check("manifest_present", ok=True)
    step.add_check("ids_unique", ok=False, detail="3 duplicate ids")
    step.add_check("schema_valid", ok=False)

    assert not step.ok
    assert step.errors == ["3 duplicate ids", "Unknown check fail."]

    step.end()

    assert step.failed
    assert step.errors == [
        "3 duplicate ids",
        "Unknown check fail.",
        "One or more file steps failed",
    ]
    assert step.percent == 0


def test_explicit_status_and_errors_take_precedence_in_ok():
    failed = StepReport.begin("Compute result")
    failed.fail("boom")
    succeeded = StepReport.begin("Validate result")
    succeeded.add_check("quality", ok=False, detail="low quality")
    succeeded.succeed()

    assert not failed.ok
    assert succeeded.ok
    assert succeeded.errors == []


def test_add_check_appends_check_models_in_order():
    step = StepReport.begin("Validate result")

    assert step.add_check("manifest_present", ok=True) is None
    assert step.add_check("ids_unique", ok=False, detail="duplicate ids") is None

    assert step.checks == [
        Check(name="manifest_present", ok=True, detail=None),
        Check(name="ids_unique", ok=False, detail="duplicate ids"),
    ]


def test_review_helpers_flag_and_clear_review():
    step = StepReport.begin("Validate result")

    assert step.requires_human_review is False
    assert step.request_review("Confidence below threshold") is step
    assert step.requires_human_review is True
    assert step.review.flagged
    assert step.review.reason == "Confidence below threshold"

    assert step.clear_review() is step
    assert step.requires_human_review is False
    assert step.review.flagged is False
    assert step.review.reason is None


def test_duration_ms_is_available_after_start_and_terminal_state():
    step = StepReport.begin("Compute result")

    assert step.duration_ms is not None
    assert step.duration_ms >= 0

    step.succeed()

    assert step.duration_ms is not None
    assert step.duration_ms >= 0


def test_model_dump_contains_json_friendly_step_fields_and_excludes_defer_start():
    step = StepReport.begin("Validate result", id="validate")
    step.add_check("manifest_present", ok=True)
    step.note("Validated manifest")
    step.metadata["record_count"] = 3
    step.succeed()

    data = step.model_dump(mode="json")

    assert data["label"] == "Validate result"
    assert data["id"] == "validate"
    assert data["status"] == Status.SUCCEEDED
    assert data["percent"] == 100
    assert data["checks"] == [
        {"name": "manifest_present", "ok": True, "detail": None},
    ]
    assert data["notes"] == ["Validated manifest"]
    assert data["metadata"] == {"record_count": 3}
    assert data["review"] == {"flagged": False, "reason": None}
    assert "duration_ms" in data
    assert "defer_start" not in data


@pytest.mark.parametrize("label", [123, None])
def test_invalid_label_is_rejected(label):
    with pytest.raises(ValidationError):
        StepReport(label=label)
