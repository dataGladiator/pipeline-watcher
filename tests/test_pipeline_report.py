import json
from pathlib import Path

from pipeline_watcher import FileReport, PipelineReport, Status, StepReport


def _step(
    label: str,
    status: Status = Status.SUCCEEDED,
    *,
    percent: int | None = None,
) -> StepReport:
    if status is Status.PENDING:
        step = StepReport(label=label, defer_start=True)
    else:
        step = StepReport.begin(label)

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


def _file(
    name: str,
    status: Status = Status.SUCCEEDED,
    *,
    file_id: str | None = None,
    percent: int | None = None,
    review: bool = False,
) -> FileReport:
    path = Path("outputs") / name
    if status is Status.PENDING:
        file_report = FileReport(path=path, file_id=file_id, defer_start=True)
    else:
        file_report = FileReport.begin(path=path, file_id=file_id)

    if status is Status.SUCCEEDED:
        file_report.succeed()
    elif status is Status.FAILED:
        file_report.fail("file failed")
    elif status is Status.SKIPPED:
        file_report.skip("not needed")
    elif status is Status.RUNNING:
        pass

    if percent is not None:
        file_report.percent = percent
    if review:
        file_report.request_review("Needs review")
    return file_report


def test_pipeline_report_defaults():
    report = PipelineReport(label="process-report")

    assert report.label == "process-report"
    assert report.kind == "process"
    assert report.percent == 0
    assert report.stage == ""
    assert report.message == ""
    assert report.metadata == {}
    assert report.report_version == "v2"
    assert report.updated_at is not None
    assert report.steps == []
    assert report.files == []
    assert report.status is Status.PENDING


def test_set_progress_updates_banner_and_clamps_percent():
    report = PipelineReport(label="process-report")
    before = report.updated_at

    report.set_progress("discover", -5, "scanning")

    assert report.stage == "discover"
    assert report.percent == 0
    assert report.message == "scanning"
    assert report.updated_at >= before

    report.set_progress("render", 125, "rendering")

    assert report.stage == "render"
    assert report.percent == 100
    assert report.message == "rendering"


def test_add_completed_step_appends_succeeded_terminal_step_with_label_id():
    report = PipelineReport(label="process-report")

    step = report.add_completed_step("Load inputs")

    assert step is report.steps[0]
    assert step.label == "Load inputs"
    assert step.id == "load-inputs"
    assert step.succeeded
    assert step.terminal


def test_add_completed_step_deduplicates_repeated_label_ids():
    report = PipelineReport(label="process-report")

    report.add_completed_step("Load inputs")
    report.add_completed_step("Load inputs")
    report.add_completed_step("Load inputs")

    assert [step.id for step in report.steps] == [
        "load-inputs",
        "load-inputs-2",
        "load-inputs-3",
    ]


def test_append_step_finalizes_running_step_and_updates_timestamp():
    report = PipelineReport(label="process-report")
    before = report.updated_at
    step = StepReport.begin("Compute result")

    report.append_step(step)

    assert report.steps == [step]
    assert step.succeeded
    assert step.terminal
    assert report.updated_at >= before


def test_append_step_preserves_failed_step_and_rolls_pipeline_failed():
    report = PipelineReport(label="process-report")
    failed = _step("Compute result", Status.FAILED)

    report.append_step(failed)

    assert report.steps == [failed]
    assert report.steps[0].failed
    assert report.status is Status.FAILED


def test_iter_steps_filters_by_status():
    report = PipelineReport(label="process-report")
    succeeded = _step("Load inputs", Status.SUCCEEDED)
    failed = _step("Compute result", Status.FAILED)
    skipped = _step("Render output", Status.SKIPPED)
    report.steps.extend([succeeded, failed, skipped])

    assert list(report.iter_steps()) == [succeeded, failed, skipped]
    assert list(report.iter_steps(status=Status.SUCCEEDED)) == [succeeded]
    assert list(report.iter_steps(status=Status.FAILED)) == [failed]
    assert list(report.iter_steps(status=Status.SKIPPED)) == [skipped]


def test_last_step_returns_none_then_latest_step():
    report = PipelineReport(label="process-report")
    first = _step("Load inputs")
    second = _step("Compute result")

    assert report.last_step() is None

    report.steps.extend([first, second])

    assert report.last_step() is second


def test_recompute_overall_from_steps_averages_step_percents():
    report = PipelineReport(label="process-report")
    report.set_progress("existing-stage", 0, "existing message")
    report.steps.extend([
        _step("Load inputs", percent=25),
        _step("Compute result", percent=50),
        _step("Render output", percent=100),
    ])

    report.recompute_overall_from_steps()

    assert report.percent == 58
    assert report.stage == "existing-stage"
    assert report.message == "existing message"


def test_append_file_finalizes_file_and_updates_timestamp():
    report = PipelineReport(label="process-report")
    before = report.updated_at
    file_report = FileReport.begin(path="outputs/result.json", file_id="result")

    report.append_file(file_report)

    assert report.files == [file_report]
    assert file_report.succeeded
    assert file_report.terminal
    assert report.updated_at >= before


def test_file_lookup_matches_file_id_path_basename_and_normalized_key():
    report = PipelineReport(label="process-report")
    file_report = _file("Result File.json", file_id="result-1")
    report.files.append(file_report)

    assert report.files_for("result-1") == [file_report]
    assert report.files_for("outputs/Result File.json") == [file_report]
    assert report.files_for("Result File.json") == [file_report]
    assert report.files_for("  result file.json ") == [file_report]
    assert report.get_file("result-1") is file_report
    assert report.file_seen("Result File.json")
    assert not report.file_seen("missing.json")


def test_file_processed_default_accepts_terminal_statuses():
    report = PipelineReport(label="process-report")
    report.files.extend([
        _file("succeeded.json", Status.SUCCEEDED, file_id="succeeded"),
        _file("failed.json", Status.FAILED, file_id="failed"),
        _file("skipped.json", Status.SKIPPED, file_id="skipped"),
    ])

    assert report.file_processed("succeeded")
    assert report.file_processed("failed")
    assert report.file_processed("skipped")


def test_file_processed_require_success_only_accepts_succeeded():
    report = PipelineReport(label="process-report")
    report.files.extend([
        _file("succeeded.json", Status.SUCCEEDED, file_id="succeeded"),
        _file("failed.json", Status.FAILED, file_id="failed"),
        _file("skipped.json", Status.SKIPPED, file_id="skipped"),
        _file("pending.json", Status.PENDING, file_id="pending"),
    ])

    assert report.file_processed("succeeded", require_success=True)
    assert not report.file_processed("failed", require_success=True)
    assert not report.file_processed("skipped", require_success=True)
    assert not report.file_processed("pending", require_success=True)


def test_unseen_expected_returns_missing_keys():
    report = PipelineReport(label="process-report")
    report.files.extend([
        _file("one.json", file_id="one"),
        _file("two.json", file_id="two"),
    ])

    assert report.unseen_expected(["one", "two.json", "missing.json"]) == ["missing.json"]


def test_status_is_pending_with_no_units():
    assert PipelineReport(label="process-report").status is Status.PENDING


def test_status_is_failed_if_any_step_or_file_failed():
    report = PipelineReport(label="process-report")
    report.steps.append(_step("Load inputs", Status.SUCCEEDED))
    report.files.append(_file("failed.json", Status.FAILED))

    assert report.status is Status.FAILED


def test_status_is_running_if_any_unit_running_and_none_failed():
    report = PipelineReport(label="process-report")
    report.steps.append(_step("Load inputs", Status.SUCCEEDED))
    report.files.append(_file("running.json", Status.RUNNING))

    assert report.status is Status.RUNNING


def test_status_is_skipped_if_all_units_skipped():
    report = PipelineReport(label="process-report")
    report.steps.append(_step("Load inputs", Status.SKIPPED))
    report.files.append(_file("skipped.json", Status.SKIPPED))

    assert report.status is Status.SKIPPED


def test_status_is_succeeded_if_terminal_with_success_and_no_failures():
    report = PipelineReport(label="process-report")
    report.steps.append(_step("Load inputs", Status.SUCCEEDED))
    report.files.append(_file("skipped.json", Status.SKIPPED))

    assert report.status is Status.SUCCEEDED


def test_status_is_pending_for_nonterminal_pending_units():
    report = PipelineReport(label="process-report")
    report.steps.append(_step("Load inputs", Status.PENDING))

    assert report.status is Status.PENDING


def test_table_rows_for_files_map_includes_seen_and_missing_rows():
    report = PipelineReport(label="process-report")
    seen = _file(
        "result.json",
        Status.SUCCEEDED,
        file_id="result",
        percent=75,
        review=True,
    )
    report.files.append(seen)

    rows = report.table_rows_for_files_map({
        "result": {"expected": True},
        "missing.json": {"expected": False},
    })

    seen_row, missing_row = rows
    assert seen_row == {
        "filename": "result",
        "seen": True,
        "status": Status.SUCCEEDED,
        "percent": 75,
        "flagged_human_review": True,
        "human_review_reason": "File-level review: Needs review",
        "file_id": "result",
        "path": Path("outputs/result.json"),
        "other": {"expected": True},
    }
    assert missing_row == {
        "filename": "missing.json",
        "seen": False,
        "status": "MISSING",
        "percent": None,
        "flagged_human_review": False,
        "human_review_reason": "",
        "file_id": None,
        "path": None,
        "other": {"expected": False},
    }


def test_save_writes_valid_json_to_explicit_path(tmp_path: Path):
    report = PipelineReport(label="process-report", kind="validation")
    report.metadata["process_dir"] = "/tmp/process"
    report.append_step(_step("Load inputs"))
    report.append_file(_file("result.json", file_id="result"))

    output_path = tmp_path / "reports" / "progress.json"
    report.save(output_path)

    data = json.loads(output_path.read_text())
    assert data["label"] == "process-report"
    assert data["kind"] == "validation"
    assert data["report_version"] == "v2"
    assert data["metadata"] == {"process_dir": "/tmp/process"}
    assert data["steps"][0]["label"] == "Load inputs"
    assert data["files"][0]["file_id"] == "result"


def test_save_uses_output_path_when_no_path_is_passed(tmp_path: Path):
    output_path = tmp_path / "progress.json"
    report = PipelineReport(label="process-report", output_path=output_path)

    report.save()

    data = json.loads(output_path.read_text())
    assert data["label"] == "process-report"
