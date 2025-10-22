import json
from pathlib import Path
from pipeline_watcher import (
    StepReport, FileReport, PipelineReport, StepStatus, dump_report, now_utc
)

def test_step_end_idempotent_and_success():
    s = StepReport.begin("parse", label="Parse")
    # no checks, no errors -> ok True -> succeed on end
    s1 = s.end()
    assert s1.status == StepStatus.SUCCESS
    assert s1.percent == 100
    finished = s1.finished_at
    # calling end again should not change terminal status, only ensure finished_at
    s2 = s1.end()
    assert s2.status == StepStatus.SUCCESS
    assert s2.finished_at == finished


def test_step_end_failure_with_failed_check():
    s = StepReport.begin("analyze", label="Analyze")
    s.add_check("unique_ids", ok=False, detail="dupes found")
    s.end()
    assert s.status == StepStatus.FAILED
    assert any("dupes" in (s.errors[0] if s.errors else "") or not s.ok for _ in [0])  # ok==False implies failure


def test_filereport_append_auto_finalizes_and_rolls_percent():
    fr = FileReport.begin(file_id="f1", path="inputs/f1.txt", name="f1.txt")
    # step 1 (explicit succeed)
    st1 = StepReport.begin("parse").succeed()
    fr.append_step(st1)
    assert fr.steps[0].status == StepStatus.SUCCESS
    assert fr.percent == 100  # only one step so far

    # step 2 (implicit end -> success since no checks/errors)
    st2 = StepReport.begin("analyze")
    fr.append_step(st2)
    assert fr.steps[1].status in (StepStatus.SUCCESS, StepStatus.FAILED)
    # average of percents: SUCCESS=100, SUCCESS=100 -> 100
    assert fr.percent in (50, 100)  # if analyze failed, 50; else 100
    fr.end()
    assert fr.status in (StepStatus.SUCCESS, StepStatus.FAILED)


def test_pipelinereport_append_and_overall(tmp_path: Path):
    report = PipelineReport(batch_id=7, kind="process")
    report.set_progress("parse", 10, "starting")
    # two steps: first succeeds explicitly, second finalizes via end()
    report.append_step(StepReport.begin("discover").succeed())
    report.append_step(StepReport.begin("ingest"))  # will end() to success

    assert len(report.steps) == 2
    assert all(s.status == StepStatus.SUCCESS for s in report.steps)

    report.recompute_overall_from_steps()
    assert 0 <= report.percent <= 100

    out = tmp_path / "progress.json"
    dump_report(out, report)
    data = json.loads(out.read_text())
    assert data["batch_id"] == 7
    assert data["report_version"] == "v2"
    assert isinstance(data["steps"], list)


def test_append_file_auto_finalizes():
    report = PipelineReport(batch_id=1, kind="validation")
    fr = FileReport.begin(file_id="x")
    fr.append_step(StepReport.begin("validate"))
    report.append_file(fr)
    assert report.files[0].status in (StepStatus.SUCCESS, StepStatus.FAILED)

