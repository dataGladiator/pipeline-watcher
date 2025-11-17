# test_pipeline_report.py
import json
from pathlib import Path
from pipeline_watcher import PipelineReport, FileReport, StepReport, Status, dump_report


def test_pipeline_report_add_step():
    report = PipelineReport(label="label")
    assert report.updated_at is not None
    report.add_completed_step("Test Label")
    assert report.steps[0].label == "Test Label"
    assert report.steps[0].id == 'test-label'
    assert report.steps[0].terminal
    assert report.steps[0].succeeded

def test_pipeline_report_append_file():
    report = PipelineReport(label="label")
    fr = FileReport(path=Path("path/to/file"), file_id="file-id")
    report.append_file(fr)
    assert report.files[0].succeeded
    assert report.files[0].terminal

def test_pipeline_report_append_step():
    report = PipelineReport(label="label")
    step = StepReport(label="Test Label")
    report.append_step(step)
    assert report.steps[0].label == "Test Label"
    assert report.steps[0].id == 'test-label'
    assert report.steps[0].terminal
    assert report.steps[0].succeeded
    step = StepReport(label="Test Label")
    step.error("Encountered an error.")
    report.append_step(step)
    assert report.steps[1].failed
    assert report.steps[1].id == 'test-label-1'



def test_pipeline_report_append_and_overall(tmp_path: Path):
    report = PipelineReport(label="batch 7", kind="process")
    report.set_progress("parse", 10, "starting")
    # two steps: first succeeds explicitly, second finalizes via end()
    report.append_step(StepReport.begin("discover").succeed())
    report.append_step(StepReport.begin("ingest"))  # will end() to success

    assert len(report.steps) == 2
    assert all(s.succeeded for s in report.steps)

    report.recompute_overall_from_steps()
    assert 0 <= report.percent <= 100

    out = tmp_path / "progress.json"
    dump_report(out, report)
    data = json.loads(out.read_text())
    assert data["label"] == "batch 7"
    assert data["report_version"] == "v2"
    assert isinstance(data["steps"], list)


def test_append_file_auto_finalizes():
    report = PipelineReport(label="batch 1", kind="validation")
    fr = FileReport.begin(file_id="x")
    fr.append_step(StepReport.begin("validate"))
    report.append_file(fr)
    assert report.files[0].status in (Status.SUCCEEDED, Status.FAILED)

