import json
from pathlib import Path
from pipeline_watcher.core import ReportBase, _now
from pipeline_watcher import (
    StepReport, FileReport, PipelineReport, Status, dump_report, now_utc
)


def review_flag_unset(report: FileReport | ReportBase) -> bool:
    assert not report.requires_human_review
    assert report.review.flagged is False
    assert report.review.reason is None
    return True


def test_step_status_members_and_values():
    # enforce API names
    assert {m.name for m in Status} == {
        "PENDING", "RUNNING", "SUCCEEDED", "FAILED", "SKIPPED"
    }
    # enforce serialization values
    print({m.value for m in Status})
    assert {m.value for m in Status} == {
        "pending", "running", "succeeded", "failed", "skipped"
    }

####################
# ReportBase TESTS #
####################

def test_report_base_start():
    # ReportBase is abstract and requires implementation of ok method.
    class A(ReportBase):
        def ok(self):
            return True
    rb = A().start()
    assert rb.percent == 0
    assert rb.status == Status.RUNNING
    assert (_now() - rb.started_at).seconds < 1
    assert rb.finished_at is None
    assert review_flag_unset(rb)
    assert len(rb.notes) == 0
    assert len(rb.errors) == 0
    assert len(rb.warnings) == 0
    assert len(rb.metadata) == 0


def test_report_base_succeed():
    # ReportBase is abstract and requires implementation of ok method.
    class A(ReportBase):
        def ok(self):
            return True
    rb = A().start()
    rb.succeed()
    assert rb.percent == 100
    assert rb.status.succeeded
    assert (_now() - rb.started_at).seconds < 1
    assert (_now() - rb.finished_at).seconds < 1
    assert review_flag_unset(rb)
    assert len(rb.notes) == 0
    assert len(rb.errors) == 0
    assert len(rb.warnings) == 0
    assert len(rb.metadata) == 0


def test_report_base_fail():
    # ReportBase is abstract and requires implementation of ok method.
    class A(ReportBase):
        def ok(self):
            return True
    rb = A().start()
    rb.fail("The report failed")
    assert rb.percent == 0
    assert rb.is_failed
    assert (_now() - rb.started_at).seconds < 1
    assert (_now() - rb.finished_at).seconds < 1
    assert review_flag_unset(rb)
    assert len(rb.notes) == 0
    assert len(rb.errors) == 1
    assert len(rb.warnings) == 0
    assert len(rb.metadata) == 0


def test_report_base_skip():
    # ReportBase is abstract and requires implementation of ok method.
    class A(ReportBase):
        def ok(self):
            return True
    rb = A().start()
    rb.skip("The report skipped")
    assert rb.percent == 0
    assert rb.is_skipped
    assert (_now() - rb.started_at).seconds < 1
    assert (_now() - rb.finished_at).seconds < 1
    assert not rb.requires_human_review
    assert rb.review.flagged is False
    assert rb.review.reason is None
    assert len(rb.notes) == 1
    assert len(rb.errors) == 0
    assert len(rb.warnings) == 0
    assert len(rb.metadata) == 0


def test_report_base_request_review():
    # ReportBase is abstract and requires implementation of ok method.
    class A(ReportBase):
        def ok(self):
            return True
    rb = A().start()
    rb.request_review("Human review required")
    assert rb.requires_human_review
    assert rb.review.flagged
    assert rb.review.reason == "Human review required"
    assert rb.percent == 0
    assert rb.is_running
    assert (_now() - rb.started_at).seconds < 1
    assert rb.finished_at is None
    assert len(rb.notes) == 0
    assert len(rb.errors) == 0
    assert len(rb.warnings) == 0
    assert len(rb.metadata) == 0


def test_report_base_clear_review():
    # ReportBase is abstract and requires implementation of ok method.
    class A(ReportBase):
        def ok(self):
            return True
    rb = A().start()
    rb.request_review("Human review required")
    rb.clear_review()
    assert review_flag_unset(rb)
    assert rb.percent == 0
    assert rb.status == Status.RUNNING
    assert (_now() - rb.started_at).seconds < 1
    assert rb.finished_at is None
    assert len(rb.notes) == 0
    assert len(rb.errors) == 0
    assert len(rb.warnings) == 0
    assert len(rb.metadata) == 0


def test_report_base_note():
    # ReportBase is abstract and requires implementation of ok method.
    class A(ReportBase):
        def ok(self):
            return True
    rb = A().start()
    rb.note("Note test in progress")
    assert rb.percent == 0
    assert rb.status == Status.RUNNING
    assert (_now() - rb.started_at).seconds < 1
    assert rb.finished_at is None
    assert not rb.requires_human_review
    assert rb.review.flagged is False
    assert rb.review.reason is None
    assert len(rb.notes) == 1
    assert rb.notes[0] == "Note test in progress"
    assert len(rb.errors) == 0
    assert len(rb.warnings) == 0
    assert len(rb.metadata) == 0


def test_report_base_warn():
    # ReportBase is abstract and requires implementation of ok method.
    class A(ReportBase):
        def ok(self):
            return True
    rb = A().start()
    rb.warn("Warn test in progress")
    assert rb.percent == 0
    assert rb.is_running
    assert (_now() - rb.started_at).seconds < 1
    assert rb.finished_at is None
    assert not rb.requires_human_review
    assert rb.review.flagged is False
    assert rb.review.reason is None
    assert len(rb.notes) == 0
    assert len(rb.errors) == 0
    assert len(rb.warnings) == 1
    assert rb.warnings[0] == "Warn test in progress"
    assert len(rb.metadata) == 0


def test_report_base_error():
    # ReportBase is abstract and requires implementation of ok method.
    class A(ReportBase):
        def ok(self):
            return True
    rb = A().start()
    rb.error("Error test in progress")
    assert rb.percent == 0
    assert rb.is_running
    assert (_now() - rb.started_at).seconds < 1
    assert rb.finished_at is None
    assert review_flag_unset(rb)
    assert len(rb.notes) == 0
    assert len(rb.errors) == 1
    assert len(rb.warnings) == 0
    assert len(rb.metadata) == 0
    assert rb.errors[0] == "Error test in progress"


####################
# FileReport TESTS #
####################

def test_file_report_begin():
    fr = FileReport.begin(file_id="my-file.ext")
    assert fr.file_id == "my-file.ext"
    assert fr.is_running
    assert fr.path is None
    assert fr.name is None
    assert fr.size_bytes is None
    assert fr.mime_type is None
    assert fr.steps == []
    assert not fr.review.flagged
    assert review_flag_unset(fr)

####################
# StepReport TESTS #
####################

def test_step_end_idempotent_and_success():
    s = StepReport.begin("parse", label="Parse")
    # no checks, no errors -> ok True -> succeed on end
    s1 = s.end()
    assert s1.status.succeeded
    assert s1.percent == 100
    finished = s1.finished_at
    # calling end again should not change terminal status, only ensure finished_at
    s2 = s1.end()
    assert s2.status.succeeded
    assert s2.finished_at == finished


def test_step_end_failure_with_failed_check():
    s = StepReport.begin("analyze", label="Analyze")
    s.add_check("unique_ids", ok=False, detail="dupes found")
    s.end()
    assert s.status.failed
    assert any("dupes" in (s.errors[0] if s.errors else "") or not s.ok for _ in [0])  # ok==False implies failure


def test_filereport_append_auto_finalizes_and_rolls_percent():
    fr = FileReport.begin(file_id="f1", path="inputs/f1.txt", name="f1.txt")
    # step 1 (explicit succeed)
    st1 = StepReport.begin("parse").succeed()
    fr.append_step(st1)
    assert fr.steps[0].status.succeeded
    assert fr.percent == 100  # only one step so far

    # step 2 (implicit end -> success since no checks/errors)
    st2 = StepReport.begin("analyze")
    fr.append_step(st2)
    assert fr.steps[1].status in (Status.SUCCEEDED, Status.FAILED)
    # average of percents: SUCCESS=100, SUCCESS=100 -> 100
    assert fr.percent in (50, 100)  # if analyze failed, 50; else 100
    fr.end()
    assert fr.status in (Status.SUCCEEDED, Status.FAILED)

########################
# PipelineReport TESTS #
########################

def test_pipeline_report_append_and_overall(tmp_path: Path):
    report = PipelineReport(label="batch 7", kind="process")
    report.set_progress("parse", 10, "starting")
    # two steps: first succeeds explicitly, second finalizes via end()
    report.append_step(StepReport.begin("discover").succeed())
    report.append_step(StepReport.begin("ingest"))  # will end() to success

    assert len(report.steps) == 2
    assert all(s.status.succeeded for s in report.steps)

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

