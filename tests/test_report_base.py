from pathlib import Path
from pipeline_watcher.core import ReportBase, _now
from .helpers import review_flag_unset


# ReportBase is abstract and requires implementation of ok method.
class ReportBaseWithOK(ReportBase):
    def ok(self):
        return True


def test_report_base_clear_review():
    rb = ReportBaseWithOK().start()
    rb.request_review("Human review required")
    rb.clear_review()
    assert review_flag_unset(rb)
    assert rb.percent == 0
    assert rb.running
    assert rb.duration_ms < 1000.
    assert rb.finished_at is None
    assert len(rb.notes) == 0
    assert len(rb.errors) == 0
    assert len(rb.warnings) == 0
    assert len(rb.metadata) == 0


def test_report_base_end():
    rb = ReportBaseWithOK()
    assert not rb.terminal
    assert rb.running
    assert rb.started_at is not None
    assert rb.finished_at is None
    assert not rb.succeeded
    rb.end()
    assert rb.terminal
    assert not rb.running
    assert rb.started_at is not None
    assert rb.finished_at is not None
    assert rb.succeeded


def test_report_base_error():
    rb = ReportBaseWithOK().start()
    rb.error("Error test in progress")
    assert rb.percent == 0
    assert rb.running
    assert rb.duration_ms < 1000.
    assert rb.finished_at is None
    assert review_flag_unset(rb)
    assert len(rb.notes) == 0
    assert len(rb.errors) == 1
    assert len(rb.warnings) == 0
    assert len(rb.metadata) == 0
    assert rb.errors[0] == "Error test in progress"


def test_report_base_start():
    # ReportBase is abstract and requires implementation of ok method.

    rb = ReportBaseWithOK().start()
    assert rb.percent == 0
    assert rb.running
    assert (_now() - rb.started_at).seconds < 1
    assert rb.finished_at is None
    assert review_flag_unset(rb)
    assert len(rb.notes) == 0
    assert len(rb.errors) == 0
    assert len(rb.warnings) == 0
    assert len(rb.metadata) == 0


def test_report_base_succeed():
    rb = ReportBaseWithOK().start()
    rb.succeed()
    assert rb.percent == 100
    assert rb.succeeded
    assert (_now() - rb.started_at).seconds < 1
    assert (_now() - rb.finished_at).seconds < 1
    assert review_flag_unset(rb)
    assert len(rb.notes) == 0
    assert len(rb.errors) == 0
    assert len(rb.warnings) == 0
    assert len(rb.metadata) == 0


def test_report_base_fail():
    rb = ReportBaseWithOK().start()
    rb.fail("The report failed")
    assert rb.percent == 0
    assert rb.failed
    assert (_now() - rb.started_at).seconds < 1
    assert (_now() - rb.finished_at).seconds < 1
    assert review_flag_unset(rb)
    assert len(rb.notes) == 0
    assert len(rb.errors) == 1
    assert len(rb.warnings) == 0
    assert len(rb.metadata) == 0


def test_report_base_skip():
    rb = ReportBaseWithOK().start()
    rb.skip("The report skipped")
    assert rb.percent == 0
    assert rb.skipped
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
    rb = ReportBaseWithOK().start()
    rb.request_review("Human review required")
    assert rb.requires_human_review
    assert rb.review.flagged
    assert rb.review.reason == "Human review required"
    assert rb.percent == 0
    assert rb.running
    assert (_now() - rb.started_at).seconds < 1
    assert rb.finished_at is None
    assert len(rb.notes) == 0
    assert len(rb.errors) == 0
    assert len(rb.warnings) == 0
    assert len(rb.metadata) == 0


def test_report_base_note():
    rb = ReportBaseWithOK().start()
    rb.note("Note test in progress")
    assert rb.percent == 0
    assert rb.running
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
    rb = ReportBaseWithOK().start()
    rb.warn("Warn test in progress")
    assert rb.percent == 0
    assert rb.running
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
