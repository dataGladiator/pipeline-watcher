from pipeline_watcher import StepReport, FileReport, Status
from .helpers import *

############################
#  TEST INHERITED METHODS  #
############################

def test_step_report_report_base_clear_review():
    step = StepReport(label="test-label").start()
    assert report_base_clear_review(step)

def test_step_report_report_base_end():
    step = StepReport(label="test-label").start()
    assert report_base_end(step)

def test_step_report_report_base_error():
    step = StepReport(label="test-label").start()
    assert report_base_error(step)

def test_step_report_report_base_fail():
    step = StepReport(label="test-label").start()
    assert report_base_fail(step)

def test_step_report_report_base_model_post_init():
    step = StepReport(label="test-label")
    assert report_base_model_post_init(step)

def test_step_report_report_base_model_post_init_with_keyword_args():
    percent = 10
    metadata = {'datum': 'value'}
    step = StepReport(label="test-label",
                      percent=percent,
                      metadata=metadata)
    assert report_base_model_post_init_with_keyword_args(step, percent, metadata)

def test_step_report_report_base_note():
    step = StepReport(label="test-label")
    assert report_base_note(step)

def test_step_report_report_base_request_review():
    step = StepReport(label="test-label")
    assert report_base_request_review(step)

def test_step_report_report_base_skip():
    step = StepReport(label="test-label")
    assert report_base_skip(step)

def test_step_report_report_base_start():
    step = StepReport(label="test-label").start()
    assert report_base_start(step)

def test_step_report_report_base_succeed():
    step = StepReport(label="test-label")
    assert report_base_succeed(step)

def test_step_report_report_base_warn():
    step = StepReport(label="test-label")
    assert report_base_warn(step)

##################################
#  TEST StepReport ONLY METHODS  #
##################################

def test_step_report_end_idempotent_and_success():
    s = StepReport.begin("Parse")
    # no checks, no errors -> ok True -> succeed on end
    s1 = s.end()
    assert s1.succeeded
    assert s1.percent == 100
    assert s1.finished_at is not None
    assert s1.status.succeeded
    finished = s1.finished_at
    # calling end again should not change terminal status, only ensure finished_at
    s2 = s1.end()
    assert s2.status.succeeded
    assert s2.finished_at == finished


def test_step_report_end_success_with_ok_check():
    s = StepReport.begin(label="Analyze")
    s.add_check("unique_ids", ok=True, detail="passed check")
    s.end()
    assert s.succeeded
    assert s.terminal

def test_step_report_end_failure_with_failed_check():
    s = StepReport.begin(label="Analyze")
    s.add_check("unique_ids", ok=False, detail="duplicates found")
    s.end()
    assert s.status.failed
    assert "duplicates" in s.errors[0] # surfaces errors

def test_step_report_begin():
    step = StepReport.begin(label="Analyze")
    assert step.id != ""
    assert step.id is not None
    assert step.id == 'analyze'
    assert step.running
    assert not step.terminal
    assert step.started_at is not None
    assert step.finished_at is None


