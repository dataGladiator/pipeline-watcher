from pipeline_watcher import StepReport
from pathlib import Path
from .helpers import *

############################
#  TEST INHERITED METHODS  #
############################

def test_file_report_report_base_clear_review():
    fr = FileReport(path=Path("")).start()
    assert report_base_clear_review(fr)

def test_file_report_report_base_end():
    fr = FileReport(path=Path("")).start()
    assert report_base_end(fr)

def test_file_report_report_base_error():
    fr = FileReport(path=Path("")).start()
    assert report_base_error(fr)

def test_file_report_report_base_fail():
    fr = FileReport(path=Path("")).start()
    assert report_base_fail(fr)

def test_file_report_report_base_model_post_init():
    fr = FileReport(path=Path(""))
    assert report_base_model_post_init(fr)

def test_file_report_report_base_model_post_init_with_keyword_args():
    percent = 10
    metadata = {'datum': 'value'}
    fr = FileReport(path=Path(""),
                    percent=percent,
                    metadata=metadata)
    assert report_base_model_post_init_with_keyword_args(fr, percent, metadata)

def test_file_report_report_base_note():
    fr = FileReport(path=Path(""))
    assert report_base_note(fr)

def test_file_report_report_base_request_review():
    fr = FileReport(path=Path(""))
    assert report_base_request_review(fr)

def test_file_report_report_base_skip():
    fr = FileReport(path=Path(""))
    assert report_base_skip(fr)

def test_file_report_report_base_start():
    fr = FileReport(path=Path("")).start()
    assert report_base_start(fr)

def test_file_report_report_base_succeed():
    fr = FileReport(path=Path(""))
    assert report_base_succeed(fr)

def test_file_report_report_base_warn():
    fr = FileReport(path=Path(""))
    assert report_base_warn(fr)

############################
#  TEST INHERITED METHODS  #
############################

def test_file_report_add_completed_step():
    fr = FileReport(path=Path(""))
    fr.add_completed_step(label="label", id="id")
    assert len(fr.steps) == 1
    assert fr.steps[0].label == "label"
    assert fr.steps[0].id == "id"
    assert fr.steps[0].succeeded
    assert fr.steps[0].terminal


def test_file_report_add_failed_step():
    fr = FileReport(path=Path(""))
    fr.add_failed_step(label="label", id="id")
    assert len(fr.steps) == 1
    assert fr.steps[0].label == "label"
    assert fr.steps[0].id == "id"
    assert fr.steps[0].failed
    assert fr.steps[0].terminal
    assert fr.running # failure has not yet bubbled.
    assert not fr.ok #
    assert fr.running  # ok does not alter status
    fr.end()
    assert fr.failed # end bubbles failure

def test_file_report_add_review_step():
    fr = FileReport(path=Path(""))
    fr.add_review_step(label="label", id="id")
    assert fr.requires_human_review
    # the step is completed and successful
    assert fr.steps[0].terminal
    assert fr.steps[0].succeeded
    # but the file process is still running
    assert fr.running

def test_file_report_add_skipped_step():
    fr = FileReport(path=Path(""))
    fr.add_skipped_step(label="label", id="id")
    assert not fr.requires_human_review
    assert fr.steps[0].terminal
    assert fr.steps[0].skipped
    assert fr.running

def test_file_report_append_step_basic():
    fr = FileReport(path=Path(""))
    step = StepReport(label="label", id="id")
    fr.append_step(step)
    assert not fr.requires_human_review
    assert fr.steps[0].terminal
    assert fr.steps[0].succeeded
    assert fr.steps[0] == step
    assert fr.running

def test_file_report_append_step_construct_id():
    fr = FileReport(path=Path(""))
    step = StepReport(label="label")
    assert step.id is None
    fr.append_step(step)
    assert not fr.requires_human_review
    assert fr.steps[0].terminal
    assert fr.steps[0].succeeded
    assert fr.steps[0] == step # note it is the same step
    assert fr.steps[0].id == "label" # but its id has been set.
    assert fr.running

def test_file_report_begin():
    file_id = "id"
    metadata = {'data': 'value'}
    fr = FileReport.begin(path=Path(""),
                          file_id=file_id,
                          metadata=metadata)
    assert fr.running
    assert fr.metadata == metadata
    assert fr.file_id == file_id
    assert fr.started_at is not None

def test_file_report_last_step():
    fr = FileReport(path=Path(""))
    assert fr.last_step() is None
    fr.add_completed_step(label="label", id="id")
    assert fr.last_step() is not None
    step = StepReport(label="label", id="id")
    fr.append_step(step)
    assert fr.last_step() == step
