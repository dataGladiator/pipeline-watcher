from pathlib import Path
from pipeline_watcher.core import ReportBase, _now
from .helpers import *


# ReportBase is abstract and requires implementation of ok method.
class ReportBaseWithOK(ReportBase):
    def ok(self):
        return True

def test_report_base_clear_review():
    rb = ReportBaseWithOK().start()
    assert report_base_clear_review(rb)

def test_report_base_end():
    rb = ReportBaseWithOK().start()
    assert report_base_end(rb)

def test_report_base_error():
    rb = ReportBaseWithOK().start()
    assert report_base_error(rb)

def test_report_base_fail():
    rb = ReportBaseWithOK().start()
    assert report_base_fail(rb)

def test_report_base_model_post_init():
    rb = ReportBaseWithOK()
    assert report_base_model_post_init(rb)

def test_report_base_model_post_init_with_keyword_args():
    percent = 10
    metadata = {'datum': 'value'}
    rb = ReportBaseWithOK(percent=percent, metadata=metadata)
    assert report_base_model_post_init_with_keyword_args(rb, percent, metadata)

def test_report_base_note():
    rb = ReportBaseWithOK()
    assert report_base_note(rb)

def test_report_base_request_review():
    rb = ReportBaseWithOK()
    assert report_base_request_review(rb)


def test_report_base_skip():
    rb = ReportBaseWithOK()
    assert report_base_skip(rb)

def test_report_base_start():
    rb = ReportBaseWithOK().start()
    assert report_base_start(rb)

def test_report_base_succeed():
    rb = ReportBaseWithOK().start()
    assert report_base_succeed(rb)

def test_report_base_warn():
    rb = ReportBaseWithOK()
    assert report_base_warn(rb)