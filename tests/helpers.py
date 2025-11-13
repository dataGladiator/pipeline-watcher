from pipeline_watcher.core import ReportBase, FileReport


def review_flag_unset(report: FileReport | ReportBase) -> bool:
    assert not report.requires_human_review
    assert report.review.flagged is False
    assert report.review.reason is None
    return True
