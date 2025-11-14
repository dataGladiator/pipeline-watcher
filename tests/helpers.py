from pipeline_watcher.core import ReportBase, FileReport


def review_flag_is_unset(report: FileReport | ReportBase) -> bool:
    assert not report.requires_human_review
    assert report.review.flagged is False
    assert report.review.reason is None
    return True


def are_unset(report: FileReport | ReportBase, attribute_list: list) -> bool:
    for attribute in attribute_list:
        if hasattr(report, attribute):
            assert len(getattr(report, attribute)) == 0
        else:
            raise ValueError(f"Attribute '{attribute}' not found on object '{str(report)}'")
    return True


def report_base_clear_review(rb: ReportBase):
    rb.request_review("Human review required")
    rb.clear_review()
    assert review_flag_is_unset(rb)
    assert rb.percent == 0
    assert rb.running
    assert rb.duration_ms < 1000.
    assert rb.finished_at is None
    assert are_unset(rb, ['notes', 'errors', 'warnings', 'metadata'])
    return True


def report_base_end(rb: ReportBase):
    assert not rb.terminal
    assert rb.running
    assert rb.started_at is not None
    assert rb.finished_at is None
    assert not rb.succeeded
    assert are_unset(rb, ['notes', 'errors', 'warnings', 'metadata'])
    rb.end()
    assert rb.terminal
    assert not rb.running
    assert rb.started_at is not None
    assert rb.finished_at is not None
    assert rb.succeeded
    assert are_unset(rb, ['notes', 'errors', 'warnings', 'metadata'])
    return True


def report_base_error(rb: ReportBase) -> bool:
    rb.error("Error test in progress")
    assert rb.percent == 0
    assert rb.running
    assert rb.duration_ms < 1000.
    assert rb.finished_at is None
    assert review_flag_is_unset(rb)
    assert are_unset(rb, ['notes', 'warnings', 'metadata'])
    assert len(rb.errors) == 1
    assert rb.errors[0] == "Error test in progress"
    return True


def report_base_fail(rb: ReportBase) -> bool:
    rb.fail("The report failed")
    assert rb.percent == 0
    assert rb.failed
    assert rb.duration_ms < 1000.
    assert rb.started_at is not None
    assert rb.finished_at is not None
    assert review_flag_is_unset(rb)
    assert are_unset(rb, ['notes', 'warnings', 'metadata'])
    assert len(rb.errors) == 1
    return True


def report_base_model_post_init(rb: ReportBase) -> bool:
    assert rb.percent == 0
    assert rb.running
    assert rb.started_at is not None
    assert rb.finished_at is None
    assert review_flag_is_unset(rb)
    assert are_unset(rb, ['notes', 'errors', 'warnings', 'metadata'])
    return True

def report_base_model_post_init_with_keyword_args(
        rb: ReportBase,
        percent: int,
        metadata: dict
) -> bool:
    assert rb.percent == percent
    assert rb.metadata == metadata
    assert rb.running
    assert rb.started_at is not None
    assert rb.finished_at is None
    assert review_flag_is_unset(rb)
    assert are_unset(rb, ['notes', 'errors', 'warnings'])
    return True

def report_base_note(rb: ReportBase, note: str="Note test in progress") -> bool:
    rb.note(note)
    assert rb.percent == 0
    assert rb.running
    assert rb.started_at is not None
    assert rb.finished_at is None
    assert review_flag_is_unset(rb)
    assert are_unset(rb, ['errors', 'warnings', 'metadata'])
    assert len(rb.notes) == 1
    assert rb.notes[0] == note
    return True

def report_base_request_review(
        rb: ReportBase,
        reason: str="Human review required"
) -> bool:
    rb.request_review(reason)
    assert rb.requires_human_review
    assert rb.review.flagged
    assert rb.review.reason == reason
    assert rb.percent == 0
    assert rb.running
    assert rb.finished_at is None
    assert rb.started_at is not None
    assert are_unset(rb, ['notes', 'errors', 'warnings', 'metadata'])
    return True

def report_base_skip(
        rb: ReportBase,
        note: str = "The report skipped"
) -> bool:
    rb.skip(note)
    assert rb.percent == 0
    assert rb.skipped
    assert rb.started_at is not None
    assert rb.finished_at is not None
    assert rb.duration_ms < 1000.
    assert review_flag_is_unset(rb)
    assert are_unset(rb, ['errors', 'metadata', 'warnings'])
    assert len(rb.notes) == 1
    assert note in rb.notes[0]
    return True

def report_base_start(rb: ReportBase) -> bool:
    assert rb.percent == 0
    assert rb.running
    assert rb.started_at is not None
    assert rb.finished_at is None
    assert review_flag_is_unset(rb)
    assert are_unset(rb, ['notes', 'errors', 'warnings', 'metadata'])
    return True

def report_base_succeed(rb: ReportBase) -> bool:
    rb.succeed()
    assert rb.percent == 100
    assert rb.succeeded
    assert rb.started_at is not None
    assert rb.finished_at is not None
    assert rb.duration_ms < 1000.
    assert review_flag_is_unset(rb)
    assert are_unset(rb, ['notes', 'errors', 'metadata', 'warnings'])
    return True

def report_base_warn(
        rb: ReportBase,
        warning: str = "Warn test in progress"
):
    rb.warn(warning)
    assert rb.percent == 0
    assert rb.running
    assert rb.started_at is not None
    assert rb.finished_at is None
    assert review_flag_is_unset(rb)
    assert are_unset(rb, ['notes', 'errors', 'metadata'])
    assert len(rb.warnings) == 1
    assert rb.warnings[0] == warning
    return True