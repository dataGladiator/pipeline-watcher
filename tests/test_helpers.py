from pipeline_watcher import Status, Check
from pipeline_watcher.core import ReviewFlag, ReviewHelpers


def test_status_members_and_values():
    # enforce API names
    assert {m.name for m in Status} == {
        "PENDING", "RUNNING", "SUCCEEDED", "FAILED", "SKIPPED"
    }
    # enforce serialization values
    assert {m.value for m in Status} == {
        "pending", "running", "succeeded", "failed", "skipped"
    }


def test_status_properties():
    assert Status.RUNNING.running
    assert Status.FAILED.failed
    assert Status.PENDING.pending
    assert Status.SKIPPED.skipped
    assert Status.SUCCEEDED.succeeded
    assert Status.SUCCEEDED.terminal
    assert Status.FAILED.terminal
    assert Status.SKIPPED.terminal
    assert not Status.PENDING.terminal
    assert not Status.RUNNING.terminal


def test_check():
    check = Check(name="ids_unique", ok=False, detail="3 duplicates")
    assert check.name == "ids_unique"
    assert not check.ok
    assert check.detail == "3 duplicates"


def test_review_flag():
    review_flag = ReviewFlag()
    assert not review_flag.flagged
    assert review_flag.reason is None

    review_flag = ReviewFlag(flagged=True, reason='check failed.')
    assert review_flag.flagged
    assert review_flag.reason == 'check failed.'


def test_review_helpers():
    class A(ReviewHelpers):
        review = ReviewFlag()
    a = A()
    assert not a.requires_human_review
    assert not a.review.flagged
    assert a.review.reason is None

    a.request_review("check failed.")
    assert a.requires_human_review
    assert a.review.flagged
    assert a.review.reason == 'check failed.'

    a.clear_review()
    assert not a.requires_human_review
    assert not a.review.flagged
    assert a.review.reason is None
