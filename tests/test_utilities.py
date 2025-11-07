from pipeline_watcher.utilities import _norm_text

def test_norm_text():
    assert _norm_text(None) == ""
    assert _norm_text(" the wild   dog  ") == "the wild dog"
    assert _norm_text("another TEST") == "another test"