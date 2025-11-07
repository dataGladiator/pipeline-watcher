from pathlib import Path
from pipeline_watcher.utilities import (_norm_text,
                                        _norm_key,
                                        _file_keys,
                                        _slugify)


def test_norm_text():
    assert _norm_text(None) == ""
    assert _norm_text(" the wild   dog  ") == "the wild dog"
    assert _norm_text("another TEST") == "another test"


def test_norm_key():
    assert _norm_key(None) == "none"
    assert _norm_key(None) == "none"
    class X:
        def __repr__(self):
            return "<X>"
    assert _norm_key(X()) == "<x>"
    assert _norm_key(Path()) == "."


def test_file_keys():
    class Dummy:
        file_id, name, path = "F1", "doc.pdf", "inputs/doc.pdf"
    file_keys = sorted(_file_keys(Dummy()))
    assert file_keys[0] == 'doc.pdf'
    assert file_keys[1] == 'f1'
    assert file_keys[2] == 'inputs/doc.pdf'

def test_slugify():
    assert _slugify("  File Name (v2)! ") == 'file-name-v2'