from __future__ import annotations
import re
from pathlib import Path


def _norm_text(s: str | None) -> str:
    """Normalize a text fragment for comparisons.

    Collapses internal whitespace, strips leading/trailing space,
    and lowercases the result. ``None`` becomes an empty string.

    Parameters
    ----------
    s
        Input string (or ``None``).

    Returns
    -------
    str
        Normalized, lowercased text.

    Examples
    --------
    >>> _norm_text("  Hello   World  ")
    'hello world'
    >>> _norm_text(None)
    ''
    """
    return "" if s is None else " ".join(str(s).strip().split()).lower()


def _norm_key(x) -> str:
    """Produce a normalized string key from arbitrary input.

    Accepts strings, :class:`pathlib.Path`, or any object that implements
    ``__str__``. Falls back to ``repr(x)`` if ``str(x)`` fails.

    Parameters
    ----------
    x
        Input to convert to a comparison key.

    Returns
    -------
    str
        Normalized key suitable for case/space-insensitive matching.

    Notes
    -----
    Internally uses :func:`_norm_text` to standardize output.
    """
    if isinstance(x, Path):
        return _norm_text(str(x))
    try:
        return _norm_text(str(x))
    except Exception:
        return _norm_text(repr(x))


def _file_keys(fr: "FileReport") -> set[str]:
    """Collect comparable keys for a file record.

    Returns normalized identifiers that can be used to match a file by
    multiple handles (``file_id``, ``name``, full ``path``, and basename).

    Parameters
    ----------
    fr
        A :class:`FileReport` instance.

    Returns
    -------
    set[str]
        Set of normalized keys for matching.

    Examples
    --------
    >>> class Dummy:  # minimal stand-in
    ...     file_id, name, path = "F1", "doc.pdf", "inputs/doc.pdf"
    >>> sorted(_file_keys(Dummy()))
    ['doc.pdf', 'f1', 'inputs/doc.pdf', 'doc.pdf']
    """
    keys = set()
    if fr.file_id:
        keys.add(_norm_key(fr.file_id))
    if fr.name:
        keys.add(_norm_key(fr.name))
    if fr.path:
        p = Path(fr.path)
        keys.add(_norm_key(str(p)))
        keys.add(_norm_key(p.name))
    return keys


def _slugify(s: str) -> str:
    """Create a URL/filename-friendly slug.

    Converts to lowercase, collapses whitespace to ``-``,
    and removes characters outside ``[a-z0-9-]``.

    Parameters
    ----------
    s
        Input string.

    Returns
    -------
    str
        Slugified string.

    Examples
    --------
    >>> _slugify("  File Name (v2)! ")
    'file-name-v2'
    """
    s = re.sub(r"\s+", "-", s.strip().lower())
    s = re.sub(r"[^a-z0-9\-]+", "", s)
    return s
