# settings.py
"""
Watcher settings and context management for pipeline-watcher.

This module defines an immutable :class:`WatcherSettings` dataclass and a small
context-management layer that lets you configure how the watcher behaves when
recording pipeline activity (exception routing, traceback capture, persistence,
etc.). Settings are stored in a :class:`contextvars.ContextVar`, so overrides
are **per logical context** (safe for async, threads, nested calls).

The precedence model (highest → lowest) is:

1. Explicit overrides passed to a local context (e.g., :class:`use_settings`,
   or overrides applied inside a `pipeline_file` / `file_step` block)
2. File-local overrides (e.g., attached to a specific `FileReport`)
3. Run / pipeline-wide overrides (outer :class:`use_settings`)
4. Global defaults (module default)

Examples
--------
Basic usage with run-wide overrides::

    from pipeline_watcher.settings import use_settings, current_settings

    with use_settings(save_on_exception=True, capture_warnings=True):
        eff = current_settings()
        # ... run pipeline with these effective settings ...

Nesting and shadowing overrides::

    with use_settings(store_traceback=False):
        # inner block temporarily re-enables tracebacks and raises on exception
        with use_settings(store_traceback=True, raise_on_exception=True):
            ...
        # after inner block, store_traceback returns to False

Creating a derived settings object (without changing context)::

    from pipeline_watcher.settings import current_settings, with_overrides
    eff_for_step = with_overrides(current_settings(), traceback_limit=50)
"""
from __future__ import annotations
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Optional, Tuple, Type


__all__ = [
    "WatcherSettings",
    "current_settings",
    "use_settings",
    "with_overrides",
    "set_global_settings",
]


@dataclass(frozen=True)
class WatcherSettings:
    """
    Immutable settings controlling watcher behavior.

    These flags determine how exceptions are routed, what incident data is
    captured (tracebacks, I/O streams, warnings), and whether the pipeline
    should be persisted immediately after an exception. Instances are intended
    to be **read-only** and layered via context overrides.

    Parameters
    ----------
    raise_on_exception : bool, default False
        If ``True``, re-raise exceptions not listed in suppressed_exceptions.
        If ``False``, catch exceptions not listed in fatal_exceptions.
    store_traceback : bool, default True
        If ``True``, attach a formatted traceback string to metadata when
        exceptions occur.
    traceback_limit : int or None, default None
        If set, limit the traceback to the last ``N`` frames. ``None`` keeps
        the full traceback.
    capture_streams : bool, default False
        If ``True``, capture ``stdout`` and ``stderr`` during watched blocks,
        storing them in metadata for UI inspection.
    capture_warnings : bool, default True
        If ``True``, capture Python warnings emitted during watched blocks.
    suppressed_exceptions : tuple of Exception types or None, default None
        Exceptions that do not raise when `raise_on_exception=True` (always recorded).
    fatal_exceptions : tuple of Exception types, default (KeyboardInterrupt, SystemExit)
        Exceptions that are always raised (never suppressed).

    save_on_exception : bool, default True
        If ``True``, attempt to persist the pipeline report immediately on
        exception (best-effort).
    exception_save_path_override : str or None, default None
        If provided, use this path instead of a pipeline's default output path
        when saving on exception.
    min_seconds_between_exception_saves : float, default 0.0
        Minimum time (seconds) between successive auto-saves triggered by
        exceptions. ``0`` disables throttling.

    Notes
    -----
    - The watcher treats :class:`KeyboardInterrupt` and :class:`SystemExit`
      as non-swallowable by default via :attr:`reraise`.
    - Prefer layering settings with :class:`use_settings` or
      :func:`with_overrides` rather than mutating state.
    """

    # Exception behavior
    raise_on_exception: bool = False
    store_traceback: bool = True
    traceback_limit: Optional[int] = None
    capture_streams: bool = False
    capture_warnings: bool = True

    # Routing policy
    suppressed_exceptions: Optional[Tuple[Type[BaseException], ...]] = None
    fatal_exceptions: Tuple[Type[BaseException], ...] = (KeyboardInterrupt, SystemExit)

    # Persistence policy
    save_on_exception: bool = True
    exception_save_path_override: Optional[str] = None
    min_seconds_between_exception_saves: float = 0.0

    def is_fatal(self, e: BaseException) -> bool:
        """True if e must always be raised (ignores raise_on_exception)."""
        fx = self.fatal_exceptions
        return bool(fx) and isinstance(e, fx)

    def is_suppressed(self, e: BaseException) -> bool:
        """True if e is allowed to NOT raise when raise_on_exception=True."""
        sx = self.suppressed_exceptions
        return bool(sx) and isinstance(e, sx)

    def should_raise(self, e: BaseException) -> bool:
        """
        Decide whether to raise `e` according to the simplified policy:

        1) Fatal exceptions always raise.
        2) If raise_on_exception is True, raise unless e is suppressed.
        3) Otherwise, do not raise (but always record).
        """
        if self.is_fatal(e):
            return True
        if self.raise_on_exception and not self.is_suppressed(e):
            return True
        return False

    def suppression_breadcrumb(self, e: BaseException) -> Optional[str]:
        """
        Optional message explaining why an exception wasn't raised.
        Returned only when raise_on_exception=True and e was suppressed.
        """
        if self.raise_on_exception and self.is_suppressed(e):
            return f"suppressed raise_on_exception for {type(e).__name__} via suppressed_exceptions"
        return None

#: Module-level default settings used when no overrides are active.
_default_settings = WatcherSettings()


#: Context-local settings for the current logical flow (async/thread safe).
_settings_var: ContextVar[WatcherSettings] = ContextVar("watcher_settings")


def current_settings() -> WatcherSettings:
    """
    Return the effective :class:`WatcherSettings` for the current context.

    Returns
    -------
    WatcherSettings
        The settings object currently active for this context.

    Notes
    -----
    The value is resolved from the :class:`~contextvars.ContextVar` stack.
    If no overrides were applied, the module default is returned.

    Examples
    --------
    >>> from pipeline_watcher.settings import current_settings
    >>> s = current_settings()
    >>> isinstance(s, WatcherSettings)
    True
    """
    return _settings_var.get(_default_settings)

class use_settings:
    """
    Context manager to apply temporary settings overrides.

    Keyword arguments correspond to fields on :class:`WatcherSettings` and
    replace the current context's settings immutably for the duration of
    the ``with`` block.

    Parameters
    ----------
    **overrides
        Field-value pairs to override in the current context.

    Returns
    -------
    WatcherSettings
        The effective settings object installed for the context.

    Notes
    -----
    - Overrides are **stackable**; inner contexts take precedence.
    - On exit, the previous settings are restored.
    - This context manager never suppresses exceptions raised inside it.

    Examples
    --------
    Temporarily enable persistence and traceback capture::

        with use_settings(save_on_exception=True, store_traceback=True) as eff:
            # eff reflects the merged settings within this block
            ...

    Nested overrides that re-enable interrupts swallowing (not recommended)::

        with use_settings(reraise=()):
            ...
    """

    def __init__(self, **overrides):
        self._overrides = overrides
        self._token: Optional[Token] = None
        self._effective: Optional[WatcherSettings] = None

    def __enter__(self) -> WatcherSettings:
        base = current_settings()

        # No-op reader path: return current settings without pushing a value
        if not self._overrides:
            self._effective = base
            self._token = None
            return base

        # Override path: create merged settings and set them for this context
        eff = replace(base, **self._overrides)
        self._effective = eff
        self._token = _settings_var.set(eff)
        return eff

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._token is not None:
            _settings_var.reset(self._token)
        # Never suppress exceptions
        return False


def with_overrides(base: WatcherSettings, **overrides) -> WatcherSettings:
    """
    Return a new :class:`WatcherSettings` with selected fields replaced.

    This is useful when you need a derived settings object (e.g., for a specific
    file or step) without mutating context-global state.

    Parameters
    ----------
    base : WatcherSettings
        The base settings object to copy.
    **overrides
        Field-value pairs to override on the returned object.

    Returns
    -------
    WatcherSettings
        A new immutable settings instance with the requested overrides applied.

    Examples
    --------
    Create a per-step effective settings object::

        eff_for_step = with_overrides(current_settings(), traceback_limit=25)
    """
    return replace(base, **overrides)


def set_global_settings(**overrides) -> WatcherSettings:
    """
    Permanently replace the process-wide default settings.

    Notes
    -----
    - Intended for top-level scripts and one-off runs.
    - Subsequent calls to :func:`current_settings` or :class:`use_settings`
      will inherit from this new base.
    - This affects the entire interpreter process.
    - Not suitable for libraries and concurrent pipelines.
    """
    global _default_settings
    new = replace(_default_settings, **overrides)
    _default_settings = new
    _settings_var.set(new)
    return new
