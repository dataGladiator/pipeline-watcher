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
        If ``True``, re-raise exceptions after they are recorded. If ``False``,
        exceptions are recorded and suppressed by default (subject to
        :attr:`reraise` and :attr:`catch` policies).
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

    catch : tuple of Exception types or None, default None
        If provided, only exceptions that are instances of these types will be
        **caught** (others will be re-raised). If ``None``, the watcher will
        catch all exceptions **except** those in :attr:`reraise`.
    reraise : tuple of Exception types, default (KeyboardInterrupt, SystemExit)
        Exceptions that must always be re-raised (i.e., never swallowed).

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
    catch: Optional[Tuple[Type[BaseException], ...]] = None
    reraise: Tuple[Type[BaseException], ...] = (KeyboardInterrupt, SystemExit)

    # Persistence policy
    save_on_exception: bool = True
    exception_save_path_override: Optional[str] = None
    min_seconds_between_exception_saves: float = 0.0

#: Module-level default settings used when no overrides are active.
_default_settings = WatcherSettings()

#: Context-local settings for the current logical flow (async/thread safe).
_settings_var: ContextVar[WatcherSettings] = ContextVar(
    "watcher_settings", default=_default_settings
)

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
    return _settings_var.get()

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

