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
from dataclasses import dataclass, field, fields, replace
from typing import Iterable, Optional, Tuple, Type


_SYSTEM_FATAL_EXCEPTIONS: Tuple[Type[BaseException], ...] = (
    KeyboardInterrupt,
    SystemExit,
)


def _dedupe_exception_types(
    *groups: Optional[Tuple[Type[BaseException], ...]],
) -> Tuple[Type[BaseException], ...]:
    out: list[Type[BaseException]] = []

    for group in groups:
        if not group:
            continue

        for exc in group:
            if not isinstance(exc, type) or not issubclass(exc, BaseException):
                raise TypeError(
                    "Expected exception classes subclassing BaseException, "
                    f"got {exc!r}"
                )

            if exc not in out:
                out.append(exc)

    return tuple(out)


@dataclass(frozen=True)
class WatcherSettings:
    # Exception behavior
    raise_on_exception: bool = False
    store_traceback: bool = True
    traceback_limit: Optional[int] = None
    capture_streams: bool = False
    capture_warnings: bool = True

    # Routing policy
    suppressed_exceptions: Optional[Tuple[Type[BaseException], ...]] = None

    # User/project fatal exceptions.
    # These are added to _system_fatal_exceptions.
    pipeline_fatal_exceptions: Tuple[Type[BaseException], ...] = ()

    # System fatal exceptions.
    # Override only if you intentionally want to change interrupt/exit behavior.
    _system_fatal_exceptions: Tuple[Type[BaseException], ...] = field(
        default=_SYSTEM_FATAL_EXCEPTIONS,
        repr=False,
    )

    # Persistence policy
    save_on_exception: bool = True
    exception_save_path_override: Optional[str] = None
    min_seconds_between_exception_saves: float = 0.0

    @property
    def fatal_exceptions(self) -> Tuple[Type[BaseException], ...]:
        """
        Effective fatal exceptions.

        Includes system fatal exceptions plus project/pipeline fatal exceptions.
        """
        return _dedupe_exception_types(
            self._system_fatal_exceptions,
            self.pipeline_fatal_exceptions,
        )

    def is_fatal(self, e: BaseException) -> bool:
        return isinstance(e, self.fatal_exceptions)

    def is_suppressed(self, e: BaseException) -> bool:
        sx = self.suppressed_exceptions
        return bool(sx) and isinstance(e, sx)

    def should_raise(self, e: BaseException) -> bool:
        if self.is_fatal(e):
            return True
        if self.raise_on_exception and not self.is_suppressed(e):
            return True
        return False

__all__ = [
    "WatcherSettings",
    "current_settings",
    "use_settings",
    "with_overrides",
    "set_global_settings",
]


def _normalize_exception_types(
    value: (
        None
        | Type[BaseException]
        | Iterable[Type[BaseException]]
    ),
) -> tuple[Type[BaseException], ...]:
    """
    Normalize exception type settings into a validated tuple.

    Accepts:
    - None
    - a single exception class
    - an iterable of exception classes
    """
    if value is None:
        return ()

    if isinstance(value, type) and issubclass(value, BaseException):
        candidates = (value,)
    else:
        candidates = tuple(value)

    out: list[Type[BaseException]] = []

    for exc in candidates:
        if not isinstance(exc, type) or not issubclass(exc, BaseException):
            raise TypeError(
                "Expected exception classes subclassing BaseException, "
                f"got {exc!r}"
            )

        if exc not in out:
            out.append(exc)

    return tuple(out)


def _normalize_settings_overrides(overrides: dict) -> dict:
    overrides = dict(overrides)

    if "fatal_exceptions" in overrides:
        overrides["_pipeline_fatal_exceptions"] = _normalize_exception_types(
            overrides.pop("fatal_exceptions") or ()
        )

    if "_system_fatal_exceptions" in overrides:
        overrides["_system_fatal_exceptions"] = _normalize_exception_types(
            overrides["_system_fatal_exceptions"] or ()
        )

    unknown = set(overrides) - _SETTINGS_FIELD_NAMES
    if unknown:
        raise TypeError(
            f"Unknown WatcherSettings override(s): {sorted(unknown)}"
        )

    return overrides


@dataclass(frozen=True)
class WatcherSettings:
    # Exception behavior
    raise_on_exception: bool = False
    store_traceback: bool = True
    traceback_limit: Optional[int] = None
    capture_streams: bool = False
    capture_warnings: bool = True

    # Routing policy
    suppressed_exceptions: Optional[Tuple[Type[BaseException], ...]] = None

    # Internal storage for user/project fatal exceptions.
    _pipeline_fatal_exceptions: Tuple[Type[BaseException], ...] = field(
        default=(),
        repr=False,
    )

    # Internal system-level fatal exceptions.
    # Override only in unusual cases.
    _system_fatal_exceptions: Tuple[Type[BaseException], ...] = field(
        default=_SYSTEM_FATAL_EXCEPTIONS,
        repr=False,
    )

    # Persistence policy
    save_on_exception: bool = True
    exception_save_path_override: Optional[str] = None
    min_seconds_between_exception_saves: float = 0.0

    @property
    def fatal_exceptions(self) -> Tuple[Type[BaseException], ...]:
        """
        Effective fatal exceptions.

        Includes system fatal exceptions plus user/project fatal exceptions.
        """
        return _dedupe_exception_types(
            self._system_fatal_exceptions,
            self._pipeline_fatal_exceptions,
        )

    def is_fatal(self, e: BaseException) -> bool:
        """True if e must always be raised."""
        return isinstance(e, self.fatal_exceptions)


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


_SETTINGS_FIELD_NAMES = {f.name for f in fields(WatcherSettings)}
_VIRTUAL_SETTINGS_KEYS = {"fatal_exceptions"}
_SETTINGS_KEYS = _SETTINGS_FIELD_NAMES | _VIRTUAL_SETTINGS_KEYS
