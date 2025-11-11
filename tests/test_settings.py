import builtins
import types
import pytest
import threading
from pipeline_watcher.settings import (
    WatcherSettings,
    current_settings,
    set_global_settings,
    use_settings,
)


def test_defaults_are_expected():
    s = WatcherSettings()
    assert s.raise_on_exception is False
    assert s.store_traceback is True
    assert s.traceback_limit is None
    assert s.capture_streams is False
    assert s.capture_warnings is True

    assert s.suppressed_exceptions is None
    assert isinstance(s.fatal_exceptions, tuple)
    # Default should include KeyboardInterrupt and SystemExit
    assert builtins.KeyboardInterrupt in s.fatal_exceptions
    assert builtins.SystemExit in s.fatal_exceptions

    assert s.save_on_exception is True
    assert s.exception_save_path_override is None
    assert s.min_seconds_between_exception_saves == 0.0


def test_is_immutable_frozen_dataclass():
    s = WatcherSettings()
    with pytest.raises((AttributeError, TypeError)): # type: ignore[arg-type]
        setattr(s, "raise_on_exception", True)


def test_hashable_and_equality_semantics():
    s1 = WatcherSettings()
    s2 = WatcherSettings()
    # Frozen dataclasses are hashable by default; can be dict keys / set elements
    d = {s1: "ok"}
    assert d[s2] == "ok"  # equal instances have same hash/equality


def test_replace_creates_new_instance_with_overrides():
    s1 = WatcherSettings()
    # dataclasses.replace should yield a new instance, leaving original untouched
    s2 = types.MappingProxyType({})  # just to confirm we're not mutating s1
    s2 = s1.__class__(**{**s1.__dict__, "raise_on_exception": True})
    assert s1 is not s2
    assert s1.raise_on_exception is False
    assert s2.raise_on_exception is True


def test_custom_catch_and_reraise_types_accept_exception_subclasses():
    s = WatcherSettings(
        suppressed_exceptions=(ValueError, RuntimeError),
        fatal_exceptions=(KeyboardInterrupt, SystemExit, MemoryError),
    )
    assert s.suppressed_exceptions == (ValueError, RuntimeError)
    assert MemoryError in s.fatal_exceptions


def test_global_update_affects_current_and_future_contexts():
    # read the baseline, which is the module default
    # current_settings() falls back to the current module default (_default_settings)
    base = current_settings()
    assert base is not None # confirms a WatcherSettings was returned
    assert base.raise_on_exception is False # matches the module default

    # Update the process-wide default by creating a new instance (D1) and
    # also set _settings_var in THIS context to that new instance.
    # here "this context" means "the current execution context of the main test thread".
    new = set_global_settings(raise_on_exception=True)
    assert new.raise_on_exception is True # verify that the value was set.

    # Now, because set_global_settings() called _settings_var.set(new),
    # current_settings() returns the CONTEXT value (D1), not the fallback.
    assert current_settings().raise_on_exception is True

    # a fresh thread should also see the updated default
    seen = {}
    def worker():
        seen["val"] = current_settings().raise_on_exception
    t = threading.Thread(target=worker); t.start(); t.join()
    assert seen["val"] is True
