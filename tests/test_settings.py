import pytest

from pipeline_watcher import PipelineReport, Status, file_step, pipeline_file
from pipeline_watcher import settings as settings_module
from pipeline_watcher.settings import (
    WatcherSettings,
    current_settings,
    set_global_settings,
    use_settings,
    with_overrides,
)


class UnitException(Exception):
    """A single item failed, but the process may continue."""


class ProcessException(Exception):
    """The whole process should fail."""


@pytest.fixture(autouse=True)
def restore_global_settings():
    original_default = settings_module._default_settings
    settings_module._settings_var.set(original_default)
    try:
        yield
    finally:
        settings_module._default_settings = original_default
        settings_module._settings_var.set(original_default)


def test_default_settings_keep_system_fatal_exceptions():
    settings = current_settings()

    assert KeyboardInterrupt in settings.fatal_exceptions
    assert SystemExit in settings.fatal_exceptions
    assert settings.should_raise(KeyboardInterrupt())
    assert settings.should_raise(SystemExit())
    assert not settings.should_raise(ValueError("ordinary failure"))


def test_raise_on_exception_and_suppressed_exceptions_matrix():
    default_settings = WatcherSettings()
    fail_fast = WatcherSettings(raise_on_exception=True)
    suppress_unit = WatcherSettings(
        raise_on_exception=True,
        suppressed_exceptions=(UnitException,),
    )

    assert not default_settings.should_raise(ValueError("ordinary failure"))
    assert fail_fast.should_raise(ValueError("ordinary failure"))
    assert not suppress_unit.should_raise(UnitException("known fail point"))
    assert suppress_unit.should_raise(ValueError("ordinary failure"))
    assert suppress_unit.suppression_breadcrumb(UnitException("known fail point")) == (
        "suppressed raise_on_exception for UnitException via suppressed_exceptions"
    )


def test_fatal_exceptions_override_suppression():
    settings = WatcherSettings(
        raise_on_exception=True,
        suppressed_exceptions=(ProcessException,),
        pipeline_fatal_exceptions=(ProcessException,),
    )

    assert settings.is_suppressed(ProcessException("stop process"))
    assert settings.is_fatal(ProcessException("stop process"))
    assert settings.should_raise(ProcessException("stop process"))


def test_pipeline_fatal_exceptions_are_deduped_with_system_defaults():
    with use_settings(
        pipeline_fatal_exceptions=(
            KeyboardInterrupt,
            ProcessException,
            ProcessException,
        ),
    ) as settings:
        fatal_exceptions = settings.fatal_exceptions

    assert fatal_exceptions.count(KeyboardInterrupt) == 1
    assert fatal_exceptions.count(SystemExit) == 1
    assert fatal_exceptions.count(ProcessException) == 1


def test_use_settings_layers_and_restores_exception_policy():
    outer = current_settings()

    with use_settings(
        raise_on_exception=True,
        suppressed_exceptions=(UnitException,),
    ) as run_settings:
        assert current_settings() is run_settings
        assert run_settings.raise_on_exception
        assert not run_settings.should_raise(UnitException("known fail point"))

        with use_settings(suppressed_exceptions=()) as inner_settings:
            assert current_settings() is inner_settings
            assert inner_settings.raise_on_exception
            assert inner_settings.should_raise(UnitException("known fail point"))

        assert current_settings() is run_settings
        assert not current_settings().should_raise(UnitException("known fail point"))

    assert current_settings() is outer


def test_with_overrides_returns_new_settings_without_mutating_base():
    base = WatcherSettings()

    derived = with_overrides(
        base,
        raise_on_exception=True,
        pipeline_fatal_exceptions=(ProcessException,),
    )

    assert derived is not base
    assert derived.raise_on_exception
    assert ProcessException in derived.fatal_exceptions
    assert not base.raise_on_exception
    assert ProcessException not in base.fatal_exceptions


def test_set_global_settings_applies_agents_exception_policy_and_is_reset():
    settings = set_global_settings(
        suppressed_exceptions=(UnitException,),
        pipeline_fatal_exceptions=(ProcessException,),
    )

    assert current_settings() is settings
    assert current_settings().is_suppressed(UnitException("known fail point"))
    assert current_settings().should_raise(ProcessException("stop process"))
    assert current_settings().pipeline_fatal_exceptions == (ProcessException,)
    assert KeyboardInterrupt in current_settings().fatal_exceptions
    assert SystemExit in current_settings().fatal_exceptions

    with use_settings(raise_on_exception=True) as run_settings:
        assert not run_settings.should_raise(UnitException("known fail point"))
        assert run_settings.should_raise(ValueError("ordinary failure"))


def test_fatal_exceptions_alias_remains_supported_for_compatibility():
    settings = set_global_settings(
        suppressed_exceptions=(UnitException,),
        fatal_exceptions=(ProcessException,),
    )

    assert settings.pipeline_fatal_exceptions == (ProcessException,)
    assert settings.should_raise(ProcessException("stop process"))
    assert KeyboardInterrupt in settings.fatal_exceptions
    assert SystemExit in settings.fatal_exceptions


@pytest.mark.parametrize(
    "override",
    [
        {"suppressed_exceptions": ("not-an-exception",)},
        {"fatal_exceptions": ("not-an-exception",)},
        {"pipeline_fatal_exceptions": ("not-an-exception",)},
        {"_system_fatal_exceptions": ("not-an-exception",)},
    ],
)
def test_invalid_exception_policy_values_are_rejected(override):
    with pytest.raises(TypeError, match="Expected exception classes"):
        with use_settings(**override):
            pass

    with pytest.raises(TypeError, match="Expected exception classes"):
        with_overrides(current_settings(), **override)

    with pytest.raises(TypeError, match="Expected exception classes"):
        set_global_settings(**override)


def test_context_managers_inherit_agents_exception_policy():
    report = PipelineReport(label="process-report")

    with use_settings(
        suppressed_exceptions=(UnitException,),
        pipeline_fatal_exceptions=(ProcessException,),
    ):
        with pipeline_file(report, "outputs/unit-result.json", file_id="unit-result") as file_report:
            with file_step(file_report, "Compute result", raise_on_exception=True):
                raise UnitException("known fail point")

        with pytest.raises(ProcessException, match="stop process"):
            with pipeline_file(
                report,
                "outputs/process-result.json",
                file_id="process-result",
            ):
                raise ProcessException("stop process")

    assert [file.file_id for file in report.files] == [
        "unit-result",
        "process-result",
    ]

    unit_file = report.files[0]
    assert unit_file.failed
    step = unit_file.steps[0]
    assert step.failed
    assert "UnitException: known fail point" in step.errors
    assert step.warnings == [
        "suppressed raise_on_exception for UnitException via suppressed_exceptions",
    ]

    recorded_file = report.files[1]
    assert recorded_file.failed
    assert "ProcessException: stop process" in recorded_file.errors
    assert "Unhandled exception while processing file" in recorded_file.errors
    assert report.status is Status.FAILED
