# pipeline-watcher Usage Guide

This guide describes the recommended `pipeline-watcher` idioms for process orchestrators. It is intended for both human developers and agent/context-file use.

This is not a full API reference. It is the preferred usage pattern.

---

## Core rule

Create one `PipelineReport` per process orchestrator.

Create one `FileReport` per output artifact or unit of work.

Pass the `FileReport` down into the functions that perform the work.

Use `file_step(...)` inside those functions to record meaningful stages.

Let step IDs be derived from clear, unique labels. Do not pass `id=...` unless there is a strong reason.

---

## Imports used in examples

```python
from pathlib import Path

from pipeline_watcher import (
    PipelineReport,
    FileReport,
    pipeline_step,
    pipeline_file,
    file_step,
)
```

The examples below use generic project classes such as `ProcessContext`, `ProcessPlan`, `ItemPlan`, `Processor`, and `Result`. Replace these with your project-specific classes.

---

## 1. Create the pipeline report early

Each process orchestrator should own one `PipelineReport`.

Create it early enough that initialization work can be recorded.

```python
class Orchestrator:
    def __init__(self, process_dir: str | Path):
        self.process_dir = Path(process_dir)

        self.report = PipelineReport(
            label="process-report",
            output_path=self.process_dir / "reports" / "process-report.json",
        )

        with pipeline_step(self.report, "Initialize orchestrator"):
            self.process_ctx = ProcessContext.from_process_dir(self.process_dir)
            self.process_plan = ProcessPlan.from_file(
                self.process_ctx.process_plan_path
            )
            self.process_ctx.enforce_run_contract()

            self.processor = Processor()
            self.renderer = Renderer()
```

Use `pipeline_step(...)` for initialization blocks that should be timed, finalized, and exception-recorded.

Use `add_completed_step(...)` only for simple markers where there is no meaningful block to protect.

```python
self.report.add_completed_step("Loaded static configuration")
```

---

## 2. Put process metadata on `PipelineReport.metadata`

Process-level metadata belongs on the pipeline report.

Do this once during initialization. Do not repeat the same metadata on every file, item, or step.

```python
class Orchestrator:
    def __init__(self, process_dir: str | Path):
        self.process_dir = Path(process_dir)

        self.report = PipelineReport(
            label="process-report",
            output_path=self.process_dir / "reports" / "process-report.json",
        )

        with pipeline_step(self.report, "Initialize orchestrator"):
            self.process_ctx = ProcessContext.from_process_dir(self.process_dir)
            self.process_plan = ProcessPlan.from_file(
                self.process_ctx.process_plan_path
            )
            self.process_ctx.enforce_run_contract()

            self.report.metadata.update({
                "process_dir": str(self.process_dir),
                "process_plan_path": str(self.process_ctx.process_plan_path),
                "output_dir": str(self.process_ctx.output_dir),
            })

            self.processor = Processor()
            self.renderer = Renderer()
```

Metadata should be JSON-serializable.

Good metadata:

```python
self.report.metadata.update({
    "process_dir": str(self.process_dir),
    "mode": "full",
    "item_count": len(self.process_plan.items_to_process),
})
```

Avoid redundant metadata:

```python
# Avoid copying this into every FileReport or StepReport.
file_report.metadata["process_dir"] = str(self.process_dir)
```

---

## 3. Add the process loop

The orchestrator should iterate over units of work and delegate each unit to a method.

```python
class Orchestrator:
    def run(self) -> None:
        for item_plan in self.process_plan.items_to_process:
            self.process_item(item_plan)

        self.report.save()
```

Call `save()` at the end of the run or at major orchestration boundaries.

If `PipelineReport.output_path` is set, `pipeline_file(...)` and `file_step(...)` can autosave on block exit. Do not add manual `save()` calls after every line unless you intentionally want explicit snapshots.

---

## 4. Use one `FileReport` per output artifact

`FileReport` can be read as an output artifact report.

Even if the output is not literally an input file, the `FileReport` is the audit log for one output-producing unit of work.

```python
class Orchestrator:
    def process_item(self, item_plan: ItemPlan) -> None:
        output_path = self.process_ctx.output_dir / item_plan.output_filename

        with pipeline_file(self.report, output_path) as file_report:
            self.load_inputs(item_plan, file_report)
            result = self.compute_result(item_plan, file_report)
            self.render_output(result, file_report)
```

The orchestrator creates the `FileReport` boundary.

The computation functions receive the `FileReport` and append steps to it.

Do not create a separate `PipelineReport` for each item inside one process. Use `FileReport` for per-item timelines.

---

## 5. Pass `FileReport` into computation functions

Every function that performs meaningful work for an item should accept the current `FileReport`.

```python
class Orchestrator:
    def compute_result(
        self,
        item_plan: ItemPlan,
        file_report: FileReport,
    ) -> Result:
        with file_step(file_report, "Compute result") as step:
            result = self.processor.run(item_plan)
            step.note("Computed result")
            return result
```

Do not hide report creation inside low-level computation functions.

Good:

```python
def compute_result(self, item_plan: ItemPlan, file_report: FileReport) -> Result:
    with file_step(file_report, "Compute result"):
        ...
```

Avoid:

```python
def compute_result(self, item_plan: ItemPlan) -> Result:
    # Avoid creating unrelated reports here.
    report = PipelineReport(label="nested-report")
    ...
```

---

## 6. Use steps for meaningful stages

Use `file_step(...)` for stages a human would want to see in the report UI.

```python
class Orchestrator:
    def load_inputs(
        self,
        item_plan: ItemPlan,
        file_report: FileReport,
    ) -> InputData:
        with file_step(file_report, "Load inputs") as step:
            input_data = InputData.load(item_plan.input_path)
            step.note(f"Loaded input: {item_plan.input_path.name}")
            return input_data
```

Step labels should be clear and usually unique within the file report.

Good labels:

```python
with file_step(file_report, "Load inputs"):
    ...

with file_step(file_report, "Compute result"):
    ...

with file_step(file_report, "Render output"):
    ...
```

Avoid repeated generic labels:

```python
with file_step(file_report, "Process"):
    ...

with file_step(file_report, "Process"):
    ...
```

Duplicate labels are technically supported through ID deduplication, but they make reports harder to inspect and usually indicate unclear design.

---

## 7. Let labels drive IDs

Do not pass `id=...` by default.

Preferred:

```python
with file_step(file_report, "Validate result") as step:
    ...
```

Avoid by default:

```python
with file_step(file_report, "Validate result", id="validate_result") as step:
    ...
```

`pipeline-watcher` derives step IDs from labels using normalization and sluggification. It also deduplicates conflicting label-derived IDs.

Use explicit IDs only when an external contract requires them, such as:

* a dashboard expects a stable machine key
* another system links to a specific step ID
* historical reports must preserve old IDs
* a label must change while preserving an existing ID

For ordinary usage, write better labels and let IDs be computed.

---

## 8. Use notes for runtime decisions

Use notes to record important runtime choices that would otherwise be hidden in comments or logs.

```python
class Orchestrator:
    def load_inputs(
        self,
        item_plan: ItemPlan,
        file_report: FileReport,
    ) -> InputData:
        with file_step(file_report, "Load inputs") as step:
            if item_plan.use_cached_input:
                step.note("Using cached input")
                return InputData.load(item_plan.cached_input_path)

            step.note("Computing input from source")
            return InputData.compute_from_source(item_plan.source_path)
```

Use notes for:

* selected branches
* thresholds and decisions
* cache hits and misses
* fallback paths
* compact summaries of external calls

Do not use notes as a dumping ground for large payloads. Put structured details in `metadata` when needed.

```python
step.metadata["input_count"] = len(input_data.records)
step.metadata["source_path"] = str(item_plan.source_path)
```

---

## 9. Use review flags for human-in-the-loop checks

Use `request_review(...)` when a step succeeds technically but requires human attention.

```python
class Orchestrator:
    def validate_result(
        self,
        result: Result,
        file_report: FileReport,
    ) -> None:
        with file_step(file_report, "Validate result") as step:
            step.metadata["confidence"] = result.confidence

            if result.confidence < 0.90:
                step.request_review(
                    f"Result confidence below threshold: {result.confidence:.2f}"
                )
            else:
                step.note("Result confidence meets threshold")
```

Review reasons should be short and actionable.

Put supporting values in `step.metadata`.

---

## 9.1 Check statuses through Status properties

`status` values are `Status` enum instances from `src/pipeline_watcher/core.py`.

When checking lifecycle state, prefer enum-backed properties such as `.pending`, `.running`, `.succeeded`, `.failed`, `.skipped`, and `.terminal`.

Preferred:

```python
if file_report.status.failed:
    ...

if step.terminal:
    ...
```

Avoid hard-coded string checks in application code:

```python
if file_report.status == "failed":
    ...
```

This also applies when reading saved report JSON, but only after hydrating the raw JSON into a report model.

```python
report = PipelineReport.from_file(path)

if report.status.failed:
    ...
```

Direct Pydantic hydration is also acceptable when the JSON text or decoded data is already available:

```python
report = PipelineReport.model_validate_json(path.read_text())
```

Raw `json.loads(...)` dictionaries contain plain strings, so `data["status"].failed` will not work. Hydration restores nested `Status` enum values while preserving JSON-compatible metadata as user data.

---

## 10. Use exception classes to define failure boundaries

`pipeline-watcher` context managers already record exceptions.

On exception, `file_step(...)` records an error summary, optionally records a traceback, marks the step failed, finalizes the step, appends it to the `FileReport`, and autosaves the attached pipeline when configured.

Do not manually duplicate that work at the `pipeline_file(...)` boundary unless you need custom behavior.

The main question is whether the exception should escape the context manager.

Use explicit exception classes to define that policy.

```python
class UnitException(Exception):
    """A single item failed. The process may continue."""


class ProcessException(Exception):
    """The whole process should fail."""
```

Keep these exception classes independent. Do not make `UnitException` inherit from `ProcessException`, or both inherit from a shared project base that is also used in watcher exception policy.

Configure watcher behavior so unit failures are recorded and suppressed.

```python
from pipeline_watcher.settings import set_global_settings

set_global_settings(
    suppressed_exceptions=(UnitException,),
)
```

By default, `fatal_exceptions` already includes `KeyboardInterrupt` and `SystemExit`. Do not manually restate those.

If the project defines an additional process-level exception that should always stop the process, pass it as a project-level fatal exception with `pipeline_fatal_exceptions`.

```python
from pipeline_watcher.settings import set_global_settings

set_global_settings(
    suppressed_exceptions=(UnitException,),
    pipeline_fatal_exceptions=(ProcessException,),
)
```

`pipeline_fatal_exceptions` is added to the built-in system fatal exceptions. The effective `fatal_exceptions` property will still include `KeyboardInterrupt` and `SystemExit`.

`fatal_exceptions=(...)` is supported as a compatibility alias for `pipeline_fatal_exceptions=(...)`, but prefer `pipeline_fatal_exceptions` in new code so the intent is clear and system fatal exceptions are not manually repeated.

With this policy, a `UnitException` raised inside `pipeline_file(...)` or `file_step(...)` is recorded into the report and does not stop the outer process loop.

A `ProcessException`, `KeyboardInterrupt`, or `SystemExit` still stops the process.

---

## 11. Wrap known unit failures as `UnitException`

Unit processors should convert expected failures into `UnitException`.

This gives the watcher one clean exception type to suppress while preserving the original cause through exception chaining.

```python
class Orchestrator:
    def process_item(self, item_plan: ItemPlan) -> None:
        output_path = self.process_ctx.output_dir / item_plan.output_filename

        with pipeline_file(self.report, output_path) as file_report:
            input_data = self.load_inputs(item_plan, file_report)
            result = self.compute_result(item_plan, input_data, file_report)
            self.validate_result(result, file_report)
            self.render_output(result, file_report)
```

The item method does not need a manual `try/except` just to mark the file failed. The context manager handles that.

Wrap known and unknown unit-level failures inside the computation step:

```python
class Orchestrator:
    def compute_result(
        self,
        item_plan: ItemPlan,
        input_data: InputData,
        file_report: FileReport,
    ) -> Result:
        with file_step(
            file_report,
            "Compute result",
            raise_on_exception=True,
        ) as step:
            try:
                result = process_with_known_fail_points(input_data)
                step.note("Computed result")
                return result

            except KnownFailpointException1 as exc:
                step.note("Known fail point 1 encountered")
                raise UnitException("KnownFailpointException1") from exc

            except KnownFailpointException2 as exc:
                step.note("Known fail point 2 encountered")
                raise UnitException("KnownFailpointException2") from exc

            except Exception as exc:
                step.note("Unexpected exception during result computation")
                raise UnitException("Exception") from exc
```

Use this pattern when one failed unit should not fail the whole process.

The original exception is still available through the exception chain and recorded traceback when traceback storage is enabled.

---

## 11.1 Treat exception policy as inherited orchestration policy

Context managers accept settings such as:

```python
raise_on_exception=True
suppressed_exceptions=(...)
pipeline_fatal_exceptions=(...)
```

Use these deliberately.

Although settings can be passed to individual context managers, they are inherited through watcher settings context. Treat `suppressed_exceptions` and `pipeline_fatal_exceptions` as orchestration-level policy unless you have a specific tested reason to override them locally.

Recommended rule:

* set `suppressed_exceptions` once at process startup
* rely on default fatal exceptions for `KeyboardInterrupt` and `SystemExit`
* add project-specific fatal exceptions with `pipeline_fatal_exceptions`
* use independent exception classes for unit-level and process-level failures
* use `raise_on_exception=True` locally when a failed step should stop the current item
* do not rely on deeply nested exception-policy overrides unless the behavior is tested and intentional

`fatal_exceptions=(...)` remains available as a compatibility alias. Prefer `pipeline_fatal_exceptions=(...)` in examples and new code.

---

## 12. Render or persist outputs inside a step

Output writing should usually be recorded as a step.

```python
class Orchestrator:
    def render_output(
        self,
        result: Result,
        file_report: FileReport,
    ) -> None:
        with file_step(file_report, "Render output") as step:
            output_path = self.renderer.render(result)
            step.note(f"Rendered output: {output_path.name}")
            step.metadata["output_path"] = str(output_path)
```

If rendering fails, the step should record the failure and the containing `FileReport` should reflect the failed output artifact.

---

## 13. Complete reference skeleton

Use this as the default pattern for new process orchestrators.

```python
from pathlib import Path

from pipeline_watcher import (
    PipelineReport,
    FileReport,
    pipeline_step,
    pipeline_file,
    file_step,
)


class Orchestrator:
    def __init__(self, process_dir: str | Path):
        self.process_dir = Path(process_dir)

        self.report = PipelineReport(
            label="process-report",
            output_path=self.process_dir / "reports" / "process-report.json",
        )

        with pipeline_step(self.report, "Initialize orchestrator"):
            self.process_ctx = ProcessContext.from_process_dir(self.process_dir)
            self.process_plan = ProcessPlan.from_file(
                self.process_ctx.process_plan_path
            )
            self.process_ctx.enforce_run_contract()

            self.report.metadata.update({
                "process_dir": str(self.process_dir),
                "process_plan_path": str(self.process_ctx.process_plan_path),
                "output_dir": str(self.process_ctx.output_dir),
                "item_count": len(self.process_plan.items_to_process),
            })

            self.processor = Processor()
            self.renderer = Renderer()

    def run(self) -> None:
        for item_plan in self.process_plan.items_to_process:
            self.process_item(item_plan)

        self.report.save()

    def process_item(self, item_plan: ItemPlan) -> None:
        output_path = self.process_ctx.output_dir / item_plan.output_filename

        with pipeline_file(self.report, output_path) as file_report:
            input_data = self.load_inputs(item_plan, file_report)
            result = self.compute_result(item_plan, input_data, file_report)
            self.validate_result(result, file_report)
            self.render_output(result, file_report)

    def load_inputs(
        self,
        item_plan: ItemPlan,
        file_report: FileReport,
    ) -> InputData:
        with file_step(file_report, "Load inputs") as step:
            if item_plan.use_cached_input:
                step.note("Using cached input")
                return InputData.load(item_plan.cached_input_path)

            step.note("Computing input from source")
            return InputData.compute_from_source(item_plan.source_path)

    def compute_result(
        self,
        item_plan: ItemPlan,
        input_data: InputData,
        file_report: FileReport,
    ) -> Result:
        with file_step(
            file_report,
            "Compute result",
            raise_on_exception=True,
        ) as step:
            try:
                result = self.processor.run(item_plan, input_data)
                step.note("Computed result")
                return result

            except KnownFailpointException1 as exc:
                step.note("Known fail point 1 encountered")
                raise UnitException("KnownFailpointException1") from exc

            except KnownFailpointException2 as exc:
                step.note("Known fail point 2 encountered")
                raise UnitException("KnownFailpointException2") from exc

            except Exception as exc:
                step.note("Unexpected exception during result computation")
                raise UnitException("Exception") from exc

    def validate_result(
        self,
        result: Result,
        file_report: FileReport,
    ) -> None:
        with file_step(file_report, "Validate result") as step:
            step.metadata["confidence"] = result.confidence

            if result.confidence < 0.90:
                step.request_review(
                    f"Result confidence below threshold: {result.confidence:.2f}"
                )
            else:
                step.note("Result confidence meets threshold")

    def render_output(
        self,
        result: Result,
        file_report: FileReport,
    ) -> None:
        with file_step(file_report, "Render output") as step:
            output_path = self.renderer.render(result)
            step.note(f"Rendered output: {output_path.name}")
            step.metadata["output_path"] = str(output_path)
```

---

## Anti-patterns

### Do not share one report across separate orchestrators

Use one `PipelineReport` per process orchestrator.

For parent/child orchestration, give the parent and each child process their own report.

This keeps reports easier to inspect and avoids confusing cross-process state.

---

### Do not pass explicit IDs by default

Avoid:

```python
with file_step(file_report, "Render output", id="render_output"):
    ...
```

Prefer:

```python
with file_step(file_report, "Render output"):
    ...
```

---

### Do not duplicate process metadata everywhere

Avoid:

```python
file_report.metadata["process_dir"] = str(self.process_dir)
step.metadata["process_dir"] = str(self.process_dir)
```

Prefer:

```python
self.report.metadata["process_dir"] = str(self.process_dir)
```

---

### Do not use vague repeated step labels

Avoid:

```python
with file_step(file_report, "Process"):
    ...

with file_step(file_report, "Process"):
    ...
```

Prefer:

```python
with file_step(file_report, "Load inputs"):
    ...

with file_step(file_report, "Compute result"):
    ...
```

---

### Do not create reports deep inside computation functions

The orchestrator owns report boundaries.

Low-level functions should receive a `FileReport` and append steps to it.

---

### Do not overuse manual `save()`

Avoid saving after every small operation unless explicit snapshots are required.

Prefer saving at:

* pipeline initialization completion
* item boundaries, when needed
* process completion
* exception boundaries, when configured

---

## Agent rules

When modifying code that uses `pipeline-watcher`, follow these rules:

1. Use one `PipelineReport` per process orchestrator.
2. Use one `FileReport` per output artifact or unit of work.
3. Pass `FileReport` into computation functions.
4. Use `file_step(...)` for meaningful stages inside a unit of work.
5. Do not pass `id=...` unless an external contract requires it.
6. Make step labels clear and usually unique.
7. Put process-level metadata on `PipelineReport.metadata` once.
8. Put step-specific metadata on `StepReport.metadata`.
9. Use notes for branch decisions, cache choices, and compact runtime explanations.
10. Use review flags for ambiguous or low-confidence results.
11. Treat exception policy as inherited orchestration-level behavior.
12. Prefer independent exception classes such as `UnitException` and `ProcessException`.
13. Configure `UnitException` as suppressed when one failed unit should not stop the process.
14. Do not manually restate default fatal exceptions; `KeyboardInterrupt` and `SystemExit` are already fatal by default.
15. Add project-specific fatal exceptions with `pipeline_fatal_exceptions`.
16. Wrap known unit-level failures as `UnitException` from the original exception.
17. Use `raise_on_exception=True` only when a failed step should stop the current item.
18. Do not manually duplicate failure recording that the context managers already perform.
19. Avoid nested report creation unless intentionally creating a separate orchestrator report.
20. Check lifecycle state with `Status` or report properties such as `.failed`, `.succeeded`, and `.terminal` instead of hard-coded status strings.
