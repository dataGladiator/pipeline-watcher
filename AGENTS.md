# pipeline-watcher Usage Guide

This guide describes the preferred `pipeline-watcher` patterns for process orchestrators. It is intended for developers and agent/context-file use.

This is not a full API reference. It is the recommended usage model.

---

## Core model

Use one `PipelineReport` per process orchestrator.

Use one `FileReport` per output artifact or output-producing unit of work.

Pass the active `FileReport` down into the functions that perform the work.

Use `file_step(...)` inside those functions to record meaningful stages.

Let step IDs be derived from clear, unique labels. Do not pass `id=...` unless an external contract requires a stable machine key.

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
from pipeline_watcher.settings import use_settings
```

The examples use placeholder project types such as `ProcessContext`, `ProcessPlan`, `ItemPlan`, `InputData`, `Processor`, `Renderer`, and `Result`. Replace them with project-specific classes.

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

            self.report.metadata.update({
                "process_dir": str(self.process_dir),
                "process_plan_path": str(self.process_ctx.process_plan_path),
                "output_dir": str(self.process_ctx.output_dir),
                "item_count": len(self.process_plan.items_to_process),
            })

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

Do this once during initialization. Do not copy the same metadata into every file, item, or step.

Good:

```python
self.report.metadata.update({
    "process_dir": str(self.process_dir),
    "mode": "full",
    "item_count": len(self.process_plan.items_to_process),
})
```

Avoid:

```python
# Avoid copying process-level metadata into every FileReport or StepReport.
file_report.metadata["process_dir"] = str(self.process_dir)
step.metadata["process_dir"] = str(self.process_dir)
```

Metadata should be JSON-serializable.

---

## 3. Configure exception policy at the orchestration boundary

Exception routing is orchestration policy. Configure it once at the start of a process or run.

Use independent exception classes for unit-level failures and process-level failures.

```python
class UnitException(Exception):
    """A single item failed. The process may continue."""


class ProcessException(Exception):
    """The whole process should fail."""
```

For application scripts or one-off processes, a top-level global setting is acceptable:

```python
from pipeline_watcher.settings import set_global_settings

set_global_settings(
    suppressed_exceptions=(UnitException,),
    pipeline_fatal_exceptions=(ProcessException,),
)
```

For libraries, tests, concurrent pipelines, or code that should not mutate interpreter-wide defaults, prefer a run-scoped settings context:

```python
with use_settings(
    suppressed_exceptions=(UnitException,),
    pipeline_fatal_exceptions=(ProcessException,),
):
    orchestrator.run()
```

`pipeline_fatal_exceptions` is added to the built-in system fatal exceptions. Do not manually restate `KeyboardInterrupt` or `SystemExit`; they are already fatal by default.

Avoid passing exception-policy settings such as `raise_on_exception`, `suppressed_exceptions`, or `pipeline_fatal_exceptions` directly to `pipeline_step(...)`, `pipeline_file(...)`, or `file_step(...)` in ordinary application code.

Those options are ambient watcher settings for the current logical context, not true properties of only that file or step. Nested watcher calls may observe them. Use block-level settings overrides only in rare, tested cases.

---

## 4. Add the process loop

The orchestrator should iterate over units of work and delegate each unit to a method.

```python
class Orchestrator:
    def run(self) -> None:
        for item_plan in self.process_plan.items_to_process:
            self.process_item(item_plan)

        self.report.save()
```

Default to one final `self.report.save()` at process completion.

If `PipelineReport.output_path` is set, `pipeline_file(...)` and `file_step(...)` can autosave at block boundaries. Add manual saves only at intentional checkpoint, recovery, or orchestration boundaries.

---

## 5. Use one `FileReport` per output artifact or unit of work

`FileReport` is the audit log for one output-producing unit.

Even when the output is not literally an input file, the `FileReport` represents the timeline for one result, artifact, or item.

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

When one unit of work produces multiple tightly related files, use one `FileReport` for the unit unless each output needs its own independent audit trail.

Do not create a separate `PipelineReport` for each item inside one process. Use `FileReport` for per-item timelines.

---

## 6. Pass `FileReport` into computation functions

Every function that performs meaningful work for an item should accept the current `FileReport`.

Good:

```python
def compute_result(
    self,
    item_plan: ItemPlan,
    input_data: InputData,
    file_report: FileReport,
) -> Result:
    with file_step(file_report, "Compute result") as step:
        result = self.processor.run(item_plan, input_data)
        step.note("Computed result")
        return result
```

Avoid hiding report creation inside low-level computation functions.

```python
def compute_result(self, item_plan: ItemPlan) -> Result:
    # Avoid creating unrelated reports inside lower-level code.
    report = PipelineReport(label="nested-report")
    ...
```

The orchestrator owns report boundaries. Lower-level functions append steps to the active `FileReport`.

---

## 7. Use steps for meaningful stages

Use `file_step(...)` for stages a human would want to see in the report UI.

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

Avoid vague repeated labels:

```python
with file_step(file_report, "Process"):
    ...

with file_step(file_report, "Process"):
    ...
```

Duplicate labels are technically supported through ID deduplication, but they make reports harder to inspect and usually indicate unclear design.

---

## 8. Let labels drive IDs

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

## 9. Use notes, metadata, and review flags deliberately

Use notes to record important runtime choices that would otherwise be hidden in comments or logs.

Good uses for `step.note(...)`:

* selected branches
* thresholds and decisions
* cache hits and misses
* fallback paths
* compact summaries of external calls

```python
with file_step(file_report, "Load inputs") as step:
    if item_plan.use_cached_input:
        step.note("Using cached input")
        return InputData.load(item_plan.cached_input_path)

    step.note("Computing input from source")
    return InputData.compute_from_source(item_plan.source_path)
```

Do not use notes as a dumping ground for large payloads. Put structured details in metadata when needed.

```python
step.metadata["input_count"] = len(input_data.records)
step.metadata["source_path"] = str(item_plan.source_path)
```

Use `request_review(...)` when a step succeeds technically but requires human attention.

```python
with file_step(file_report, "Validate result") as step:
    step.metadata["confidence"] = result.confidence

    if result.confidence < 0.90:
        step.request_review(
            f"Result confidence below threshold: {result.confidence:.2f}"
        )
    else:
        step.note("Result confidence meets threshold")
```

Review reasons should be short and actionable. Put supporting values in `step.metadata`.

---

## 10. Wrap unit-level failures as `UnitException`

Unit processors should convert expected item-level failures into `UnitException`.

This gives the watcher one clean exception type to record and suppress while preserving the original cause through exception chaining.

```python
class Orchestrator:
    def compute_result(
        self,
        item_plan: ItemPlan,
        input_data: InputData,
        file_report: FileReport,
    ) -> Result:
        with file_step(file_report, "Compute result") as step:
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
```

With `UnitException` configured as suppressed at the orchestration boundary, one failed item is recorded and the outer process can continue to the next item.

A `ProcessException`, `KeyboardInterrupt`, or `SystemExit` still stops the process.

Do not manually duplicate failure recording that the context managers already perform. On exception, watcher context managers record the error summary, optionally record a traceback, mark the step or file failed, finalize it, and save when configured.

---

## 11. Render or persist outputs inside a step

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

## 12. Load saved reports as typed report objects

When reading a saved report, prefer loading it with `PipelineReport.from_file(path)`.

This restores the saved JSON into the full typed report model:

* `PipelineReport`
* `FileReport`
* `StepReport`
* `Status` enum values
* convenience properties such as `.failed`, `.succeeded`, `.running`, `.skipped`, and `.terminal`

Preferred:

```python
from pathlib import Path

report = PipelineReport.from_file(Path("reports/process-report.json"))

if report.status.failed:
    ...

for file_report in report.files:
    if file_report.status.failed:
        ...

    for step in file_report.steps:
        if step.terminal:
            ...
```

This is more useful than ad hoc JSON or dictionary parsing because the loaded report keeps the same typed behavior as reports created during execution.

Avoid raw dictionaries for report inspection unless you specifically need low-level JSON processing:

```python
import json

raw = json.loads(path.read_text())

# Avoid this in application code.
if raw["status"] == "failed":
    ...
```

Raw `json.loads(...)` dictionaries contain plain strings and plain nested dictionaries, so `raw["status"].failed` will not work, and nested file and step records will not have `FileReport` or `StepReport` methods and properties.

If the JSON text or decoded data is already available, direct Pydantic hydration is also acceptable:

```python
report = PipelineReport.model_validate_json(path.read_text())

if report.status.failed:
    ...
```

Use typed loading whenever report status, file status, step status, metadata, notes, review flags, errors, warnings, or nested report structure will be inspected programmatically.

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
from pipeline_watcher.settings import use_settings


class UnitException(Exception):
    """A single item failed. The process may continue."""


class ProcessException(Exception):
    """The whole process should fail."""


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
        with file_step(file_report, "Compute result") as step:
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


def main(process_dir: str | Path) -> None:
    orchestrator = Orchestrator(process_dir)

    with use_settings(
        suppressed_exceptions=(UnitException,),
        pipeline_fatal_exceptions=(ProcessException,),
    ):
        orchestrator.run()
```

---

## Anti-patterns

### Do not share one report across separate orchestrators

Use one `PipelineReport` per process orchestrator.

For parent/child orchestration, give the parent and each child process their own report.

---

### Do not create reports deep inside computation functions

The orchestrator owns report boundaries.

Low-level functions should receive a `FileReport` and append steps to it.

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

### Do not pass local exception-policy overrides in normal code

Avoid:

```python
with file_step(file_report, "Compute result", raise_on_exception=True):
    ...
```

Prefer configuring exception routing once at the run or orchestrator boundary.

```python
with use_settings(
    suppressed_exceptions=(UnitException,),
    pipeline_fatal_exceptions=(ProcessException,),
):
    orchestrator.run()
```

---

### Do not overuse manual `save()`

Avoid saving after every small operation unless explicit snapshots are required.

Prefer saving at:

* process completion
* intentional checkpoint boundaries
* recovery boundaries
* exception/autosave boundaries configured by the watcher

---

### Do not inspect saved reports as raw dictionaries by default

Avoid:

```python
raw = json.loads(path.read_text())
if raw["status"] == "failed":
    ...
```

Prefer:

```python
report = PipelineReport.from_file(path)
if report.status.failed:
    ...
```

---

## Agent rules

When modifying code that uses `pipeline-watcher`, follow these rules:

1. Use one `PipelineReport` per process orchestrator.
2. Use one `FileReport` per output artifact or output-producing unit of work.
3. Pass `FileReport` into computation functions.
4. Use `file_step(...)` for meaningful stages inside a unit of work.
5. Do not pass `id=...` unless an external contract requires it.
6. Make step labels clear and usually unique.
7. Put process-level metadata on `PipelineReport.metadata` once.
8. Put step-specific metadata on `StepReport.metadata`.
9. Use notes for branch decisions, cache choices, fallback paths, thresholds, and compact runtime explanations.
10. Use review flags for ambiguous, low-confidence, or human-review-required results.
11. Configure exception routing once at the run or orchestrator boundary.
12. Prefer independent exception classes such as `UnitException` and `ProcessException`.
13. Configure `UnitException` as suppressed when one failed unit should not stop the process.
14. Do not manually restate default fatal exceptions; `KeyboardInterrupt` and `SystemExit` are already fatal by default.
15. Add project-specific fatal exceptions with `pipeline_fatal_exceptions`.
16. Wrap known unit-level failures as `UnitException` from the original exception.
17. Do not pass exception-policy overrides to individual watcher context managers in normal usage.
18. Do not manually duplicate failure recording that watcher context managers already perform.
19. Avoid nested report creation unless intentionally creating a separate orchestrator report.
20. Load saved reports with `PipelineReport.from_file(path)` before inspecting them programmatically.
21. Check lifecycle state with `Status` or report properties such as `.failed`, `.succeeded`, `.running`, `.skipped`, and `.terminal` instead of hard-coded status strings.
22. Prefer `use_settings(...)` for run-scoped settings in libraries, tests, or concurrent pipelines.
23. Use `set_global_settings(...)` only for top-level scripts or one-off process entry points where interpreter-wide mutation is acceptable.
24. Default to one final `report.save()` at process completion, plus intentional checkpoints or configured autosave behavior.
25. Keep saved-report inspection typed; avoid ad hoc JSON or dictionary parsing unless low-level JSON processing is specifically required.
