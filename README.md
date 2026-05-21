<img title="" src="docs/images/pipeline-watcher-logo-white.webp" alt="pipeline-watcher-logo-white.webp" width="116">

## pipeline-watcher

`pipeline-watcher` is a lightweight, type-safe, thread-safe structured logger for **AI/ML and scientific pipelines**, built on **Pydantic v2**. Instead of free-form text logs, it produces **structured JSON reports** that capture:

- comments and notes

- timing information

- warnings, errors, and tracebacks

- branch decisions and metadata

- Review flags (perfect for HITL systems)

The result is a clean, UI-ready log format that your browser or dashboard can render directly. Included are helper Jinja2 templates for compiling the reports into HTML.

> **In short:** pipeline-watcher gives you structured, type-safe logs for algorithm monitoring—viewable directly in your browser.

---

Key Features

- **Type-safe models** for pipelines, artifacts, and steps

- **Thread-safe runtime state** using `contextvars`

- **Automatic timing** for every step and `FileReport`

- **Minimal-ceremony context managers** for safe logging and exception capture

- **HITL review flags** for ambiguous or low-confidence outputs

- **Robust serialization** handled via Pydantic

- **Minimal dependencies** (Pydantic + standard library only)

- **pipeline-watcher-site**: optional companion for turning logs into navigable HTML

---

## Table of Contents

- [Demo (Quick Glance)](#demo-quick-glance)

- [Features](#features)

- [Quick Start](#quick-start)
  
  - [PipelineReport](#pipelinereport)
    
    - [Create a PipelineReport](#create-a-pipelinereport)
    - [Use a context manager](#use-a-context-manager)
    - [Set progress and save](#set-progress-and-save)
  
  - [Manage Settings](#manage-settings)
    
    - [Global settings](#global-settings)
    - [Local overrides](#local-overrides)
    - [Additional settings](#additional-settings)
  
  - [FileReport](#filereport)
    
    - [Getting started with FileReports](#getting-started-with-filereports)
    - [Use with PipelineReport](#use-with-pipelinereport)
    - [FileReport Summary](#filereport-summary)
  
- [Code Structure](#code-structure)

---

## Demo (Quick Glance)

This example shows the recommended orchestration shape: create one
`PipelineReport` for the process, one `FileReport` for each output artifact,
and pass the `FileReport` into the functions that do the work.

```python
from pathlib import Path

from user_lib import extract_text  # user provided demo

from pipeline_watcher import (
    PipelineReport,
    FileReport,
    pipeline_step,
    pipeline_file,
    file_step,
)


class OcrRunner:
    def __init__(self, input_dir: str | Path, output_dir: str | Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.report = PipelineReport(
            label="OCR of PDFs",
            output_path="reports/progress.json",
        )

        with pipeline_step(self.report, "Initialize OCR run"):
            self.pdf_paths = sorted(self.input_dir.glob("*.pdf"))
            self.report.metadata.update({
                "input_dir": str(self.input_dir),
                "output_dir": str(self.output_dir),
                "pdf_count": len(self.pdf_paths),
            })

    def run(self) -> None:
        for pdf_path in self.pdf_paths:
            self.process_pdf(pdf_path)

        self.report.save()

    def process_pdf(self, pdf_path: Path) -> None:
        output_path = self.output_dir / f"{pdf_path.stem}.txt"

        with pipeline_file(self.report, output_path) as file_report:
            text = self.extract_text(pdf_path, file_report)
            self.write_output(output_path, text, file_report)

    def extract_text(self, pdf_path: Path, file_report: FileReport) -> str:
        with file_step(file_report, "Extract text with OCR") as step:
            extracted_text = extract_text(pdf_path)
            step.note("Performed OCR on the PDF")
            step.metadata["input_path"] = str(pdf_path)
            step.metadata["ocr_quality"] = extracted_text.quality

            # Specific threshold decision with HITL:
            if extracted_text.quality < 0.90:
                step.request_review(
                    f"OCR quality below threshold: {extracted_text.quality:.2f}"
                )
            else:
                step.note("OCR quality meets threshold")

            return extracted_text.text

    def write_output(
        self,
        output_path: Path,
        text: str,
        file_report: FileReport,
    ) -> None:
        with file_step(file_report, "Write text output") as step:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")
            step.metadata["output_path"] = str(output_path)
            step.note(f"Wrote output: {output_path.name}")


OcrRunner("inputs/pdfs", "outputs/text").run()
```

Yields `reports/progress.json` with a pipeline banner and per-artifact timelines.

---

## Features

### **Pipeline → File → Step hierarchy**

`pipeline-watcher` organizes all logs into a strict, type-checked tree:

- **PipelineReport** – high-level report for one process orchestrator

- **FileReport** – record associated with one output artifact or unit of work

- **StepReport** – individual processing steps inside a `FileReport`

Each node holds:

- timestamps

- duration

- comments/notes

- metadata

- warnings/errors

- optional review flags

### **HITL Review**

Any step may request human review with `request_review(...)`, including:

- reason string

- metadata (confidence, heuristics, exceptions)

- a `review.flagged` marker

- success, failure, or skipped lifecycle status

### **Thread-Safe Global State**

Bound pipeline state and watcher settings use `contextvars` to ensure:

- safety in async environments

- safety in multi-thread loops

- no accidental global mutation

### **Serialization**

Serialization is handled internally by Pydantic. All you have to do is call `save()` on a `PipelineReport` instance.

## Quick Start

### PipelineReport

#### Create a PipelineReport

The core object is the `PipelineReport`: one report for one process orchestrator. It is a Pydantic v2 model. Some of the core fields are:

- label: `str` — human-readable run label (**required**)

- output_path: `Optional[Path]` — where the report is saved

- kind: `{"validation", "process", "test"}` — category for UI/routing (defaults to `"process"`)

- metadata: `dict` — process-level metadata for UI, search, and audit context

Only label is mandatory:

```python
from pipeline_watcher import PipelineReport

report = PipelineReport(
    label="ocr-report",
    output_path=logs_dir / "progress.json",
)

report.metadata.update({
    "input_dir": str(input_dir),
    "output_dir": str(output_dir),
})
```

`output_path` may be omitted, but providing one is **strongly recommended**, even for dry runs. When it is set, `pipeline_file()` and `file_step()` autosave the pipeline report on every block exit.

#### Use a context manager

```python
with pipeline_file(report, output_path) as file_report:
    # ... produce or validate the output artifact
    # ... suppose an exception is raised here, e.g.
    raise ValueError("Processing artifact failed due to ...")
```

Under default settings, `pipeline-watcher` records non-fatal exceptions and continues. Fatal exceptions, including `KeyboardInterrupt`, `SystemExit`, and configured `pipeline_fatal_exceptions`, are recorded and re-raised.

On exception, `pipeline-watcher` will:

- **Autofinalize the file report**
  
  - status set to FAILED
  
  - exception type stored in errors
  
  - exception traceback stored in metadata
  
  - duration computed

- **Insert the file report** into `report.files`

- **Autosave the pipeline report** to `output_path`. If configured, exception-only copies can also be written to `exception_save_path_override`, and explicit snapshots can be written with `pipeline_save_to` or `file_save_to`.

#### Set progress and save

```python
from pathlib import Path

from pipeline_watcher import PipelineReport, pipeline_file

report = PipelineReport(
    label="ocr-report",
    output_path=logs_dir / "progress.json",
)

report.set_progress("initialization", 0)
files = [
    file_path
    for file_path in Path("/path/to/pdfs").rglob("*.pdf")
    if file_path.is_file()
]
n_files = len(files)
for j, file_path in enumerate(files):
    output_path = output_dir / f"{file_path.stem}.txt"
    with pipeline_file(report, output_path) as file_report:
        report.set_progress(
            f"processing {file_path.stem}",
            int(100 * j / max(n_files, 1)),
        )
        # process this output artifact...
...
report.save()
```

### Manage Settings

Most pipelines only need two ways to manage the settings: managing global settings, and managing local settings in specific contexts. We've provided convenient tools for both.

#### Global settings

`set_global_settings()` lets you configure watcher behavior once at the start of a script or application:

```python
from pipeline_watcher.settings import set_global_settings

class UnitException(Exception):
    """A single item failed. The process may continue."""


class ProcessException(Exception):
    """The whole process should fail."""


set_global_settings(
    suppressed_exceptions=(UnitException,),
    pipeline_fatal_exceptions=(ProcessException,),
)
```

These become the **default settings** for the entire process.  
All pipelines and context managers inherit these values unless overridden.

`pipeline_fatal_exceptions` is added to the built-in fatal exceptions. You do not need to repeat `KeyboardInterrupt` or `SystemExit`; they are already fatal by default.

---

#### Local overrides

You can override any setting locally for a single artifact or step by passing them
into `pipeline_file()` or `file_step()`:

```python
with pipeline_file(
    report,
    "outputs/a.txt",
    raise_on_exception=True,   # local fail-fast override
):
    ...
```

Local overrides apply **only inside that block** and do not affect anything else.

This allows a simple pattern:

- Set sensible global defaults for your script.
- Override specific behavior only where needed.

That's all you need to get started.  

#### Additional settings:

A few additional settings that might be of interest (see documentation for complete list):

```python
# Exception behavior
raise_on_exception: bool = False
store_traceback: bool = True
traceback_limit: Optional[int] = None
capture_streams: bool = False
capture_warnings: bool = True

# Routing policy
suppressed_exceptions: Optional[Tuple[Type[BaseException], ...]] = None
pipeline_fatal_exceptions: Tuple[Type[BaseException], ...] = ()
fatal_exceptions: Tuple[Type[BaseException], ...]  # computed effective property

# Persistence policy
save_on_exception: bool = True
exception_save_path_override: Optional[str] = None
min_seconds_between_exception_saves: float = 0.0
```

`fatal_exceptions=(...)` is still accepted as a compatibility alias for `pipeline_fatal_exceptions=(...)`, but prefer `pipeline_fatal_exceptions` in new code.

## FileReport

`FileReport` is the core object for tracking the processing lifecycle of one **output artifact or unit of work**. The path can point to an input file, output file, intermediate artifact, or other filesystem location that represents the unit being audited.

It automatically records:

- lifecycle status (`RUNNING → SUCCEEDED/FAILED/SKIPPED`)
- warnings, errors, notes
- HITL review flags
- **computed metadata**:
  - `name` (from `path.name`)
  - `mime_type` (extension-based)
  - `size_bytes` (best-effort filesystem probe)

Most orchestrators should let `pipeline_file(...)` create the `FileReport`, then pass that report into functions that perform meaningful work.

---

### Getting started with FileReports

```python
from pipeline_watcher import FileReport, pipeline_file, file_step


def render_output(result: Result, file_report: FileReport) -> None:
    with file_step(file_report, "Render output") as step:
        output_path = renderer.render(result)
        step.note(f"Rendered output: {output_path.name}")
        step.metadata["output_path"] = str(output_path)


with pipeline_file(report, "outputs/result.json") as file_report:
    result = compute_result(file_report)
    render_output(result, file_report)
```

---

##### ✅ What FileReport gives you “for free”

A `FileReport` automatically:

- **tracks lifecycle state** (`RUNNING` → terminal state)
- **timestamps** `started_at` and `finished_at`
- safely probes:
  - **file name** (`path.name`)
  - **MIME type** (`mimetypes.guess_type`)
  - **size in bytes** (`os.path.getsize`)
- stores arbitrary structured metadata (`metadata` dict)
- holds an ordered list of steps (`steps: List[StepReport]`)
- supports HITL review workflows (`review.flagged`, roll-up to parent)

All computed fields are **safe**: if the path cannot be probed (missing, inaccessible, remote), they simply return `None` without failing your pipeline.

---

##### ✅ Typical lifecycle

```python
fp = FileReport.begin("/tmp/some.pdf")

# ... your processing logic ...
fp.add_completed_step("Initial validation")
fp.warn("Low resolution detected")

if not fp.ok:
    fp.fail("One or more checks failed")

fp.end()       # infers success/failure if not already terminal
fp_dict = fp.model_dump()  # ready for JSON serialization
```

---

#### Use with PipelineReport

`FileReport` objects are often created automatically by the higher-level `pipeline_file(...)` context manager:

```python
with pipeline_file(report, "outputs/a.txt") as fr:
    fr.note("Running OCR")
    # raise ValueError("...") → automatically recorded, autosaved
```

This is the recommended way to use `FileReport` in real pipelines, since it captures:

- stdout / stderr (optional)
- warnings
- exceptions + traceback
- duration
- banner updates
- autosave behavior when attached to a `PipelineReport` with `output_path`

---

#### FileReport Summary

`FileReport` is your per-artifact audit log, designed to:

- require almost no input
- behave predictably
- serialize cleanly
- automatically collect actionable metadata

Use `FileReport.begin(path)` only when manually controlling artifact-level logic. In real orchestrators, prefer `pipeline_file()` so exception handling, finalization, autosave, and inherited settings are applied consistently.

## Code Structure

### Abstract Base (optional pattern)

`StepReport` and `FileReport` share a common shape (status, timestamps, percent, notes, errors, metadata, optional review flag). If you want to enforce this across custom report types, you can introduce an abstract base (`ReportBase(abc.ABC)`) that declares:

- `ok: bool` – whether the unit ultimately succeeded
- `end()` – auto-finalize based on `ok`

### StepReport

Represents a single unit of work (e.g., `"parse"`, `"validate_index"`).

- Fields: `status`, `percent`, `started_at/finished_at`, `notes`, `warnings`, `errors`, `checks`, `metadata`, optional `review`.
- Lifecycle: `begin()`, `start()`, `succeed()`, `fail()`, `skip()`, `end()`.
- `ok` property determines success when `end()` is used.

### FileReport

Represents processing for one output artifact or unit of work (ordered list of `StepReport`s).

- Metadata: `file_id`, `path`, `name`, `size_bytes`, `mime_type`.
- Progress rolls up from steps.
- Fluent API: `append_step(step)` returns `self`.

#### Convenience Methods

To log progress with minimal ceremony:

- `add_completed_step(label, note=None, metadata=None)` – add a SUCCEEDED step.
- `add_failed_step(label, reason=None, metadata=None)` – add a FAILED step.
- `add_skipped_step(label, reason=None, metadata=None)` – add a SKIPPED step.
- `add_review_step(label, reason=None, metadata=None, mark_success=True)` – SUCCEEDED + HITL request by default.

```python
fr = FileReport.begin("outputs/doc1.pdf")
fr.add_completed_step("Verified file exists")\
  .add_review_step("Low OCR confidence", reason="score=0.42")\
  .add_failed_step("Render PDF", reason="timeout")
```

---

## Comments → Structured Notes (debuggable “comment replacement”)

Use `step.note(...)` for concise runtime decisions that should ship to the UI.
This turns what you would normally leave in logs or comments into a reviewable
narrative.

```python
st = StepReport.begin("Calculate result")
result = some_calculation()

if result > 100:
    st.note("result > 100; taking branch A")
else:
    st.note("result <= 100; taking branch B")

st.end()  # infers SUCCEEDED (no failed checks or errors)
```

The same helper style is available for warnings and errors:

```python
st.note("raw_result=%s" % result)
st.warn("slow path used")
st.error("contract violated")
```

This pattern makes runtime behavior **discoverable** in the UI without attaching a
debugger or tailing logs.

---

## Context Managers for Exception Handling & Debugging

Context managers simplify the *try/except/finally* ceremony and guarantee that steps
and `FileReport`s are finalized, even on early returns or errors. They also record `duration_ms`
for quick SLO/troubleshooting.

### `pipeline_step` (pipeline-level step)

```python
from pipeline_watcher import pipeline_step

with pipeline_step(report, "Validate input manifest") as st:
    st.add_check("manifest_present", ok=True)
    st.add_check("ids_unique", ok=False, detail="3 duplicates")  # will cause FAILED
# The step is appended, finalized, and timed automatically.
```

### `pipeline_file` (per-artifact block)

```python
from pipeline_watcher import pipeline_file

with pipeline_file(
    report,
    "outputs/a.docx",
    raise_on_exception=False,   # record failure and continue (default)
    save_on_exception=True      # save exception copy when configured
) as fr:
    fr.add_completed_step("Verified file exists")
    risky_work()  # if this raises, fr is recorded as FAILED
```

### `file_step` (per-step inside a FileReport)

```python
from pipeline_watcher import file_step

with pipeline_file(report, "outputs/b.csv") as fr:
    with file_step(fr, "Calculate result") as st:
        r = some_calculation()
        st.note(f"raw result={r}")
        if r > 100:
            st.note("branch A")
            # ... do A ...
        else:
            st.note("branch B")
            # ... do B ...
```

All context managers support exception handling: they **record** the failure (status, error,
traceback), **finalize** the object, and by default **do not re-raise** non-fatal
exceptions, so your pipeline can continue to the next artifact/step. Fatal exceptions
are always re-raised. You can opt into fail-fast with `raise_on_exception=True`,
and then allow specific unit-level failures to continue with `suppressed_exceptions`.
`pipeline_file` and `file_step` autosave the pipeline to `output_path` on every exit when
the `FileReport` is attached to a pipeline. Exception-only copies use
`exception_save_path_override`, while explicit snapshots can be written with
`pipeline_save_to`, `file_save_to`, or `step_save_to`.

### Binding (less boilerplate)

Bind a report once so helpers don’t need the `report` parameter:

```python
from pipeline_watcher import pipeline_file
from pipeline_watcher.core import bind_pipeline

with bind_pipeline(report):
    with pipeline_file(None, "outputs/b.docx") as fr:
        # Any nested helpers can discover the bound pipeline
        ...
```

> Under the hood, binding uses `contextvars` for thread/async safety.

---

## Jinja2 Templates (starter idea)

You can pass the Pydantic models (or their dicts) straight to templates.  
Consider providing templates like:

```
templates/
├─ pipeline_summary.html.j2  # banner, percent, message, updated_at
├─ artifact_table.html.j2    # artifact rows with status/HITL badges
└─ artifact_detail.html.j2   # steps, notes/checks/errors per artifact
```

Example snippet:

```jinja2
<h1>Pipeline {{ report.label }} — {{ report.stage }}</h1>
<p>Status: {{ report.percent }}% — {{ report.message }}</p>

<ul>
{% for f in report.files %}
  <li>
    Artifact {{ f.name or f.file_id }}: {{ f.status }}
    {% if f.review and f.review.flagged %} 🔎 Review requested {% endif %}
    <ul>
      {% for s in f.steps %}
        <li>
          {{ s.label }} — {{ s.status }}
          {% if s.review and s.review.flagged %}
            🔎 Requires review: {{ s.review.reason }}
          {% endif %}
        </li>
      {% endfor %}
    </ul>
  </li>
{% endfor %}
</ul>
```

---

## Parsing JSON back into Pydantic models (reconstruct & render)

To go beyond “just JSON”, reconstruct the full report object and pass it
to templates or programmatic tooling.

```python
import json
from pipeline_watcher import PipelineReport

def load_report(path: str) -> PipelineReport:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Validate & construct the full object graph (PipelineReport → FileReport → StepReport)
    return PipelineReport.model_validate(data)  # pydantic v2

# Usage
report = load_report("reports/progress.json")
# now pass `report` directly to Jinja2 or other code
```

If you need resilience for older schema versions, add a small migration layer before
`model_validate()` (e.g., `if data.get("report_version") == "v1": transform(data)`).
Pydantic will do the heavy lifting for nested models and enums.

---

## Persistence

- JSON-friendly: use Pydantic serialization (`model_dump()` / `model_dump_json()`) on any report.
- Atomic helpers: `dump_report(path, report)` or `PipelineReport.save(output_path)`.

```python
from pipeline_watcher import dump_report

dump_report("reports/progress.json", report)  # atomic helper
# or save atomically from the object:
report.output_path = "reports/progress.json"
report.save()
```

---

## When to reach for heavier tools

- **Orchestration (Prefect/Dagster)**: scheduling, retries, distributed runs, and fleet UIs.
- **Experiment tracking (MLflow/W&B)**: params, metrics, artifacts, and comparisons.
- **Data validation (Great Expectations)**: formalized expectations & HTML data docs.

`pipeline-watcher` stays intentionally small: append-only, JSON-first, HITL‑aware, and UI-ready.

---
