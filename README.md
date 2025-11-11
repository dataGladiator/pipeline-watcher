<img title="" src="docs/images/pipeline-watcher-logo-white.webp" alt="pipeline-watcher-logo-white.webp" width="116">

# pipeline-watcher

`pipeline-watcher` is a lightweight, type-safe, thread-safe structured logger for **AI/ML and scientific pipelines**, built on **Pydantic v2**. Instead of free-form text logs, it produces **structured JSON reports** that capture:

- comments and notes

- timing information

- warnings, errors, and tracebacks

- branch decisions and metadata

- HITL (Human-In-The-Loop) review flags

The result is a clean, UI-ready log format that your browser or dashboard can render directly. Included are helper Jinja2 templates for compiling the reports into HTML.

> **In short:** pipeline-watcher gives you structured, type-safe logs for algorithm monitoring—viewable directly in your browser.

---

Key Features

- **Type-safe models** for batches, files, and steps

- **Thread-safe runtime state** using `contextvars`

- **Automatic timing** for every step and file

- **Minimal-ceremony context managers** for safe logging and exception capture

- **HITL review flags** for ambiguous or low-confidence outputs

- **Robust serialization** handled via Pydantic’s `model_dump_json()`

- **Zero non-standard dependencies** (Pydantic + standard library only)

- **pipeline-watcher-site**: optional companion for turning logs into navigable HTML

---

# Table of Contents

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
    
    - [Getting started with files](#getting-started-with-files)
    - [Use with PipelineReport](#use-with-pipelinereport)
    - [FileReport Summary](#filereport-summary)
  
  - [StepReport](#stepreport)
    
    - [Recording steps inside a file](#recording-steps-inside-a-file)
    - [Failure behavior](#failure-behavior)

- [Code Structure](#code-structure)

- [HITL Review](#hitl-review)

- [Installation](#installation)

---

## Demo (Quick Glance)

This example shows:

- Iterating over a directory of PDFs

- Attaching user notes and metadata

- Automatic timing and exception handling via context managers

- Raising HITL review steps when conditions fail

- Using the `file_step` helper to minimize boilerplate

```python
from pathlib import Path
from user_lib import extract_text, index_text # user provided demo 
from pipeline_watcher import PipelineReport, pipeline_file, file_step

report = PipelineReport(label="OCR of pdfs",
                        kind="process",
                        output_path="reports/progress.json")

for file_path in Path("inputs/pdfs").glob("*.pdf"):

    # The context manager handles exceptions and auto-finalizes logs
    with pipeline_file(report, file_path) as file_report:
        with file_step(file_report, "extract_text", label="Extract text (OCR)") as step:
            extracted_text = extract_text(file_path) # user provided function
            step.notes.append("Performed OCR on the PDF")
            step.metadata["ocr_quality"] = extracted_text.quality

            # Specific threshold decision with HITL:
            if extracted_text.quality < 0.90:
                step.request_review(f"OCR quality below threshold (0.90): {extracted_text.quality:.2f} ")
            else:
                step.notes.append("OCR quality meets threshold")
            # continue processing file ...

# Persist the whole batch report (direct write to output_path)
report.save()
```

Yields `reports/progress.json` with a batch banner and per-file timelines.

---

## Features

### **Batch → File → Step hierarchy**

`pipeline-watcher` organizes all logs into a strict, type-checked tree:

- **BatchReport** – high-level banner for an entire run

- **FileReport** – record associated with a single input file

- **StepReport** – individual processing steps inside a file

Each node holds:

- timestamps

- duration

- comments/notes

- metadata

- warnings/errors

- optional review flags

### **HITL Review**

Any step may request review by adding a *review step*, including:

- reason string

- metadata (confidence, heuristics, exceptions)

- mark as required / optional

- success or failure indicators

### **Thread-Safe Global State**

Settings such as `current_report` use `contextvars` to ensure:

- safety in async environments

- safety in multi-thread loops

- no accidental global mutation

### **Serialization**

Serialization is handled internally by Pydantic. All you have to do is call save on the a PipelineReport instance.

## Quick Start

### PipelineReport

#### Create a PipelineReport

The core object is the PipelineReport object. This object is actually a Pydantic v2 data model. Some of the core fields on this model are:

- label: `str` — human-readable run label (**required**)

- output_path: `Optional[Path]` — where the report is saved

- kind: `{"validation", "process", "test"}` — category for UI/routing (defaults to `"process"`)

Only label is mandatory:

```python
from pipeline_watcher import PipelineReport
report = PipelineReport(label="ocr-report",
                        output_path=logs_dir / "progress.json")
```

`output_path` may be omitted, but providing one is **strongly recommended**, even for dry runs—especially if you intend to use context managers, since pipeline-watcher will autosave on exceptions.

#### Use a context manager

```python
with pipeline_file(report, path_to_file) as file_report:
    # ... process file
    # ... suppose an exception is raised here, e.g.
    raise ValueError("Processing file failed due to ...")
```

Under default settings, pipeline-watcher will:

- **Autofinalize the file report**
  
  - status set to FAILED
  
  - exception type stored in errors
  
  - exception traceback stored in metadata
  
  - duration computed

- **Insert the file report** into `report.files`

- **Autosave the pipeline report** to `output_path` (or to the override configured in `WatcherSettings`or passed to `pipeline_file`).

#### Set progress and save

```python
from pipeline_watcher import PipelineReport
report = PipelineReport(label="ocr-report",
                        output_path=logs_dir / "progress.json")

report.set_progress("initialization", 0)
files = [file_path in Path("/path/to/pdfs").rglob(f"*.pdf") if file_path.is_file()]
n_files = len(files)
for j, file_path in enumerate(files):
    with pipeline_file(report, file_path) as file_report:
        report.set_progress("loading file {file_path.stem}", j // n_files)
        # process files...
...
report.save()
```

### Manage Settings

Most pipelines only need two ways to manage the settings: managing global settings, and managing local settings in specific contexts. We've provided convenient tools for both.

#### Global settings

`set_global_settings()` lets you configure watcher behavior once at the start of a script or application:

```python
from pipeline_watcher.settings import set_global_settings

# Fail-fast mode (recommended for development & CI)
set_global_settings(raise_on_exception=True)
```

These become the **default settings** for the entire process.  
All pipelines and context managers inherit these values unless overridden.

---

#### Local overrides

You can override any setting locally for a single file or step by passing them
into `pipeline_file()` or `file_step()`:

```python
with pipeline_file(report,
                   path="inputs/a.pdf",
                   raise_on_exception=False):   # local override
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

# Routing policy
suppressed_exceptions: Optional[Tuple[Type[BaseException], ...]] = None
fatal_exceptions: Tuple[Type[BaseException], ...] = (KeyboardInterrupt, SystemExit)

# Persistence policy
save_on_exception: bool = True
exception_save_path_override: Optional[str] = None
```

## FileReport

`FileReport` is the core object for tracking the processing lifecycle of a **single file**.  
It automatically records:

- lifecycle status (`RUNNING → SUCCEEDED/FAILED/SKIPPED`)
- warnings, errors, notes
- HITL review flags
- **computed metadata**:
  - `name` (from `path.name`)
  - `mime_type` (extension-based)
  - `size_bytes` (best-effort filesystem probe)

All you need to create one is a filesystem path — no additional ceremony.

---

### Getting started with files

```python
from pipeline_watcher import FileReport

fp = FileReport("/path/to/some/file")

fp.note("Here is a note about this file")
assert fp.running          # passes

fp.warn("Here is a warning about this file")
assert fp.running          # warnings do not change status

fp.fail("File processing failed due to ...")
assert fp.failed           # fail() sets FAILED + timestamps

fp.request_review("Needs manual validation")
assert fp.requires_human_review
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
fp.add_warning("Low resolution detected")

if not fp.ok:
    fp.fail("One or more checks failed")

fp.end()       # infers success/failure if not already terminal
fp_dict = fp.model_dump()  # ready for JSON serialization
```

---

#### Use with PipelineReport

`FileReport` objects are often created automatically by the higher-level `pipeline_file(...)` context manager:

```python
with pipeline_file(report, "/inputs/a.pdf") as fr:
    fr.note("Running OCR…")
    # raise ValueError("...") → automatically recorded, autosaved
```

This is the recommended way to use `FileReport` in real pipelines, since it captures:

- stdout / stderr (optional)
- warnings
- exceptions + traceback
- duration
- banner updates
- autosave-on-exception behavior (based on settings)

---

#### FileReport Summary

`FileReport` is your “per-file audit log,” designed to:

- require almost no input
- behave predictably
- serialize cleanly
- automatically collect actionable metadata

Use `FileReport.begin(path)` when manually controlling file-level logic, or let `pipeline_file()` manage it for you with full exception and setting handling.

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

Represents processing for a single file (ordered list of `StepReport`s).

- Metadata: `file_id`, `path`, `name`, `size_bytes`, `mime_type`.
- Progress rolls up from steps.
- Fluent API: `append_step(step)` returns `self`.

#### Convenience Methods

To log progress with minimal ceremony:

- `add_completed_step(label, note=None, metadata=None)` – add a SUCCESS step.
- `add_failed_step(label, reason=None, metadata=None)` – add a FAILED step.
- `add_skipped_step(label, reason=None, metadata=None)` – add a SKIPPED step.
- `add_review_step(label, reason=None, metadata=None, mark_success=True)` – SUCCESS + HITL request.

```python
fr = FileReport.begin(file_id="42", path="inputs/doc1.docx")
fr.add_completed_step("Verified file exists")\
  .add_review_step("Low OCR confidence", reason="score=0.42")\
  .add_failed_step("Render PDF", reason="timeout")
```

---

## Comments → Structured Notes (debuggable “comment replacement”)

Use `StepReport.notes` as **comments that ship to the UI**. This turns what you’d
normally write as `# comments` into a reviewable narrative.

```python
st = StepReport.begin("calc_result", label="Calculate result")
result = some_calculation()

if result > 100:
    st.notes.append("result > 100 → taking branch A")
else:
    st.notes.append("result ≤ 100 → taking branch B")

st.end()  # infers SUCCESS (no failed checks or errors)
```

Tip: add ergonomic helpers to avoid touching lists directly:

```python
st.note("raw_result=%s" % result)   # your helper that appends to notes
st.warn("slow path used")           # appends to warnings
st.error("contract violated")       # appends to errors
```

This pattern makes runtime behavior **discoverable** in the UI without attaching a
debugger or tailing logs.

---

## Context Managers for Exception Handling & Debugging

Context managers simplify the *try/except/finally* ceremony and guarantee that steps
and files are finalized, even on early returns or errors. They also record `duration_ms`
for quick SLO/troubleshooting.

### `pipeline_step` (batch-level step)

```python
from pipeline_watcher import pipeline_step

with pipeline_step(report, "validate", label="Validate batch") as st:
    st.add_check("manifest_present", ok=True)
    st.add_check("ids_unique", ok=False, detail="3 duplicates")  # will cause FAILED
# The step is appended, finalized, and timed automatically.
```

### `pipeline_file` (per-file block)

```python
from pipeline_watcher import pipeline_file

with pipeline_file(
    report,
    file_id="f1",
    path="inputs/a.docx",
    name="a.docx",
    raise_on_exception=False,   # record failure and continue (default)
    save_on_exception=True      # save report immediately on errors
) as fr:
    fr.add_completed_step("Verified file exists")
    risky_work()  # if this raises, fr is recorded as FAILED and report is saved
```

### `file_step` (per-step inside a file)

```python
from pipeline_watcher import file_step

with pipeline_file(report, file_id="f2", path="inputs/b.csv", name="b.csv") as fr:
    with file_step(fr, "calc_result", label="Calculate result") as st:
        r = some_calculation()
        st.notes.append(f"raw result={r}")
        if r > 100:
            st.notes.append("branch A")
            # ... do A ...
        else:
            st.notes.append("branch B")
            # ... do B ...
```

All context managers support exception handling: they **record** the failure (status, error,
traceback), **finalize** the object, and by default **do not re-raise**—so your pipeline
can continue to the next file/step. You can opt into fail-fast with `raise_on_exception=True`.
`pipeline_step`/`pipeline_file` also support **saving on exception** via `save_on_exception`
and `output_path`/`output_path_override`.

### Binding (less boilerplate)

Bind a report once so helpers don’t need the `report` parameter:

```python
from pipeline_watcher import bind_pipeline

with bind_pipeline(report):
    with pipeline_file(None, file_id="f2", path="inputs/b.docx") as fr:
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
├─ batch_summary.html.j2     # banner, percent, message, updated_at
├─ file_table.html.j2        # file rows with status/HITL badges
└─ file_detail.html.j2       # steps, notes/checks/errors per file
```

Example snippet:

```jinja2
<h1>Batch {{ report.batch_id }} — {{ report.stage }}</h1>
<p>Status: {{ report.percent }}% — {{ report.message }}</p>

<ul>
{% for f in report.files %}
  <li>
    File {{ f.name or f.file_id }}: {{ f.status }}
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

- JSON-friendly: `model_dump_json()` on any report.
- Helper: `dump_report(path, report)` or `PipelineReport.save(output_path)` (direct write).

```python
from pipeline_watcher import dump_report

dump_report("reports/progress.json", report)  # atomic helper
# or, if you prefer a direct write on the object:
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
