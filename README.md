# pipeline-watcher

`pipeline-watcher` is a lightweight alternative to traditional logging for **AI/ML
pipelines**. Instead of unstructured logs, it emits **structured JSON reports**
that your UI (Jinja2, React, etc.) can read directly. Think of it as
> logs that the UI can render.

It models **batches**, **files**, and **steps**, with JSON that’s easy to persist
and inspect.

## Demo (Quick Glance)

```python
from pipeline_watcher import PipelineReport, FileReport, StepReport, dump_report

# Create a batch report
report = PipelineReport(batch_id=42, kind="process")
report.set_progress("load", 5, "Loading input files…")

# Add a batch-level step
report.append_step(StepReport.begin("load", label="Load inputs").succeed())

# Add a file report with quick steps (fluent, log-like)
fr = FileReport.begin(file_id="f1", path="inputs/data.csv", name="data.csv")
fr.add_completed_step("Verified file exists")\
  .add_review_step("Check class balance", reason="Minor skew detected")\
  .add_failed_step("Feature extraction", reason="NaN values found")
report.append_file(fr)

# Persist for the UI to pick up
dump_report("reports/progress.json", report)
```

Yields `reports/progress.json` with a batch banner and per-file timelines.

---

## Key Concepts

### Abstract Base (optional pattern)

`StepReport` and `FileReport` share a common shape (status, timestamps, percent,
notes, errors, metadata). If you want to enforce this across custom report types,
you can introduce an abstract base (`ReportBase(abc.ABC)`) that declares:

- `ok: bool` – whether the unit ultimately succeeded
- `end()` – auto-finalize based on `ok`

### StepReport

Represents a single unit of work (e.g., `"parse"`, `"validate_index"`).

- Fields: `status`, `percent`, `started_at/finished_at`, `notes`, `warnings`, `errors`, `checks`, `metadata`.
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

Both context managers support:
- `set_stage_on_enter` / `banner_*` to update the top banner while running.
- `raise_on_exception` (default False) to keep going after recording failures.
- `save_on_exception` + `output_path_override` (when using `PipelineReport.save`).

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

## Using with Jinja2

You can pass the Pydantic models (or their dicts) straight to templates.

### Jinja2 Template (snippet)

```jinja2
<h1>Batch {{ report.batch_id }} — {{ report.stage }}</h1>
<p>Status: {{ report.percent }}% — {{ report.message }}</p>

<ul>
{% for f in report.files %}
  <li>
    File {{ f.name or f.file_id }}: {{ f.status }}
    <ul>
      {% for s in f.steps %}
        <li>{{ s.label }} — {{ s.status }}
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

### Rendered (Markdown approximation)

```
Batch 42 — load
Status: 5% — Loading input files…

• File data.csv: SUCCESS
  - Verified file exists — SUCCESS
  - Check class balance — SUCCESS (🔎 Requires review: Minor skew detected)
  - Feature extraction — FAILED [NaN values found]
```

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

`pipeline-watcher` stays intentionally small: append-only, JSON-first, and UI-ready.

---

## License

MIT (example; change to your actual license).
