# pipeline-watcher

`pipeline-watcher` is a lightweight alternative to traditional logging for **AI/ML
pipelines**. Instead of unstructured logs, it gives you **structured JSON reports**
that can be dropped directly into a UI (for example via Jinja2 templates or a
dashboard frontend).

Think of it as *logs that the UI can read*. Each batch, file, and step produces
JSON with consistent fields, ready to be rendered in your web app.

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

# Persist to disk for the UI to pick up
dump_report("reports/progress.json", report)
```

Yields `reports/progress.json`:

```json
{
  "batch_id": 42,
  "kind": "process",
  "stage": "load",
  "percent": 5,
  "message": "Loading input files…",
  "steps": [...],
  "files": [...]
}
```

---

## Key Concepts

### Abstract Base

Both `StepReport` and `FileReport` share a common design philosophy: they represent
units of work with a `status`, timestamps, percent complete, notes, errors, and
arbitrary metadata.

If you want to enforce this consistently, you can define an abstract base (e.g.,
`ReportBase(abc.ABC)`) that declares the minimal interface:

- `ok: bool` property (indicates success/failure/needs review)
- `end()` method (auto-finalize the object)

This way, `StepReport`, `FileReport`, or any future specialized report classes all
share the same contract.

### StepReport

Represents a single processing step (e.g. "validate_index").

- Tracks `status`, `checks`, `notes`, `errors`, `warnings`, `metadata`.
- Helper methods: `start()`, `succeed()`, `fail()`, `skip()`, `end()`.
- Convenience property: `ok` for truthy success.

### FileReport

Represents the processing of a single file.

- Has metadata: `file_id`, `path`, `name`, `size_bytes`, `mime_type`.
- Tracks ordered list of `StepReport`s.
- Rolls up progress and status from its steps.
- Chainable methods: `append_step(step)`.

#### Convenience Methods

FileReport includes helpers for common patterns (all return `self` for chaining):

- `add_completed_step(label, note=None, metadata=None)` – quickly add a SUCCESS step.  
- `add_failed_step(label, reason=None, metadata=None)` – quickly add a FAILED step.  
- `add_skipped_step(label, reason=None, metadata=None)` – quickly add a SKIPPED step.  
- `add_review_step(label, reason=None, metadata=None, mark_success=True)` – flag HITL (human-in-the-loop) review; by default the step is SUCCESS but requests review.

```python
fr = FileReport.begin(file_id="42", path="inputs/doc1.docx")
fr.add_completed_step("Verified file exists")\
  .add_review_step("Low OCR confidence", reason="score=0.42")\
  .add_failed_step("Render PDF", reason="timeout")
```

### HITL / Review Flag

AI pipelines often require human-in-the-loop review. `add_review_step()` makes this explicit.

```python
fr = FileReport.begin(file_id="f2", path="inputs/b.docx", name="b.docx")

# Step completed successfully, but flagged for review
fr.add_review_step("Check OCR quality", reason="OCR confidence = 0.42")

# Later in templates or UI code
for step in fr.steps:
    if step.review.flagged:
        print(f"Step '{step.label}' requires human review: {step.review.reason}")
```

### PipelineReport

Represents a whole batch (validation or processing).

- Contains ordered batch-level `StepReport`s.
- Optionally contains `FileReport`s for per-file detail.
- Top-banner fields: `stage`, `percent`, `message`, `updated_at`.
- Append-only design: `append_step()` and `append_file()`.

---

## Using with Jinja2

Since reports are just Pydantic models that serialize to JSON/dicts, you can pass
them straight to a template.

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

Markdown won’t execute HTML/CSS, but here’s what the structure would look like
when rendered in a browser (minus whatever styling you add later):

```
Batch 42 — load
Status: 5% — Loading input files…

• File data.csv: SUCCESS
  - Verified file exists — SUCCESS
  - Check class balance — SUCCESS (🔎 Requires review: Minor skew detected)
  - Feature extraction — FAILED [NaN values found]

• File labels.csv: SUCCESS
  - Verified file exists — SUCCESS
  - Analyze label coverage — SUCCESS
  - Export artifacts — SUCCESS
```

---

## Persistence

- JSON-friendly: use `.model_dump_json(indent=2)`.
- For atomic writes, write to a temp file then replace the target.

```python
from pipeline_watcher import dump_report

dump_report("reports/progress.json", report)
```

Or send to stdout:

```python
print(report.model_dump_json(indent=2))
```
