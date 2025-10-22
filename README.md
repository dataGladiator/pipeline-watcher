# pipeline-watcher

A tiny Python package for monitoring and reporting pipeline progress, designed for
AI/ML and document-processing workflows. It gives you lightweight, JSON-serializable
reports at the **batch**, **file**, and **step** level, suitable for UI dashboards
or simple log inspection.

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

FileReport also includes helpers for common patterns:

- `add_completed_step(label, note=None, metadata=None)`  
  Quickly add a SUCCESS step.

- `add_failed_step(label, reason=None, metadata=None)`  
  Quickly add a FAILED step.

- `add_skipped_step(label, reason=None, metadata=None)`  
  Quickly add a SKIPPED step.

- `add_review_step(label, reason=None, metadata=None, mark_success=True)`  
  Add a step that flags HITL (human-in-the-loop) review. By default it marks the
  step as SUCCESS but requests review.

All these helpers return `self` so you can chain them like a mini log:

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

## Example

```python
from pipeline_watcher import PipelineReport, FileReport, StepReport, dump_report

# Create a batch report
report = PipelineReport(batch_id=101, kind="process")
report.set_progress("discover", 10, "Scanning files…")

# Add batch-level step
report.append_step(StepReport.begin("discover", label="Discover").succeed())

# Add file reports
fr = FileReport.begin(file_id="f1", path="inputs/a.docx", name="a.docx")
fr.add_completed_step("Verified file exists")
report.append_file(fr)

# Persist
dump_report("reports/progress.json", report)
```
