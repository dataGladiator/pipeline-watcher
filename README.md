# pipeline-watcher

`pipeline-watcher` is a lightweight alternative to traditional logging for **AI/ML
pipelines**—with built‑in support for **HITL (Human‑In‑The‑Loop) review**.  
Instead of unstructured logs, it emits **structured JSON reports** (batches → files → steps)
that your UI (Jinja2, React, etc.) can render directly. Think of it as:

> logs that the UI can read

It models **batches**, **files**, and **steps**, plus optional **review flags** so you can
surface “needs human review” right in your dashboards.

---

## Demo (Quick Glance)

This shows:
- iterating over a directory of PDFs,
- **comment replacement via notes** (your inline rationale becomes UI-visible),
- **context managers** that **handle exceptions** and **auto-finalize** records,
- **HITL review** when OCR quality is low,
- and the **`file_step`** helper for minimal ceremony inside a file block.

```python
from pathlib import Path
from pipeline_watcher import (
    PipelineReport,
    pipeline_file,   # per-file context manager
    file_step,       # per-step (within a file) context manager
)

# Create a batch report
report = PipelineReport(batch_id=42, kind="process", output_path="reports/progress.json")
report.set_progress("discover", 5, "Scanning PDF directory…")

data_dir = "inputs/pdfs"

for file_path in Path(data_dir).glob("*.pdf"):
    # Assume you have small helpers/wrappers with:
    #   - meta() -> dict(file_id=..., path=..., name=...)
    #   - extract_text() -> object with .text and .quality
    #   - index_text(text) -> dict with indexing info
    pdf_wrapper = get_pdf_wrapper(file_path)  # user-provided

    # Context manager notes:
    # - auto-finalizes the FileReport
    # - on exception: records FAILED + traceback, and (by default) does NOT re-raise
    #   so the loop continues to the next file
    # - set save_on_exception=True to save the batch JSON immediately on error
    with pipeline_file(
        report,
        **pdf_wrapper.meta(),
        raise_on_exception=False,   # record error and continue (default)
        save_on_exception=True,     # save report immediately on failure
    ) as file_report:

        # Step 1: OCR / text extraction
        with file_step(file_report, "extract_text", label="Extract text (OCR)") as st:
            extracted = pdf_wrapper.extract_text()
            st.notes.append("Performed OCR on the PDF")
            st.metadata["ocr_quality"] = extracted.quality

            # Specific threshold decision with HITL:
            if extracted.quality < 0.90:
                st.notes.append("OCR quality below threshold (0.90) → request review")
                file_report.add_review_step(
                    "Review OCR quality",
                    reason=f"quality={extracted.quality:.2f} < 0.90",
                    metadata={"quality": extracted.quality},
                    mark_success=True,  # step 'succeeds' but asks for human review
                )
            else:
                st.notes.append("OCR quality meets threshold")

        # Step 2: Indexing the extracted text (may raise; context manager records)
        with file_step(file_report, "index_text", label="Index extracted text") as st:
            info = pdf_wrapper.index_text(extracted.text)
            st.notes.append("Indexed text for search")
            st.metadata.update(info)

        # Optional quick “log line” for bookkeeping
        file_report.add_completed_step("Verified file exists")

# Persist the whole batch report (direct write to output_path)
report.save()
```

Yields `reports/progress.json` with a batch banner and per-file timelines.

---

## Key Concepts

### Abstract Base (optional pattern)

`StepReport` and `FileReport` share a common shape (status, timestamps, percent,
notes, errors, metadata, optional review flag). If you want to enforce this across
custom report types, you can introduce an abstract base (`ReportBase(abc.ABC)`) that declares:

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

## License

MIT (example; change to your actual license).
