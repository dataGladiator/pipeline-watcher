# pipeline-watcher

Tiny, framework-free models for **batch / per-file / per-step reporting** that serialize cleanly to JSON.
Perfect for pipelines that want a single `progress.json` the UI can poll.

- **Simple models:** `PipelineReport`, `FileReport`, `StepReport`
- **Ordered steps:** append-only lists (no dict keying)
- **Uniform end():** finalize steps/files without branching
- **No heavy deps:** just Pydantic v2
- **Atomic writes:** helper to safely write JSON for polling UIs

## Install

```bash
pip install pipeline-watcher
```

(Or in your project: add `pipeline-watcher` to your requirements and `pip install -e .` for local dev.)

## Quick start

```python
from pathlib import Path
from pipeline_watcher import StepReport, FileReport, PipelineReport, dump_report

report = PipelineReport(batch_id=123, kind="process")
report.set_progress("discover", 10, "Scanning files…")

for doc in docs:  # your iterable of documents
    fr = FileReport.begin(file_id=str(doc.id), path=doc.relpath, name=doc.name)

    fr.append_step(StepReport.begin("parse",   label="Parse"))
    fr.append_step(StepReport.begin("analyze", label="Analyze").succeed())

    report.append_file(fr)

report.recompute_overall_from_steps()
dump_report(Path("reports/progress.json"), report)
```

### Why it’s useful
- Append a step with `append_step()` — it **auto-finalizes** via `step.end()`.
- Need explicit control? You can still call `succeed()`, `fail()`, `skip()` on a step or file.
- Keep a single JSON file for your UI; no Django or pandas required.

## Core concepts

- **StepReport**: a unit of work (`id`, `label`, `status`, `percent`, `checks`, `metadata`).
- **FileReport**: per-document aggregation (`file_id`, ordered `steps`, roll-up `percent/status`).
- **PipelineReport**: batch container with top-level banner (`stage`, `percent`, `message`), plus
  ordered batch `steps` and a list of `files`.

All three use **ISO timestamps** and are Pydantic v2 models. `SCHEMA_VERSION` is currently `"v2"`.

## JSON shape (example, truncated)

```json
{
  "batch_id": 123,
  "kind": "process",
  "stage": "analyze",
  "percent": 65,
  "message": "13/20 analyzed",
  "updated_at": "2025-10-22T18:05:13.421000+00:00",
  "report_version": "v2",
  "steps": [
    {
      "id": "discover",
      "label": "Discover files",
      "status": "SUCCESS",
      "percent": 100,
      "started_at": "2025-10-22T18:01:00.102000+00:00",
      "finished_at": "2025-10-22T18:01:02.993000+00:00",
      "checks": [],
      "notes": [],
      "errors": [],
      "warnings": [],
      "metadata": {},
      "report_version": "v2"
    }
  ],
  "files": [
    {
      "file_id": "42",
      "path": "inputs/42.txt",
      "name": "42.txt",
      "status": "SUCCESS",
      "percent": 100,
      "steps": [
        {"id":"parse","status":"SUCCESS","percent":100,"report_version":"v2"},
        {"id":"analyze","status":"SUCCESS","percent":100,"report_version":"v2"}
      ],
      "report_version": "v2"
    }
  ]
}
```

## API (most used)

```python
from pipeline_watcher import (
  SCHEMA_VERSION, StepStatus, Check,
  StepReport, FileReport, PipelineReport,
  atomic_write_json, dump_report, now_utc
)
```

### StepReport
- `StepReport.begin(id: str, *, label: str | None = None, **meta) -> StepReport`
- `step.add_check(name, ok, detail=None) -> None`
- `step.end() -> StepReport`  # finalize (idempotent)
- `step.succeed()/fail(msg=None)/skip(reason=None) -> StepReport`
- Properties: `status`, `percent`, `checks`, `errors`, `metadata`, `ok`

### FileReport
- `FileReport.begin(file_id: str, **meta) -> FileReport`
- `file.append_step(step: StepReport) -> StepReport`  *(auto calls `step.end()`)*  
- `file.end()/succeed()/fail()/skip() -> FileReport`
- `file.percent` auto-averages step percents (simple default)

### PipelineReport
- `PipelineReport(batch_id: int, kind: Literal["validation","process"])`
- `report.set_progress(stage, percent, message="") -> None`
- `report.append_step(step) -> StepReport`  *(auto calls `step.end()`)*  
- `report.append_file(file) -> FileReport`  *(auto calls `file.end()`)*  
- `report.recompute_overall_from_steps() -> None` (mean of step percents)

### I/O helpers
- `dump_report(path: Path, report) -> None` (atomic JSON write)
- `atomic_write_json(path: Path, data: dict) -> None`

## Versioning

`SCHEMA_VERSION = "v2"` indicates **steps are lists** and append auto-finalizes via `end()`.

If you later change the JSON schema, bump `SCHEMA_VERSION` and gate your UI parser if needed.

## FAQ

**Q: Do I need to call `succeed()` or `fail()`?**  
A: Not usually. Call `append_step(step)` or `append_file(file)` and they’ll finalize via `end()`.

**Q: How is percent computed?**  
A: For `FileReport`, it’s the simple mean of its step percents. For the batch, call
`recompute_overall_from_steps()` (simple mean). You can compute your own and call `set_progress()`.

**Q: Timezones?**  
A: All stamps use `now_utc()` — UTC with offset.

## License

MIT
