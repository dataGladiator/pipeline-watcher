import json
from pathlib import Path

from pipeline_watcher import (
    StepReport, FileReport, PipelineReport,
    StepStatus, dump_report
)


def _make_file(file_id: str, name: str, *, hitl=False, fail=False) -> FileReport:
    fr = FileReport.begin(file_id=file_id, path=f"inputs/{name}", name=name)

    # Step 1: parse (always success)
    st_parse = StepReport.begin("parse", label="Parse").succeed()
    fr.append_step(st_parse)

    # Step 2: analyze (may request human review)
    st_an = StepReport.begin("analyze", label="Analyze")
    if hitl and hasattr(st_an, "request_review"):
        st_an.request_review("Low model confidence")
    st_an.end()
    fr.append_step(st_an)

    # Step 3: render (may fail)
    st_render = StepReport.begin("render", label="Render")
    if fail:
        st_render.fail("Renderer crashed")
    st_render.end()
    fr.append_step(st_render)

    # finalize the file roll-up
    fr.end()
    return fr


def test_demo_batch_processing(tmp_path: Path):
    # Simulate a batch with 3 files
    files = [
        _make_file("f1", "file1.txt", hitl=False, fail=False),
        _make_file("f2", "file2.txt", hitl=True,  fail=False),
        _make_file("f3", "file3.txt", hitl=False, fail=True),
    ]

    report = PipelineReport(batch_id=101, kind="process")
    report.set_progress("discover", 10, "Scanning files…")

    # Add a batch-level step for discovery
    report.append_step(StepReport.begin("discover", label="Discover").succeed())

    # Append files to the batch
    for fr in files:
        report.append_file(fr)

    # Optionally compute overall progress from batch steps
    report.recompute_overall_from_steps()

    # Write JSON (atomic) and read back for sanity
    out = tmp_path / "progress.json"
    dump_report(out, report)
    dump_report(Path('./tests/test_output.json'), report)

    data = json.loads(out.read_text())
    # Basic shape checks
    assert data["batch_id"] == 101
    assert data["kind"] == "process"
    assert isinstance(data["steps"], list)
    assert isinstance(data["files"], list)
    assert len(data["files"]) == 3

    # Status rollups:
    # - file1 should be SUCCESS
    # - file2 may still be SUCCESS but with review requested
    # - file3 should be FAILED
    status_by_id = {f["file_id"]: f["status"] for f in data["files"]}
    assert status_by_id["f1"] == "SUCCESS"
    assert status_by_id["f3"] == "FAILED"

    # Steps are auto-finalized on append; last step should be terminal
    for f in data["files"]:
        assert f["steps"], "Each file should have steps"
        assert f["steps"][-1]["status"] in ("SUCCESS", "FAILED", "SKIPPED")