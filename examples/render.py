#!/usr/bin/env python3
"""
Render a pipeline-watcher JSON report to HTML using Jinja2 templates.

Usage:
    python render.py --input reports/progress.json --out out.html --templates ./templates --title "My Run"
"""
import argparse
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

def load_report_model(json_path: Path):
    """Try to reconstruct a PipelineReport model if pipeline_watcher is installed.
    Fallback to a plain dict if import/validation fails."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    try:
        from pipeline_watcher import PipelineReport  # type: ignore
        return PipelineReport.model_validate(data)  # pydantic v2
    except Exception:
        return data  # Jinja templates can handle dict form

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", type=Path, required=True, help="Path to pipeline JSON (progress.json)")
    ap.add_argument("--out", "-o", type=Path, required=True, help="Path to output HTML")
    ap.add_argument("--templates", "-t", type=Path, default=Path("templates"), help="Templates directory")
    ap.add_argument("--title", type=str, default="Pipeline Report")
    args = ap.parse_args()

    report = load_report_model(args.input)

    env = Environment(
        loader=FileSystemLoader(args.templates),
        autoescape=select_autoescape(["html", "xml"]),
    )
    tpl = env.get_template("index.html.j2")
    html = tpl.render(title=args.title, header=args.title, report=report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
