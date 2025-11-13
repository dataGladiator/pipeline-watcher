from pathlib import Path
from pipeline_watcher import PipelineReport, pipeline_file
from pipeline_watcher.settings import set_global_settings


set_global_settings(raise_on_exception=False)


report = PipelineReport(label="test-report",
                        kind="process",
                        output_path=Path("./test.json"))
data_dir = Path("../")
files = [f for f in data_dir.glob('*') if f.is_file()]
for j, filepath in enumerate(files):
    with pipeline_file(report, filepath) as file_report:
        raise ValueError("what?")


