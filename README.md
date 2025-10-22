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

