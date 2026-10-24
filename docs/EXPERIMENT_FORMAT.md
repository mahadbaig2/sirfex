# Experiment Format

Experiments are stored under `artifacts/experiments/<experiment-id>/`:

```text
manifest.json       # versioned configuration, provenance, status, and artifact references
scenario.json       # exact scenario input
result.json         # result summary and failure records
trajectory.jsonl    # one JSON object per simulation step
```

The manifest schema is version `1`. It records the environment/controller identifiers and versions, controller metadata/configuration, scenario, evaluation settings, relevant dependency versions, outcome, failure records, Git commit when available, and UTC creation time.

Writes use a temporary file plus atomic replacement. A `.incomplete` marker makes interrupted writes detectable. Existing experiment IDs are never overwritten. IDs and artifact paths are validated to prevent traversal. Artifacts are JSON/JSONL only; Python pickle is not used.

Inspect saved evidence with:

```powershell
sirfex experiments list
sirfex experiments show <experiment-id>
```
