# Training Artifact Format

Training artifacts are local and versioned under `artifacts/training/<training-run-id>/`:

```text
manifest.json
config.json
curriculum.json
evaluation.json
regression.json
summary.json
checkpoints/
  baseline.zip
  baseline.metadata.json
  normal_continuation.zip
  normal_continuation.metadata.json
  robust_continuation.zip
  robust_continuation.metadata.json
```

`manifest.json` records the run ID, schema, SIRFEX and Git versions, environment version, training seed and budget, source search, selected discoveries and experiments, checkpoint metadata, and execution status. Checkpoint metadata includes SHA-256 hashes and dependency versions. JSON artifacts are written atomically; incomplete runs remain marked and are skipped by listing. Generated training artifacts are ignored by Git.
