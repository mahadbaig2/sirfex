# SIRFEX Phase 1 Report

## Scope

Phase 1 adds reproducible experiment infrastructure, explicit failure semantics, deterministic replay, local artifact storage, comparison, and Apache-2.0 licensing. Automated failure discovery, Optuna search, retraining, API, and Studio remain out of scope.

## Architecture changes

- `Scenario` now carries schema version, environment version, description, tags, and provenance.
- Reusable parameter validation rejects unknown names, booleans in numeric fields, non-finite values, invalid ranges, and incompatible pump boundaries.
- `FailureRecord`, `FailureCategory`, `EvidenceStatus`, and execution status separate evidence concepts.
- `EvaluationRunner` protects reset, controller initialization, validation, stepping, and result creation within one lifecycle boundary while preserving caller-owned environment reuse.
- `ExperimentStore` atomically persists JSON manifests, scenario files, result summaries, and JSONL trajectories.
- Replay reconstructs supported controllers and compares observations, actions, outcomes, failures, rewards, and lengths.
- Comparison reports describe differences and refuse to present a universal robustness score.
- CI installs the simulation extra so LunarLander integration tests execute rather than silently skipping.

## Verification

Executed locally on Windows 11 x64 with Python 3.12.10:

- Ruff format/check: passed.
- Pytest: 16 passed, 0 failed.
- Pump experiment save/reload/replay: `exact_match`.
- LunarLander experiment save/reload/replay: `exact_match`.
- Pump and LunarLander CLI commands: passed.
- Apache metadata editable installation: passed.

The suite emits three Box2D SWIG deprecation warnings from the third-party dependency; they do not fail tests.

## Licensing

The repository now uses Apache License 2.0 with attribution to Mahad Mirza Baig. Third-party dependencies retain their own licenses; no dependency license is replaced by the project license.

## Known limitations

- Exception execution manifests are not yet persisted as first-class failed experiments; exceptions remain visible to the caller.
- The random controller relies on the environment action-space seed for replay determinism.
- Stock LunarLander does not expose a documented terminal landing outcome through this adapter, so terminal episodes are classified as unknown unless explicit evidence appears in `info`.
- Storage is file-based and single-process; concurrent writers are not coordinated.
- No database, web UI, failure search, retraining, or controller registry is included.

## Recommended Phase 2 scope

Build valid scenario generation and bounded failure discovery on top of the now-persistent evidence model. Preserve the explicit distinction between observed outcomes and inferred failure categories.
