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

Phase 3.3 matched runs additionally persist frozen input and partition records,
`continuation_runs.json` with actual timestep and per-scenario episode exposure,
six arm checkpoints and metadata, final `evaluation.json`, `regression.json`,
`summary.json`, and representative replay references. The `manifest.json` status
tracks preparation, training, completion or failure; a partial study cannot
claim a completed evaluation. The compact public export is in
`docs/evidence/phase_3_3_results.json`, while binary checkpoints and full
trajectory artifacts remain in the ignored local store. See
[matched study protocol](MATCHED_ROBUSTNESS_STUDY.md).

Phase 3.4 uses the same local `artifacts/training/<study-id>/` store. It adds
`stress_benchmark.json` with physical strata, exact feasibility records and
partition hashes; `difficulty_strata.json`; `candidate_curricula.json` with
search/witness provenance; `development_results.json` with per-scenario
screening results and actual exposures; and `selected_curriculum.json` with
individual eligibility checks. A selected study additionally writes a frozen
`final_protocol.json`; only its confirmatory branch can create
`continuation_runs.json`, `final_evaluation.json`, `paired_analysis.json` and
`replay_verification.json`. The observed `no_eligible_curriculum` study has
five hash-verified screening checkpoints and no final result artifacts. Compact
public exports are in `docs/evidence/phase_3_4_preregistered.json` and
`docs/evidence/phase_3_4_results.json`; the separate
`docs/evidence/phase_3_4_search.json` preserves candidate and replay metadata.
Trusted binary checkpoints and full trajectory actions remain local.

Phase 3.5 retains the version-2 manifest and adds a versioned
`pump-curriculum-exposure:1` block to each run in `screening_runs.json` and,
only after selection, `confirmation_runs.json`. `exposure_audit.json`,
`calibration.json`, `protocol.json` and the reused hash-checked
`stress_benchmark.json` are written at preparation. `selected_curriculum.json`
records per-seed adequacy and nominal/stress checks. `final_protocol.json`,
`final_evaluation.json`, `paired_analysis.json` and
`replay_verification.json` exist only if the corresponding gated stages ran.
The compact public preregistration and calibration are in `docs/evidence/`;
local binary checkpoints remain outside Git.
# Phase 3.6 research artifacts

`artifacts/research/<run-id>/` contains a hashed `protocol.json`, `manifest.json`, evidence context, proposal, validation, plan, explicit approval, model usage, training record, development evaluation, deterministic decision, critique, and research memory when the corresponding stage completes. Candidate PPO binaries stay in this ignored local directory and are verified by SHA-256. Diagnostics use separate `policy-regression-diagnostics` runs with fixed-observation actions and independent tracking outcomes. Public compact exports omit local checkpoint paths; see [Phase 3.6 report](PHASE_3_6_REPORT.md).
