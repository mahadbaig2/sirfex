# Robustness study execution

The Phase 3.7 machine protocol is a **candidate**. It has `executable: false`; the execution and resume commands reject it. No scientific study has been run.

## Read-only checks

From the repository root:

```powershell
sirfex robustness study --mode dry-run --protocol configs/robustness/phase37_checkpoint2_protocol.candidate.json --manifest docs/evidence/phase_3_7_development_scenario_manifest.json
```

This verifies the development-only manifest, controller checkpoint hashes, exact grid, pair design, search-space bytes, episode budget and immutable work plan without rolling out controllers. `--mode preflight` performs the same local validation. The returned work-plan digest identifies ordered work items, seeds and resource limits.

## Authorization boundary

Only a separately reviewed frozen protocol with status `FROZEN — AUTHORIZED FOR EXECUTION` and `executable: true` can enter the runner. An operator must supply the exact protocol semantic SHA-256 as `--approval-digest` and its committed Git revision as `--frozen-commit`. The protocol file's bytes must match that commit, which must be reachable from HEAD. The manifest and all checkpoint hashes are rechecked before work starts. This repository does not issue a frozen protocol or approval token.

After a future authorization, the interface is:

```powershell
sirfex robustness study --mode scientific-execute --protocol <frozen-protocol.json> --manifest <development-manifest.json> --approval-digest <approved-sha256> --frozen-commit <40-character-commit> --output-dir <local-run-root>
sirfex robustness study --mode resume --protocol <frozen-protocol.json> --manifest <development-manifest.json> --approval-digest <approved-sha256> --frozen-commit <40-character-commit> --output-dir <same-local-run-root>
sirfex robustness study --mode report --protocol <frozen-protocol.json> --manifest <development-manifest.json> --output-dir <same-local-run-root>
```

The report command is read-only with respect to simulator execution. It verifies stored cases and headline archives and writes deterministic JSON and Markdown. It may produce `partial_incomplete` for unfinished runs.

## Work boundaries and stops

The plan orders fixed development stress episodes, one shared physical disturbance grid per controller/anchor/direction, deterministic differential proposals, one archive selection per pair, then reporting. Rejected proposals consume their fixed budget. Grid cells use controller-independent prepared physical scenarios; unavailable points remain censored. The runner keeps at most two controller sessions open.

Each work item is hashed and written atomically. Completed and censored items are immutable. `resume` reuses them, reruns failed or interrupted items, and refuses changed plans or corrupted stored evidence. Proposal sequences and prepared grids are recomputed and compared before reuse. Selected headline episodes are archived and independently replayed. An unverified Class B selection prevents a complete report.

Wall time is monotonic and cumulative across invocations. PPO controller evaluation time is charged per controller. Storage includes the run directory and is checked before atomic writes and archive selection. The approved limits are part of the plan; resuming does not reset them. A stop returns a distinct budget-exhausted status and leaves a partial operational report. Extending a limit changes plan identity and requires a separate protocol review.

The bounded synthetic fixture can be run with `python -m scripts.pilot_phase37_study`. It uses only classical test adapters and writes [pilot measurements](evidence/phase_3_7_study_pilot.json); it does not run the real 17-controller cohort.
