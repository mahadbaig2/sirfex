# SIRFEX Phase 3.7 Checkpoint 2B execution report

**Operational status at publication:** scientific execution in progress. This is an execution checkpoint, not a scientific results report. Completion, coverage, resource use, evidence integrity and conclusions will be updated from the durable run artifacts after the process exits.

## Approved identity and authorization

The operator's Stage 2 instruction in the project conversation explicitly approved the exact frozen bytes, work plan, controller and scenario inventories, comparisons, and limits, while accepting that a resource stop may leave incomplete results. The approved review commit was `ba8f0bda636b7475fc9b12fed1787e869f7d271a`. The authenticated project GitHub account is `mahadbaig2`; the separate local authorization record uses that identifier and records its own creation time in UTC, without asserting an unavailable timestamp for the original conversation approval. The record contains the approval source and frozen commit. It was validated with the existing `validate_authorization_record` function; its semantic digest is `e8443a2d4811b561a2ec3b71054819cb4c3a8482ca3da8f1c58f2f6b4cc7631d`. The record remains local under ignored `artifacts/robustness/phase37_checkpoint2b/authorization.json` and is not published in Git.

| Identity | Verified value |
|---|---|
| Frozen protocol commit | `5f37c0baea944e564d2bbea2da4971d528331d33` |
| Committed protocol bytes | 14,909 UTF-8 bytes; `git show` equals working file |
| Raw protocol SHA-256 | `907742ed639ac16f12ca4b4f1a0becd422285f89fafde0719723e0ca6fe4b6c6` |
| Semantic protocol SHA-256 | `27a2b4ca8bb1552c22c8d0c3ad07931b14d0b676d28656a2af4796c7fb75bb04` |
| Work-plan SHA-256 | `ead9af1ef00824016e6dea102ef4b7a3f37b3eb6948cfff238b1dae6839947c2` |
| Development manifest SHA-256 | `1bc850397b4f608b0cf7c75e63660b3bdfa52923d043b0b876f8720d763f88b0` |
| Controller inventory SHA-256 | `0fdfe5cc1c23c5f6d5592e5fc8be7e53e2f8d6117633798c262344a08336c0ea` |
| Resource budget SHA-256 | `3552a209f9fd091723118476fca381dc154b84c539b937b0ceb39ff9ec5df71a` |
| Search-space raw SHA-256 | `d99869ff60bc7025c2eccb545816743d58da79c172e1833621a7f65d7cfe31f6` |

The frozen file differs from the candidate only in approved `status`, `executable` and recomputed `semantic_sha256` fields. Its dry-run confirmed 17 controllers, 12 anchors, seven pairs, 408 fixed-stress cases, 3,264 grid cells, 1,400 differential work items, seven archives, one report and an 88,168-episode upper bound. All 15 PPO checkpoint hashes, metadata, loadability and repeated deterministic actions were freshly checked. The development manifest contains only the 55 declared development cases; no sealed final partition was inspected.

## CI and local validation

The approved review commit passed both GitHub Actions jobs in [run 40991020538](https://github.com/mahadbaig2/sirfex/actions/runs/40991020538). The frozen protocol commit passed both `test` and `training` in [run 42471643899](https://github.com/mahadbaig2/sirfex/actions/runs/42471643899). Locally, 243 tests passed with four dependency warnings; Ruff lint, Ruff format and the frozen read-only dry-run passed. PR #2 remains a draft against `main`.

## Execution host and timeline

The local host is Windows 11, Intel Core i5-8350U (four cores, eight logical processors), with 17,015,459,840 bytes of physical RAM. Python is 3.12.10; installed versions include SIRFEX 0.1.0, NumPy 2.5.4, stable-baselines3 2.9.0, PyTorch 2.14.1 and Gymnasium 1.4.0. Before launch, the C: volume had approximately 16.58 GB free, exceeding the approved 2-GiB artifact cap with temporary headroom. Python-traced memory is not a measure of total native memory.

- `2026-11-08 10:31:55 UTC`: a separate hidden local launcher started the guarded `scientific-execute` command after authorization and identity checks. The launcher PID was 18744 at start; its live state must be read from the local status file rather than assumed from this historical PID.
- The launcher writes local stdout/stderr and its exit code to ignored files in `artifacts/robustness/phase37_checkpoint2b/`. The durable study directory is `artifacts/robustness/phase37_checkpoint2b/run/study_ead9af1ef00824016e6d/`.
- The study's `plan.json`, `invocation.json`, `budget.json`, prepared grids, proposals, individual work items and evidence archives are retained locally. Completed work items are immutable. A stop or restart must preserve the same inputs and cumulative budget.

## Prespecified resource and reporting rules

The approved caps are eight cumulative wall hours, 1,800 evaluation seconds per PPO and 2 GiB of generated artifacts; the work plan permits at most 88,168 episodes and 17,633,600 transitions. A cap stop is `partial_incomplete` and cannot be reported as a completed comparison. The scientific report must regenerate from verified stored evidence, retain all planned coverage and missing work, distinguish viable, infeasible, unknown, invalid and censored cases, and verify any selected Class B headline by independent replay. The four Phase 3.5 directed pairs are primary, the three Phase 3.3 pairs secondary, and six starting-PPO/classical pairs descriptive. The study uses three correlated training seeds and a simplified pump simulator; no independent-sample significance or real-world safety claim is justified.

## Pending final verification

After the process exits, record the actual completion status, planned versus completed work, episodes and transitions, wall/controller/storage usage, any cap or crash accounting, evidence and replay checks, regenerated JSON/Markdown hashes, deviations, and conclusions. Publish a separate scientific results report only if every mandatory work item and report integrity gate passes. If the run is incomplete, publish an explicitly labeled partial-run report. Preserve all local evidence and do not merge PR #2 automatically.
