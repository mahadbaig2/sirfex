# SIRFEX Phase 3.7 Checkpoint 2B execution report

**Final execution status:** `checkpoint_2b_partial_budget_exhausted`. This is a verified partial operational report, not a completed scientific controller comparison. The first PPO controller reached its approved evaluation-time cap before the remaining cohort could run.

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

The approved review commit passed both GitHub Actions jobs in [run 40991020538](https://github.com/mahadbaig2/sirfex/actions/runs/40991020538). The frozen protocol commit passed both `test` and `training` in [run 42471643899](https://github.com/mahadbaig2/sirfex/actions/runs/42471643899). The initial execution-checkpoint documentation commit passed both jobs in [run 42472125032](https://github.com/mahadbaig2/sirfex/actions/runs/42472125032). Locally, 243 tests passed with four dependency warnings; Ruff lint, Ruff format and the frozen read-only dry-run passed. PR #2 remains a draft against `main`. The final partial-report publication commit requires its own CI verification.

## Execution host and timeline

The local host is Windows 11, Intel Core i5-8350U (four cores, eight logical processors), with 17,015,459,840 bytes of physical RAM. Python is 3.12.10; installed versions include SIRFEX 0.1.0, NumPy 2.5.4, stable-baselines3 2.9.0, PyTorch 2.14.1 and Gymnasium 1.4.0. Before launch, the C: volume had approximately 16.58 GB free, exceeding the approved 2-GiB artifact cap with temporary headroom. Python-traced memory is not a measure of total native memory.

- `2026-11-08 10:30:13 UTC`: the separate local authorization record was created and validated. This is the record creation time, not an asserted timestamp for the original conversation approval.
- `2026-11-08 10:31:55 UTC`: a separate hidden local launcher started the guarded `scientific-execute` command after authorization and identity checks. Its historical launcher PID was 18744.
- `2026-11-08 11:05:21 UTC`: the launcher exited with code 0 after the study runner returned `controller_budget_exhausted:ppo_3_10240`. Exit code 0 denotes a handled budget stop, not a complete study. Elapsed launcher time was approximately 33 minutes 26 seconds. The study invocation journal is closed, with no recovery events.
- The launcher writes local stdout/stderr and its exit code to ignored files in `artifacts/robustness/phase37_checkpoint2b/`. The durable study directory is `artifacts/robustness/phase37_checkpoint2b/run/study_ead9af1ef00824016e6d/`.
- The study's `plan.json`, `invocation.json`, `budget.json`, prepared grids, proposals, individual work items and evidence archives are retained locally. Completed work items are immutable. A stop or restart must preserve the same inputs and cumulative budget.

## Prespecified resource and reporting rules

The approved caps are eight cumulative wall hours, 1,800 evaluation seconds per PPO and 2 GiB of generated artifacts; the work plan permits at most 88,168 episodes and 17,633,600 transitions. A cap stop is `partial_incomplete` and cannot be reported as a completed comparison. The scientific report must regenerate from verified stored evidence, retain all planned coverage and missing work, distinguish viable, infeasible, unknown, invalid and censored cases, and verify any selected Class B headline by independent replay. The four Phase 3.5 directed pairs are primary, the three Phase 3.3 pairs secondary, and six starting-PPO/classical pairs descriptive. The study uses three correlated training seeds and a simplified pump simulator; no independent-sample significance or real-world safety claim is justified.

## Actual coverage and resource accounting

| Stage | Planned | Completed | Censored | Pending |
|---|---:|---:|---:|---:|
| Fixed development stress | 408 | 24 | 0 | 384 |
| Disturbance grid cells | 3,264 | 119 | 27 | 3,118 |
| Differential proposals | 1,400 | 0 | 0 | 1,400 |
| Durable pair archives | 7 | 0 | 0 | 7 |
| Final report work item | 1 | 0 | 0 | 1 |

The completed work comprises 24 fixed-stress episodes and 3,650 evaluated grid episodes (25 for each of 146 cells), or 3,674 controller episodes. The fixed-stress records retain 3,787 actual transitions. The compact grid records do not retain episode lengths, so the **exact total transition count cannot be reconstructed** from this evidence; the approved 17,633,600-transition figure remains a plan ceiling, not actual use. Ten of 12 nominal anchors have a grid-derived result for the first controller. No other controller was evaluated.

The cumulative budget ledger records 1,998.938 wall seconds and 1,833.832 evaluation seconds for `ppo_3_10240`; no other controller has charged evaluation time. The 1,800-second PPO limit was crossed within the last work item and enforced before another item was scheduled. The recorded controller usage is **33.832 seconds above the nominal cap**, a work-boundary overshoot that must not be described as strict cap compliance. No budget reset or extension occurred. The local run directory occupies 3,466,051 bytes after partial report generation, well below the 2-GiB storage cap.

## Evidence and report verification

The durable store verified the plan, work-item semantic hashes, stage coverage and budget ledger. All 170 recorded work items are completed or censored; none is failed, invalid or left running. The independently regenerated report is `partial_incomplete` with semantic SHA-256 `ba2c2cbe91d3acba7fb046015a4efb8abaf7049326b0132194b11d14156b5a68`. Its raw JSON SHA-256 is `cf8be4b4f050aab80e86aebd4fe428165b1a7334705d8f91ec48fdb005198ccc`; its raw Markdown SHA-256 is `2c4d2e60deeb2138372ec5b60a1e5ab7e527d95457c7b52e06bba962894c8fcf`. Repeating the supported `--mode report` command succeeded with the same report semantic hash and existing stored files. A local integrity manifest hashes 183 operator and run files and has semantic SHA-256 `45fe0ccf74b3efe4262a62148f9656acb9aec0d5a86705a1b5817bb75a03fe32`. It remains under ignored `artifacts/robustness/phase37_checkpoint2b/integrity_manifest.json`; no private or large evidence is committed.

No differential proposal or archive ran, so there are no selected Class A or independently replayed Class B findings. All seven planned primary/secondary matched comparisons are incomplete. The [partial-run report](PHASE_3_7_CHECKPOINT_2B_PARTIAL_RUN_REPORT.md) describes the limited single-controller observations without making a controller-comparison claim. A small [machine-readable partial summary](evidence/phase_3_7_checkpoint_2b_partial_summary.json) records verified coverage and resource use; its semantic SHA-256 is `1776370efd00b8e934c83e712f52c4bfaca50780c6fd1c1cd306ecc7d47b4025`.

## Deviation and disposition

The approved per-PPO budget prevented completion much earlier than the engineering projection. The historical warm-throughput estimate did not bound this host's actual `ppo_3_10240` evaluation time. The boundary overshoot and absent compact-grid transition lengths limit claims about strict resource compliance and exact transition usage. The scientific protocol was not altered, and the run was not resumed after exhaustion. Further execution requires a separately reviewed resource decision; the current authorization does not permit an extension. PR #2 remains draft and unmerged.
