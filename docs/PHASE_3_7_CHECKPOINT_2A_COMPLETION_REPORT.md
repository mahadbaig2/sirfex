# SIRFEX Phase 3.7 Checkpoint 2A completion report

**Engineering status:** `checkpoint_2a_complete_awaiting_protocol_review`

**Scientific execution status:** `not_authorized_not_run`

**Protocol:** [candidate JSON](../configs/robustness/phase37_checkpoint2_protocol.candidate.json), still `executable: false`.

## Scope and Git history

The implementation started from clean `main` commit `0b244a05e59b9f5de8c5a45e0d15314acf653f6e` and was developed on `phase-3.7-checkpoint-2a-completion`. Each milestone was committed and pushed to `origin`; no commit was merged into `main`.

| Commit | Milestone |
| --- | --- |
| `34b1889` | Bind scientific work to a deterministic plan and dry-run plan hash. |
| `ae6eff9` | Add durable work-item ledger and cumulative budgets. |
| `c9a269b` | Execute synthetic fixed scenarios and shared physical grids. |
| `30c96e2` | Persist deterministic proposals, differential evaluations and replayed headline evidence. |
| `b2cfbd7` | Regenerate scientific JSON and Markdown from verified stored artifacts. |
| `4a013be` | Gate execution on an explicitly approved, exact committed frozen protocol. |
| `d62d3b4` | Verify interruption recovery and cumulative resource stops. |
| `3be8321` | Guard archive storage writes, run bounded pilot and document execution. |

The starting Checkpoint 2A preflight and candidate artifacts on `main` were preserved. The candidate JSON was not modified to imply authorization.

## Inputs and execution design

The existing [controller preflight](evidence/phase_3_7_controller_preflight.json) records 15 verified PPO checkpoints and two classical references. The development-only [manifest](evidence/phase_3_7_development_scenario_manifest.json) contains 12 training exclusions, 12 nominal anchors, 24 screening stress scenarios and seven historical public discoveries. The existing [physical-domain preflight](evidence/phase_3_7_disturbance_domain_preflight.json) covers 4,800 transformed points and records 1,598 physically infeasible cases. The new read-only dry run rechecked local checkpoint hashes, manifest identities, grid, pair design and search-space bytes. It returned protocol SHA-256 `aaef7614397fadd37f96be31afad1bdde240742528348330acfd15b76b338258` and work-plan SHA-256 `1ee6babcd78ac5e51c09ebfb054c8f6e03f14f7b2ee35dadf1308f51b5c1acef`.

The plan contains 408 fixed development-stress items, 3,264 controller/anchor/direction grid cells, 1,400 differential proposals, seven pair archives and one report item. Its upper episode budget remains 88,168, with 17,633,600 maximum simulator transitions. Every cell shares a prepared, controller-independent physical grid. Each proposal is sampled from a predeclared seed and consumed even when excluded. The runner keeps at most two controller sessions open and persists compact margins, tracking errors, grid outcomes, proposal classifications and coverage. Selected findings receive durable original/repeat episodes for both controllers and independent replay checks.

The report presents controller distributions, matched primary and secondary pairs, nominal/stress context, grid thresholds, differential coverage and selected evidence. It uses `partial_incomplete` when work or a required matched comparison is missing, and does not call an unverified selected Class B complete. It rejects altered archived evidence. Identical verified artifacts regenerate the same semantic report hash.

## Recovery and resource limits

Work items, prepared grids and proposal sequences are bound to the immutable plan. Completed and censored items are write-once; interrupted or failed items require explicit `resume`. Integration tests injected failures during a grid cell, differential proposal, selected archive after durable saves, and report write. Each resumed run produced the same semantic report hash as an uninterrupted synthetic run. A corrupted completed cell stopped resume.

The budget ledger preserves cumulative monotonic wall time across restarts and tracks per-controller evaluation time. It rejects a resumed run at the same exhausted ceiling. Storage checks account for the run directory before atomic work writes and estimate archive episode bytes before individual durable saves. Stops leave completed evidence intact and do not turn partial work into a scientific result. A limit extension would change plan identity and require separate review; it is not automatic on resume.

## Bounded synthetic pilot

The [pilot artifact](evidence/phase_3_7_study_pilot.json) was generated with `python -m scripts.pilot_phase37_study`. It used three classical test controllers, one synthetic nominal anchor, one direction, three grid points, one proposal and 20-step scenarios. It completed fixed evaluation, grid, differential search, one selected Class B archive, four independent replays and report generation. Measured total time was 2.125 seconds; fixed/grid 0.781 seconds, grid preparation 0.031 seconds, differential 0.813 seconds, archive including replay 0.500 seconds, replay 0.062 seconds, and report 0.468 seconds. The run directory occupied 104,942 bytes; peak Python-traced allocations were 280,417 bytes, excluding native allocations. The completed synthetic report hash was `eb7f9c1fbaf2661468f729625ca9f00902ce9025d84f71cb18e44dbdd9c1c427`.

These timings establish that the complete software path runs and persists evidence. They do not predict the 17-controller PPO study's runtime, memory peak or scientific outcome. The pre-existing [execution budget estimate](evidence/phase_3_7_execution_budget.json) remains a planning estimate, not a performance guarantee.

## Validation and limits

Repository-wide `python -m pytest -q` passed with 230 tests and four pre-existing dependency warnings. `ruff check .` and changed-file `ruff format --check` passed. The real candidate dry run passed and returned the counts above; the candidate execution gate rejects it. `git diff --check` passed at the pilot push. The synthetic pilot is separate from any future scientific result.

Remaining decisions belong to protocol review: whether to approve the exact controller inventory and one-directional pairs, the fixed grid and disabled restricted ceiling refinement, search/exclusion rules, the eight-hour/30-minute-per-PPO/2-GiB ceilings, and the final frozen bytes and approval digest. A synthetic pilot cannot establish the real PPO cohort's headroom under those limits. A real run that exhausts a limit must remain incomplete and be reviewed before any extension. The finite grid is not a global robustness minimum, and related PPO checkpoints are not independent trajectories.

No controller was trained or modified. No sealed final scenario content or outcome was accessed. The full Phase 3.7 scientific comparison was not run. No protocol was frozen, and no approval token was issued.
