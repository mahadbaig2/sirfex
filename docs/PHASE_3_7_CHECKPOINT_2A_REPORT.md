# SIRFEX Phase 3.7 Checkpoint 2A report

## A. Repository and implementation state

Initial branch: `main`; initial commit: `3fb46ec199b647865d76515bb79fd4fc087f0907`; working tree was clean. Existing robustness identity, disturbance preparation, margin, differential search, evidence archive and immutable store were inspected and reused for design. New files are a development-only preflight script, protocol builder, guarded study entry point, tests and this review package. No unrelated file was changed.

## B. Controller preflight

[Versioned inventory](evidence/phase_3_7_controller_preflight.json) records each of 15 PPO source runs, checkpoint IDs, expected and actual hashes, metadata, file availability, factory load, observation/action compatibility and repeated deterministic action. All 15 passed locally. The two classical references were instantiated through the registered factory. No model was trained or changed. The manifest uses portable relative checkpoint identifiers; local absolute paths remain in ignored historical metadata only.

## C. Scenario preflight

[Development manifest](evidence/phase_3_7_development_scenario_manifest.json) contains 55 scenarios: 12 training exclusions, 12 nominal validation anchors, 24 Phase 3.4 screening stress scenarios and seven public development discoveries. Phase 3.2 generator output was stopped after validation; Phase 3.4 generation was stopped after screening. Both independently matched their published development hashes from the Phase 3.5 protocol. Discovery source records were read from separate search artifacts and their IDs matched the Phase 3.5 development discovery map. Existing `leakage_check` found zero overlaps across the four exported groups; it compares physical parameters within 1e-4 and requires distinct seeds. Seed disjointness alone does not establish physical independence. No bundled sealed-partition file was opened.

## D. Disturbance-domain preflight

[Domain evidence](evidence/phase_3_7_disturbance_domain_preflight.json) covers 12 anchors × 16 directions × 25 fixed grid points. All 4,800 transformed scenarios were valid; zero were bound-censored or undefined. Exact finite-horizon analysis classified 1,598 as infeasible and zero as unknown. No validity reopening or nonmonotone feasibility sequence was observed on the grid. The `dt × transformed leak` numeric condition passed for 180 adverse-axis cells, but the continuous monotonicity argument needs formal review; the candidate disables restricted refinement. Controller-independent prepared-grid identities are recorded for reuse. No controller performance was evaluated.

## E. Comparative protocol

[Candidate protocol](PHASE_3_7_CHECKPOINT_2_PROTOCOL_CANDIDATE.md) declares four primary equal-budget Phase 3.5 within-seed comparisons, three secondary Phase 3.3 historical pairs, and six separate starting-PPO versus classical contexts. X is the stress continuation, Y the matched normal continuation; reverse differential search is excluded prospectively. Six development-only screening exposure ledgers are included with realized stress interactions and completed episodes. Physical violation, signed margin, grid failure threshold and nominal/stress MAE remain distinct. Null and unfavorable outcomes are valid.

## F. Execution engineering

The new `sirfex robustness study` entry point accepts `preflight`, `dry-run`, `scientific-execute`, `resume` and `report`. Read-only modes verify the manifest, checkpoint bytes, exact grid, pair count and upper budget. Execution and resume reject candidates, missing or mismatched approval digests, and protocols not committed at HEAD. There is no issued approval token. The scientific adapter is deliberately disabled for this checkpoint. Existing case storage supports immutable completed cases, corruption detection and explicit failed-case recovery, but a full scientific grid/differential/archive runner and end-to-end resume/report tests remain to be completed before freeze. The current `report` mode reports readiness only, not scientific outcomes.

## G. Resource estimates

[Budget evidence](evidence/phase_3_7_execution_budget.json) gives 81,600 upper-bound grid episodes, 408 development-stress tracking episodes and 6,160 upper-bound differential/replay episodes, for 88,168 episodes and 17,633,600 maximum transitions. Fifteen PPO checkpoints are loaded in a nominal one-pass session plan. Measured prior warm 200-step PPO timings of 0.193–0.254 seconds suggest 3.86–5.08 hours of PPO grid rollout alone; 6,160 more PPO episodes would add 0.33–0.43 hours at that rate. Startup, feasibility preprocessing, storage, repeated loading and CPU contention add uncertainty. Eleven prior compact case records averaged 8,066 bytes, implying about 26 MiB for 3,264 similar cells. One prior trajectory was 2,571 bytes; this is a sample, not a bound. The proposed eight-hour, 30-minute/controller and 2-GiB limits require a bounded end-to-end timing/storage pilot. Pairs must be reported incomplete on uneven truncation.

## H. Testing

Commands: `.venv\Scripts\python.exe -m pytest -q` (214 passed, four dependency warnings); `.venv\Scripts\ruff.exe check .` (passed); `.venv\Scripts\ruff.exe format --check` on all changed Python files (passed); `.venv\Scripts\sirfex.exe robustness study --mode dry-run --protocol configs/robustness/phase37_checkpoint2_protocol.candidate.json --manifest docs/evidence/phase_3_7_development_scenario_manifest.json` (passed); `git diff --check` (passed); JSON parsing of five new machine artifacts (passed). Candidate `scientific-execute` was refused even with its own digest. New tests cover candidate gate, missing/mismatched approval, committed-byte refusal, scenario tampering/sealed partition, missing checkpoint, budget, exact pair design, synthetic interruption/resume, immutable reuse and corrupted case detection. These do not yet constitute complete real scientific execution/report coverage. The `pytest.exe` launcher alone had one repository import-path failure; `python -m pytest` passed. Repository-wide `ruff format --check .` flags two pre-existing unrelated files (`SIRFEX_Technical_PRD_and_Architecture_v1.md` and `scripts/export_phase36_smoke.py`); changed Python files are formatted.

## I. Scientific boundaries

No training, controller-weight, reward, physics, action-normalization, observation-normalization, Groq, or LangGraph changes. No full Phase 3.7 study was run. No sealed final scenario contents or outcomes were opened. Historical scientific evidence was not modified.

## J. Remaining decisions

Approve the exact controller/pair direction, feasible-domain refinement proof policy, hard budget after timing/storage pilot, full executor and resume/report validation, and eventual frozen protocol identity. None is approved by this candidate.

## K. Final status

`checkpoint_2a_incomplete`

The package is substantial preflight progress, but full gated execution/resume/report integration and the bounded storage pilot prevent a truthful readiness declaration. Publication was deferred until the user explicitly requested it.
