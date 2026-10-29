# SIRFEX Phase 3.7 Checkpoint 2A.1 report

## A. Repository state and Git history

Work began on clean `phase-3.7-checkpoint-2a-completion` at `04660f2abc90352857bfffddf3498823972c8a02`, tracking `origin`. The [draft PR #1](https://github.com/mahadbaig2/sirfex/pull/1) remains open against `main`; nothing was merged. This milestone's commits through the final audit are:

| Commit | Change |
| --- | --- |
| `b646e12` | Document the independently checked GitHub CI history. |
| `61dd066` | Separate unknown physical feasibility from proven viability in reports. |
| `98d85eb` | Journal unclean invocations, preserve controller time across session eviction and test resource stops. |
| `2b3bfcf` | Require a separate operator authorization record and document its procedure. |
| `4671b79` | Require complete scenario sets, reject missing budget ledgers, use exact JSON storage estimates and preserve a new synthetic pilot. |

Each commit was pushed to the working branch. Changed files include `scientific.py`, `study_store.py`, `study_execute.py`, `study_report.py`, `cli.py`, focused unit and integration tests, execution and reproducibility guides, the CI audit, authorization procedure and synthetic pilot artifact. This report documents the state before its own publication commit.

## B. GitHub Actions CI audit

The repository has one Ubuntu/Python 3.12 CI workflow. It runs on `main` pushes and pull requests; branch pushes without a PR do not directly trigger it. Draft PR #1 supplies coverage for each push. [CI audit](PHASE_3_7_CI_AUDIT.md) records the original `main` formatting failure at [run 40891587732](https://github.com/mahadbaig2/sirfex/actions/runs/40891587732), then PR failures from missing `agents` dependencies and tests expecting ignored checkpoint binaries. The earlier corrective commits `80a917a`, `1508eff` and `04660f2` fixed those causes. The latest observed run for this milestone, [40971519690](https://github.com/mahadbaig2/sirfex/actions/runs/40971519690) at `4671b79`, **passed both `test` and `training`**. Formatting and lint also passed in that run. This does not make historical `main` runs green; `main` has not received these commits.

## C. Scientific reporting correction

The authoritative analyzer returns `target_feasible` or `bounded_operation_feasible` only for proven finite-horizon viability, `infeasible` for proven physical impossibility, and `undetermined` when its horizon budget cannot decide. Scientific report schema `pump-scientific-study-report:2` counts these separately, along with invalid/missing evidence and censored domain points. Viable, infeasible and unknown evaluated counts sum exactly to valid evaluated grid points. Only proven viable controller failures enter `avoidable_grid_violations`; unknown failures remain separate. A synthetic mixed-case test covers survival, failure, infeasibility, undetermined outcomes and missing evidence. A paired comparison now requires the full planned scenario set on both sides, even if both controllers omit the same scenario.

## D. Budget and crash recovery

`invocation.json` records active versus closed execution, a durable wall-clock checkpoint, controllers active at a work item and any unclean-restart recovery event. A restart of an active invocation requires explicit `resume`. The ledger conservatively charges the wall-clock gap since the last checkpoint to cumulative wall time and to controllers active at the crash. The recorded range is lower bound zero to that gap; it includes downtime, so it is **not** a precise execution-time measurement. Backward wall-clock movement fails closed. The existing monotonic clock controls each live invocation. Controller session eviction now retains accumulated evaluation seconds. An existing study with a missing budget ledger is rejected instead of reset to zero.

Normal artifacts reserve headroom for atomic budget and journal writes. JSON size checks use the same indented serialization as the atomic writer, and Markdown checks use UTF-8 byte length. Budget and journal writes use atomic replacement. If the filesystem itself cannot accept a ledger write, execution raises an explicit error; the previous ledger remains valid and an operator must review uncertainty before recovery. Tests inject wall and controller cap crossings, an unclean restart, clock rollback, ledger write failure, case/archive/report storage stops, and preserved completed evidence. This policy is single-process local recovery, not distributed locking or hard real-time preemption.

## E. Authorization

The [authorization procedure](ROBUSTNESS_STUDY_AUTHORIZATION.md) requires explicit human review, approved protocol identity, a committed frozen protocol, exact committed bytes, a separate machine-readable approval record, approved manifest/inventory/budget/work-plan hashes, and an operator-supplied digest. The runner snapshots the record and actual invocation request in its ignored local run directory and rejects a changed record on resume. Synthetic isolated tests verify matching and mismatched record fields, missing approval, changed committed bytes and unreachable commits. A record's self-hash checks integrity, **not independent cryptographic authentication**; `approved_by` remains a human assertion that must be checked against the review trail. No real record was issued.

## F. Scientific implementation audit

PPO checkpoint hashes are verified before model loading; controller identity excludes machine-local checkpoint paths. Session eviction retains timing without changing deterministic inference semantics. Fixed stress and nominal grid evaluations retain scenario identity; primary Phase 3.5 pairs and secondary Phase 3.3 pairs remain distinct. Matched comparisons now expose missing planned scenario IDs. Prepared grids remain shared, ordered and hash-bound; physical infeasibility, unknown feasibility, controller failure, censoring and nonmonotone outcomes remain separate. Differential proposals remain seeded and nonadaptive, with excluded proposals consuming budget. Selected Class B headlines still require persisted originals, repeats and independent replay. Complete scientific reports require mandatory work, complete matched pairs and verified headline evidence; partial work is labeled. No result is interpreted as a demonstrated robustness improvement merely from a margin gap.

## G. Testing

Local checks use `.venv\Scripts\python.exe -m pytest -q`, `.venv\Scripts\ruff.exe check .`, `.venv\Scripts\ruff.exe format --check src tests scripts` and `git diff --check`. The full local suite passed **243 tests**, with four existing dependency warnings. The candidate scientific dry run passed and returned protocol digest `aaef7614397fadd37f96be31afad1bdde240742528348330acfd15b76b338258`, plan digest `1ee6babcd78ac5e51c09ebfb054c8f6e03f14f7b2ee35dadf1308f51b5c1acef`, 17 controllers, 12 anchors, seven pairs and an 88,168-episode upper bound. A CLI attempt to execute the candidate was refused before controller work. The focused report, recovery, budget and authorization suites passed. The latest independently observed PR run before publication, [40971519690](https://github.com/mahadbaig2/sirfex/actions/runs/40971519690), passed both jobs; final-publication CI status must be checked separately.

The new [bounded synthetic pilot](evidence/phase_3_7_study_pilot_2a1.json) completed three classical controllers, one anchor, one direction, three grid points, one proposal, one Class B archive, four independent replays and a schema-v2 report. It measured 5.047 seconds and 108,406 artifact bytes in this local run; Python-traced peak was 330,811 bytes and excludes native allocations. It is engineering evidence, not a scientific comparison or PPO throughput estimate. The earlier pilot artifact remains unchanged.

## H. Protocol review

The machine candidate remains unchanged and non-executable. Its design binds 15 PPO checkpoints and two classical references; 12 nominal anchors, 24 development stress cases, 16 directions and 25 grid points over 0–0.60; four directed Phase 3.5 primary pairs, three directed historical Phase 3.3 pairs and six classical context comparisons; 200 proposals per directed pair; five selected findings per class; a 0.05 Class A margin gap; and disabled restricted physical-ceiling refinement. Caps are eight cumulative hours, 30 minutes per PPO controller and 2 GiB of artifacts.

The prior warm PPO projection for the grid is about 3.86–5.08 hours, plus differential, startup, physical analysis, persistence and reporting. The 8-hour limit and 30-minute controller caps have plausible but unproven headroom; the synthetic pilot cannot validate real PPO throughput. The 2-GiB cap is far above the small synthetic archive, but prior case/trajectory byte estimates are examples rather than worst-case bounds. Sequential work can truncate pairs asymmetrically; the report marks those pairs incomplete. No budget was silently increased. **Recommendation: ready for a final human freeze decision with explicit acceptance of these resource uncertainties and the conservative crash-accounting policy.** Keep restricted refinement disabled. If an operator requires guaranteed completion within the caps, obtain a separate bounded PPO calibration and approve any proposed limit change before freeze.

## I. Scientific boundaries

No research controller was trained or modified; no weights, reward, action normalization, simulator physics or scenario partitions were changed. The existing CI training job performs a short synthetic PPO smoke test, which is distinct from the scientific controller cohort. No sealed final holdout was opened, no full Phase 3.7 comparison ran, no frozen protocol was committed, and no real authorization record was issued.

## J. Remaining blockers

**Before execution:** independent human review and explicit approval of exact protocol bytes, controller and scenario inventories, directed pairs, resource ceilings and the crash recovery policy; a committed frozen protocol; and a separate recorded approval identity. The real PPO runtime and worst-case archive size remain estimates. A budget stop during a future study must stay incomplete until separately reviewed. Optional improvement: a larger bounded CPU throughput calibration on the intended study host, without using sealed final scenarios.

## K. Final status

`checkpoint_2a_1_complete`
