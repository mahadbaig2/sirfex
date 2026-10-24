# Phase 3.1 — Controller competence, feasibility and robustness validation

## Executive outcome

Validation infrastructure is implemented. The bounded pilot did **not** establish
a competent PPO baseline: both three-input and calibrated six-input policies
failed all 12 physically feasible validation scenarios at both 10,240 and 25,600
steps. All 16 pre-existing replay-verified discoveries were analytically
infeasible over their horizons. Both continuation gates failed. No matched
continuation was run, and robustness is **inconclusive**.

This is a completed negative pilot, not evidence that failure-informed learning
improves or worsens robustness. The original pump physics, failure evidence and
reward were preserved. Phase 4 was not started.

## Evidence and audit

The [baseline audit](PHASE_3_1_AUDIT.md) records training/validation reuse, hidden
plant parameters, infeasible historical stress cases, episode-weighting problems,
and missing budget/uncertainty checks. The baseline repository had 46 passing
tests. The [reviewable JSON](evidence/phase_3_1_pilot.json) preserves this pilot's
full partitions, definitions, metrics, hashes and source references without local
absolute checkpoint paths.

Measured run: `20261024T225829_4cc56983a8`, 2026-10-25 local date, code `981a902`.
Post-pilot diagnostic traces and later continuation-path hardening do not change
the saved protocol or learned checkpoints. Search sources were
`20261024T190804_6ad4db5bbf` and `20261024T190814_e6724213b0`.

## Mathematical feasibility and discoveries

The [feasibility analysis](PUMP_FEASIBILITY.md) separates target equilibrium,
interval equilibrium, finite-horizon safe reachability and unavoidable violation.
It propagates exact rational reachable intervals under bounded continuous actions
and the unchanged affine discrete pump model. Empty safe reachability proves
infeasibility; a capped unresolved horizon yields `undetermined`.

| Existing discoveries | Count |
|---|---:|
| Valid | 16 |
| Supporting replay verified | 16 |
| Target-feasible | 0 |
| Bounded-operation-feasible | 0 |
| Infeasible | 16 |
| Undetermined or invalid | 0 |
| Selected for improvement curriculum | 0 |

The infeasible cases remain valid observed failures and useful operating-envelope
evidence. They cannot support a claim that a controller should have avoided them.
No original discovery, trajectory, replay record or historical report was edited.

## Observations, rewards and competence

The [learning-curve report](PPO_LEARNING_CURVES.md) includes every measured curve
point, reward-component mean, checkpoint SHA-256 and CPU measurement. Both schemas
used the same physical scenario pools, PPO settings and 25,600-step budget.

| Validation metric | Rule reference | Minimal PPO, both budgets | Calibrated PPO, both budgets |
|---|---:|---:|---:|
| Constraint violation frequency | 0% | 100% | 100% |
| Maximum violation magnitude | 0 | .0422524 | .0422524 |
| Mean absolute tracking error | .068919 | .246537 | .246537 |
| Worst tracking error | .234073 | .553108 | .553108 |
| Mean episode length | 200 | 11.75 | 11.75 |
| Full-horizon frequency | 100% | 0% | 0% |
| Execution failures | 0 | 0 | 0 |
| Training-form episode return | -2.516920 | -1.146768 | -1.146768 |

All 564 actions in the four-checkpoint validation trace collection were zero.
Forty-eight experiments were persisted through the existing experiment service;
four representative replays matched exactly. This supports the observed action
saturation diagnosis. Failed short episodes obtained higher undiscounted return
than the rule reference, so return alone would give a misleading ranking.
That finding motivates a separate reward/termination study, not an unplanned
reward change during this comparison.

The predeclared gate requires zero validation violations/errors, all full horizons,
and mean tracking error <= .10. No PPO checkpoint qualified; **selected baseline
and selected-baseline SHA-256: not applicable**. Diagnostic checkpoint hashes
are listed in the learning-curve report. More observations did not help this
pilot, but the experiment cannot separate reward/optimization failure from the
possible benefits of calibrated state.

## Partitions and continuation

The pilot froze four disjoint 12-scenario pools: training, nominal validation,
nominal final holdout and held-out stress. Every generated scenario was
analytically target-feasible. All physical near-duplicate and seed-disjointness
checks passed. Generator seed 3101, full definitions and partition hashes are
in the evidence export. Pump seed changes alone are not independent physics.

The [protocol](EXPERIMENT_PROTOCOL.md) specifies checkpoint selection using only
validation, fixed observation assumptions, and separate discovery eligibility.
Continuation would load identical selected checkpoint hashes, schema, reward,
PPO configuration and continuation seed; each arm receives equal additional
rollout-multiple steps, with a 0% versus 30% episode-level discovery mixture.
The standard profile provides three independent training seeds (3, 17, 41) and
25,600 additional steps per arm. It was **not run** after the pilot gates failed.

| Final comparison | Observed status |
|---|---|
| Nominal final holdout | Not evaluated; reserved |
| Known feasible stress | Unavailable; zero eligible discoveries |
| Held-out stress | Not evaluated; reserved |
| Matched continuation pairs | 0 |
| Robustness classification | Inconclusive |
| Training-seed confidence interval | Unavailable; one pilot seed, no pairs |

Nominal validation metrics above must not be relabeled as final held-out results.
The [uncertainty specification](ROBUSTNESS_VALIDATION.md) uses paired training
seeds and exploratory bootstrap intervals when at least three distinct seeds
exist. An integration test uses an explicitly stubbed gate to exercise continuation
wiring; it is not scientific evidence of competence or improvement.

## CPU and artifact costs

The study trained 51,200 transitions in 121.923 seconds total wall time, below its
900-second cap, on CPU with two Torch threads and one worker. Observed segment
throughput ranged from 353.8 to 569.6 steps/second. Final RSS was 275.9 MiB and
peak process working set 279.7 MiB. Models were 140,248–144,976 bytes each.
The in-study artifact-size snapshot was 768,352 bytes; later trace collection and
the review export are additional. Memory and timings are machine-specific.

## Verification and implementation changes

Local checks: **75 tests passed**, Ruff formatting/lint passed, `pip check` passed,
all four pilot checkpoints passed SHA-256 verification, and dependency-isolation
checks confirmed core CLI/search/replay/evidence imports do not load SB3, Torch
or Streamlit. Studio's existing server responded `ok` on port 8765. Three
upstream Box2D SWIG deprecations and one SB3 action-normalization advisory remain.
The `[0,1]` physical action contract was preserved. GitHub CI now has a separate
optional CPU-training job in addition to the core dependency job.
The older main CI job omitted the `search` extra despite running an Optuna test;
its dependency installation was corrected after inspecting the remote failure.
Both Linux jobs then passed on `4df2879` ([CI evidence](https://github.com/mahadbaig2/sirfex/actions/runs/40142046307)).

Actual implementation files changed:

- Added `training/feasibility.py`, `protocol.py`, `selection.py`, `diagnostics.py`,
  `validation.py` and `environments/pump_observations.py` under `src/sirfex`.
- Updated `cli.py`, experiment factories/replay, and training checkpoint,
  controller, adapter, configuration, evaluation, persistence, PPO, regression
  and legacy-runner modules.
- Added two Phase 3.1 configs under `configs/training`, unit validation tests and
  integration pipeline tests; expanded `.github/workflows/ci.yml`.
- Added policy-inspection and portable-report-export scripts.
- Added this report, audit, feasibility, learning-curve, protocol, uncertainty
  documentation and public evidence JSON; updated README, changelog, architecture,
  PPO controller and curriculum documentation.

`environments/pump.py` and the technical PRD were not changed. The repository's
public About description remains the official SIRFEX description.

Git checkpoints: `981a902` adds the validation infrastructure and initial tests;
`b00b887` adds continuation/recovery coverage, curriculum compatibility, baseline
comparisons and diagnostic/export scripts; `4df2879` corrects continuation seed
metadata and CI search dependencies. The documentation commit adds this measured
report and review evidence. All changes are pushed without
rewriting history; local model binaries remain ignored by Git.

## Acceptance

| Requirement | Status | Evidence or limitation |
|---|---|---|
| Phase 3 interfaces remain operational | PASS | Existing tests, compatible checkpoint defaults, original commands retained |
| Pump physics unchanged | PASS | Original pump module unchanged; transition equivalence tests |
| Feasibility analysis implemented | PASS | Exact affine interval analyzer |
| Feasible/infeasible distinguished | PASS | Boundary, equilibrium and finite-horizon tests |
| Uncertainty explicit | PASS | Capped reachability gives `undetermined` |
| Existing failures preserved | PASS | Read/replay analysis; no source artifact mutation |
| Minimal observation retained | PASS | Three-input defaults and old metadata support |
| Extended observation versioned | PASS | Six-input schema, shape/hash checks, replay tests |
| Reward components inspectable | PASS | Per-episode tracking/actuation/constraint sums |
| Learning curves generated | PASS | Four measured checkpoints |
| Competence gate implemented | PASS | Both learned variants correctly rejected |
| Selection excludes final holdouts | PASS | Validation-only selection; no final evaluation in pilot |
| Reproducible partitions | PASS | Saved physical definitions, IDs and hashes |
| Leakage checked | PASS | Physical-near-duplicate and seed checks |
| Matched budgets verified | PASS | Pipeline/test capability; no eligible real study |
| Physical metrics separate from proxies | PASS | Violations, rewards, tracking and errors separate |
| Regression guardrails enforced | PASS | Nominal guardrail and comparison tests |
| Uncertainty reported appropriately | PASS | No interval or significance claimed for this pilot |
| Local artifacts persisted | PASS | Completed and interrupted runs; replayable diagnostics |
| Automated tests pass | PASS | 75 local tests |
| CPU supported | PASS | Measured bounded CPU pilot |
| Existing CLI and Studio operational | PASS | CLI verification, evidence-reader tests, live health check |
| Documentation reflects completed work | PASS | Negative result and unavailable evaluations explicit |

Scientific outcomes: competent PPO baseline **NOT ESTABLISHED**; valid matched
continuation **NOT RUN**; held-out robustness **INCONCLUSIVE**. These are distinct
from software acceptance. Broad scientific validation remains **PARTIAL** because
there is only one training seed and no admissible continuation comparison.

## Limitations, known defects and recommended next milestone

The original reward/termination design admits a misleading return ranking and
the measured policies are unusable. The legacy Phase 3 workflow still has the
experimental limitations in the audit and is labeled accordingly; use Phase 3.1
for gated studies. The calibrated schema assumes known plant parameters, fixed
dt/bounds/horizon and omits remaining time. Decimal reachability does not certify
floating-point boundary behavior or a real plant. Wall deadlines are cooperative;
abrupt process death during a checkpoint write can leave an unusable partial ZIP.
Three-seed exploratory bootstrap intervals would still be limited, even after
competence is achieved. No unresolved test failure is known.

Recommended next milestone: a separately specified, bounded reward/termination
and action-distribution diagnosis with a simple calibrated analytical reference,
then a predeclared competence study across independent seeds. Obtain feasible,
replay-verified controller failures before a matched robustness study. Do not
scale training blindly or treat impossible demand as a controller-learning task.
No further implementation was started as part of this recommendation.
