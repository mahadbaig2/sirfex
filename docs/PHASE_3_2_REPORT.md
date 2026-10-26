# SIRFEX Phase 3.2 report — reward, action and competence remediation

Measured 26 October 2026 (Asia/Karachi) on local Windows/Python 3.12. The
unchanged simplified pump model, physical `[0,1]` command domain, constraint
definition and Phase 3.1 nominal competence gate were used throughout.

## Outcome and readiness

**Readiness: `ready` for designing the matched robustness continuation study.**
All three independent PPO seeds reached zero nominal validation violations and
errors, full 200-step horizons in all 12 cases, and mean level error ≤ .10. A
bounded feasibility-aware search then produced three distinct physical violations
by the preselected seed-3 policy. For each, both the analytical and existing rule
controllers completed the *identical* scenario safely; all failing and reference
experiments replayed exactly. These are simulated, reference-demonstrated
avoidable controller failures. No robustness continuation, final holdout test or
claim of improved robustness is part of this milestone.

The [public evidence JSON](evidence/phase_3_2_remediation.json) contains the
versioned configurations, hashes, partitions, diagnostic aggregates, learning
curve rows, search trials, physical scenarios, experiment IDs and witness commands.
Full trajectories and binary SB3 checkpoints remain in local artifact stores;
the compact export omits per-step arrays. The unchanged negative
[Phase 3.1 report](PHASE_3_1_REPORT.md) remains the historical comparator.

## Why the old PPO commands were zero

The historical calibrated and minimal PPO policies used a Gaussian in the
asymmetric policy interval `[0,1]`. New inference on four saved checkpoints,
under 12 fixed validation initial observations and 32 seeded samples per
observation, found raw Gaussian deterministic means below zero in every case.
At the calibrated 25,600-step checkpoint the mean range was -5.0448 to -4.9921,
with mean standard deviation .9251; all 384 sampled outputs were clipped to
zero. At minimal 10,240 steps, 98.44% of stochastic samples were zero after
clipping, while deterministic actions were all zero. The adapter applied those
zeros without further attenuation. The historical 564 recorded validation
actions were also zero. The raw values here are **new inspection of saved
models**, never attributed to unrecorded historical rollout samples.

The supported `model.predict` result agreed exactly with separately inspected
Gaussian means after SB3 clipping. A version-sensitive distribution inspection is
isolated in diagnostic code. `PPOController` now records deterministic/stochastic
inference mode, and seeded stochastic replay uses its own Torch RNG state.
The evidence establishes a negative learned policy mean and boundary clipping as
the immediate zero-command path. It does not prove why optimization moved the
means so negative; reward incentives, action coordinates and other PPO dynamics
may interact. [Action diagnostics](PPO_ACTION_DIAGNOSTICS.md) gives the full
checkpoint table and action-path description.

## Reward, termination and analytical reference

Legacy training reward was `-error² - .01*u² - 10*violation`. On identical
12-scenario trajectories, early-failing zero command averaged -1.1468 in
undiscounted legacy return versus -2.5169 for safe full-horizon hysteresis;
the corresponding finite discounted sums (gamma .99) were -1.0536 and -1.0910.
However, safe calibrated analytical control scored even better (-.6356
undiscounted, -.2923 discounted). Early failure was therefore advantaged over one
safe comparator, but not shown to be a globally optimal legacy strategy.
SB3's time-limit bootstrap also differs from a finite return sum.

The revised `pump-survival-tracking:2` training formula is
`1 - 4*error² - .01*u² - 10*violation - 1[terminated]`, in dimensionless
per-step units. The unit bonus values completed operation, with denser tracking
and a modest terminal cost. It is explicitly versioned and not policy-invariant
shaping. Reward scoring never alters pump states or physical failure labels.
The original terminate-on-violation behavior stays versioned as
`pump-terminate-on-violation:1`. Continuing after violation was rejected as a
training experiment because the original empty/full state clipping can break the
affine mass-balance equation on which the current feasibility proof relies.
No alternative boundary physics were silently introduced. See the
[reward study](REWARD_TERMINATION_STUDY.md).

The informed analytical reference computes equilibrium command
`(demand + leak*target)/(capacity*efficiency)` and clips it after a feedback
correction with gain `(.25/dt - leak)/(capacity*efficiency)`. Without saturation,
the modeled tracking-error pole is .75. It uses level, target and calibrated
demand/capacity/efficiency/leak plus dt; it therefore has more model information
than minimal PPO. Across 12 nominal cases it had zero violations, 200-step
horizons and MAE .001904, versus hysteresis MAE .068919. Both fixed zero and
full-command controls violated all 12 cases. See the
[derivation and assumptions](ANALYTICAL_PUMP_CONTROLLER.md).

## Controlled action/reward comparison

The [predeclared audit](PHASE_3_2_AUDIT.md) fixed observation design, data
generator, nominal validation set, PPO settings, physical model and termination.
Study A is the historical calibrated six-observation PPO (direct `[0,1]` policy
actions, legacy reward). Study B changed only policy coordinates to `[-1,1]`,
explicitly mapped to the same physical `[0,1]` pump command. Study C kept B's
action coordinate and changed only the reward to the formula above. D, continuing
after violation, was rejected before training for the stated physical reason.

| Seed-3 study | Steps | Violations / 12 | Full horizons | MAE | Physical commands |
|---|---:|---:|---:|---:|---|
| A, historical direct + legacy | 10,240 / 25,600 | 12 / 12 at each | 0 / 12 | .246537 | All zero |
| B, normalized + legacy | 10,240 / 25,600 | 12 / 12 at each | 0 / 12 | .286625 | All 169 recorded actions one at each checkpoint |
| C, normalized + revised | 10,240 | 0 / 12 | 12 / 12 | .053288 | All 2,400 actions interior |
| C, normalized + revised | 25,600 | 0 / 12 | 12 / 12 | .036859 | All 2,400 actions interior |

Action normalization alone changed lower saturation into upper saturation and
was inadequate. The combination with revised reward learned nominal control in
the pilot. The result does not isolate reward as a universal causal fix, and
evaluation uses physical competence rather than higher training reward.
Pilot B run `20261025T204620_eb148a7c0a` used 117.84 seconds wall, with
training segments 52.23/63.06 seconds. Pilot C run
`20261025T204951_e773d5feaa` used 122.92 seconds wall; its first segment
trained at about 196 steps/s. The 128/256-step smoke was used only for software
and throughput checks and did not count as competence evidence.

## Replicated nominal competence

Standard study `20261025T205339_64b622c146` used seeds 3, 17, 41 and
checkpoints 10,240/25,600/51,200, all under a 900-second deadline. The unchanged
gate selected the **first** passing checkpoint per seed from nominal validation:

| Seed | Selected steps | Checkpoint | SHA-256 | Validation MAE | Violations/errors | Full horizons | Physical lower/upper saturation |
|---:|---:|---|---|---:|---:|---:|---:|
| 3 | 10,240 | `ppo_3_10240` | `859c9244e83fce5cf31600d9f7fae0038e053fc8b3dc1663218ed616257e70f9` | .053288 | 0/0 | 12/12 | 0 / 0 |
| 17 | 10,240 | `ppo_17_10240` | `05b5fe45c1f46b7eb28699bd4a1dc693e9c11d56fba3376211c5985168a9d333` | .078340 | 0/0 | 12/12 | 0 / 0 |
| 41 | 25,600 | `ppo_41_25600` | `48fe432e373aef9f64e55e0092f9b09ba95ef407bcc6d81f76a1f61432fd83f0` | .076652 | 0/0 | 12/12 | .000417 / 0 |

Seed 41's 10,240-step MAE was .114120 and failed the gate despite zero
violations; it first passed at 25,600. The other eight checkpoints passed, and
all nine had zero violations and full horizons. At 51,200 steps the MAEs were
.016863, .028521 and .020125 for seeds 3, 17 and 41, respectively. Those later
scores were *not* used to select the search baseline. Postselection validation
trace inspection recorded 2,400 physical actions per selected checkpoint and
exact representative replays, without changing selection.

The standard run consumed 153,600 training steps across three independent
seeds, 634.48 seconds wall and 839.73 CPU seconds. Peak process working set was
294,662,144 bytes (about 281 MiB), and checkpoint ZIPs were 143,779–145,188
bytes. Training segments ranged about 199–273 steps/s, with each 12-case
validation taking about 3.5–4.2 seconds. This is local hardware evidence, not a
guaranteed throughput on other machines. Detailed curves, action statistics and
partition hashes are in [the competence study](PPO_COMPETENCE_STUDY.md) and
public JSON. No final nominal or stress holdout outcome was consulted.

## Physically avoidable failures

Search `20261026T090912_05c97a1dec` used selected seed-3 checkpoint
`ppo_3_10240`. Its SHA was verified and the full frozen nominal gate rechecked
before search. The original pump constraint-violation objective, random strategy
seed 32, 24 proposals and 120-second inclusive deadline were declared in
`configs/search/pump_avoidable.json`. Fourteen valid proposals were rejected as
not finite-horizon viable; ten feasible cases were simulated. Three produced
controller violations; all three became distinct replay-verified discoveries.
No invalid proposals or execution errors occurred. The search evaluation itself
took 7.54 seconds; competence recheck, references and replays brought total wall
time to 31.89 seconds. It executed 12 validation rechecks, ten candidate
simulations, six reference simulations, three search discovery replays and
twelve witness replays (two per reference record). These counts are separate.

| Discovery ID | Initial level | Demand | Capacity × efficiency | Leak | Stored failure event step | Safe references | Evidence |
|---|---:|---:|---:|---:|---:|---|---|
| `disc_05c97a1dec_0000` | .716162 | .780245 | 1.278721 × .647666 | .111708 | 32 | Analytical, rule | Both exact |
| `disc_05c97a1dec_0002` | .301774 | .391040 | .648879 × .633014 | .036241 | 22 | Analytical, rule | Both exact |
| `disc_05c97a1dec_0001` | .178079 | .431105 | .521581 × .955029 | .028178 | 29 | Analytical, rule | Both exact |

Each row has fixed target .5, limits .05/.95, dt .1, horizon 200 and a persisted
scenario seed. The discovery IDs are unique despite six successful reference
records. Every reference completed all 200 steps without a constraint violation;
each admissible command sequence is retained as a finite-horizon constructive
witness. The first case was classified bounded-operation feasible despite target
equilibrium limitations; the other two were target feasible. Identical physical
parameters and seeds, failure events, controller hash, experiment IDs, replay
reports and 200-command witnesses are in `avoidable.json` and the public export.
An equilibrium calculation alone did not confer an avoidability label. See the
[discovery protocol](AVOIDABLE_FAILURE_DISCOVERY.md).

## Verification, compatibility and limitations

- The original pump dynamics and Phase 3.1 report were not edited. Legacy
  checkpoints retain direct action coordinates and legacy reward defaults;
  hash, action-space and training-contract checks prevent silent reinterpretation.
- The existing CLI, search strategies, experiment persistence and replay remain
  available. Optional SB3/Torch and Streamlit imports stay outside core use.
  The Evidence Explorer can display stored avoidability records read-only.
- New tests cover action conversion, invalid values, stochastic replay, reward
  accounting and termination, analytical control, bounded filtering, witness
  validity, exclusion paths and TPE availability. CI includes a separate CPU
  training lane without the multi-minute competence study.
- The pump is a simplified deterministic numerical model. A safe reference trace
  demonstrates avoidability only under that model and its calibration and
  horizon assumptions. Three seeds and 12 nominal cases are a useful bounded
  replication, not a guarantee across arbitrary controllers or conditions.
- The `ready` classification records scientific prerequisites for a *future*
  matched study. The existing Phase 3.1 continuation command uses its historical
  direct-action/legacy-reward contract; a future continuation must explicitly
  carry the new checkpoint's action/reward contract through both matched arms.
  This report neither runs that study nor claims an improvement.

## Acceptance assessment

| Area | Status | Basis |
|---|---|---|
| Historical artifacts, pump physics, original CLI and core dependency isolation | PASS | Code diff, legacy defaults, regression tests and import check |
| Raw/applied action diagnostics, versioned action/reward/termination metadata | PASS | Four historical checkpoint inspections and contract checks |
| Analytical reference and reward/termination analysis | PASS | Formula derivation, 96 persisted diagnostic episodes and replays |
| Three-seed nominal PPO competence under frozen gate | PASS | Nine validation checkpoints; first passing per seed selected |
| Feasibility-filtered, replay-verified avoidable failure | PASS | Three discoveries, six safe reference records and witnesses |
| Optional Studio evidence visibility | PASS | Stored search evidence rendered read-only |
| Generalization beyond the frozen nominal cases | PARTIAL | No final holdout outcomes evaluated in Phase 3.2 |
| Matched robustness improvement | FAIL / not studied | Continuation deliberately not run |
| Industrial safety inference | FAIL / unsupported | Simplified, unvalidated physical model |

Implementation history was pushed as `f8e0f10` (action/reward/analytical/competence
foundation), `f61d4b8` (feasibility-aware search and witnesses), `372ead2`
(diagnostic integration, CI and initial audit), and `3249339` (selected-policy
evidence, readiness and Explorer support). The measured report and public export
follow in a documentation commit. No Phase 4 work, release or package publication
occurred.
