# Phase 3.2 audit and predeclared remediation decisions

Starting commit `5d6af19`, clean working tree; 75 baseline tests passed locally.
The PRD, reports, evidence, pump/controller/training/search/replay code and CI were
reviewed. Original physics and Phase 3.1 evidence are retained unchanged.

## Gate A: measured action path

New inference on the four saved policies, on 12 frozen validation initial states,
found Gaussian deterministic means in ranges [-2.270,-2.073], [-3.083,-2.851],
[-3.242,-3.139], and [-5.045,-4.992]. Standard deviations remained .892–.962.
SB3's unsquashed Gaussian predicts the mean deterministically and clips it to
the Box bounds [0,1]; the pump receives the resulting zero. Thus the adapter,
NumPy conversion and replay are not the source of the zeroing. On 384 newly
sampled actions per checkpoint, the earliest minimal model had six positive
actions; all samples from the other checkpoints clipped to zero. These are new
diagnostics, not reconstructed historical rollout samples.

The installed SB3 2.9.0 `common/policies.py` and `on_policy_algorithm.py` confirm
clipping at both inference and rollout collection. `DummyVecEnv` identifies time
limits as `truncated and not terminated`; PPO bootstraps the terminal observation
for time limits but not physical violation terminations. Diagnostic distribution
inspection is isolated because its Gaussian mean/stddev access is version-sensitive.

## Gate B: measured reward/reference evidence

Across the same 12 validation scenarios, discounted legacy returns (gamma .99)
were -1.0910 for hysteresis, -1.0536 for zero command, -1.5022 for full command,
and -.2923 for calibrated feedback. Both fixed commands failed all cases; both
references completed all horizons. Calibrated feedback MAE was .001904.

An early-failure incentive is plausible and demonstrated relative to hysteresis,
but failure is not the global optimum established by these examples: analytical
control obtains a better legacy return. Learning and clipping interact. A positive
survival/tracking reward is a hypothesis to test, not a proven universal repair.

## Frozen sequence

A: reuse Phase 3.1 calibrated seed 3, direct actions, legacy reward, 10,240/25,600
steps. B: change only policy coordinates to [-1,1], mapped to physical `(a+1)/2`.
C: keep B's coordinates and change only reward to `1 - 4*error^2 - .01*u^2 -
10*violation_magnitude - 1[terminated]`. The unit bonus rewards completed safe
operation, tracking remains dense, and the terminal cost is modest rather than
an arbitrarily huge multiplier. On the reference traces, revised discounted
returns are 83.659 (hysteresis), 6.653 (zero), 6.959 (full), 86.218 (analytical).

Keep calibrated six-value observations, physical dt/bounds/horizon, nominal
training/validation pools (generator 3101, 12 each), PPO [64,64], learning rate
.0003, rollout128, batch32, epochs4, gamma.99, GAE.95, entropy0 and value weight.5
fixed. B and C are separate sequential pilot commands capped at 25,600 steps and
600 seconds each. Pilot throughput is measured before larger studies. If a
promising configuration emerges, the standard study uses seeds 3,17,41 and
checkpoints 10,240/25,600/51,200 under a 900-second cap. The unchanged competence
gate is zero violations/errors, full horizons and MAE <=.10; final holdouts remain
unused. No uncontrolled optimizer or observation normalization search is planned.

D (continuing after physical violation) is rejected for the training comparison.
The original model clips volume at [0,1]; extrapolating after empty/full clipping
would silently create/remove mass relative to the affine equation used for
feasibility. Boundary violations remain terminal and explicitly versioned.
No new continuation physics or relabeling of failures is introduced.

These decisions were recorded before B/C training. Further work depends on their
measured physical results, not reward alone. No robustness continuation is part
of Phase 3.2.
