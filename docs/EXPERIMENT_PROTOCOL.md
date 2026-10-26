# Phase 3.1 experiment protocol

Phase 3.2 keeps the nominal gate and physical scenario partitions below intact.
Its separate [competence study](PPO_COMPETENCE_STUDY.md) freezes calibrated
observations, training/validation dataset seed 3101, and first-passing
checkpoint selection per seed. [Avoidable search](AVOIDABLE_FAILURE_DISCOVERY.md)
rechecks that selected PPO on nominal validation and excludes all frozen
partition physical keys and seeds. Neither command evaluates final holdouts
or runs a matched robustness continuation.

`sirfex train validate --config configs/training/pump_validation.json` runs a
bounded study. `learning-curve` uses the same diagnostics but never continues
training. The pilot configuration is `configs/training/pump_learning_curve.json`.

## Frozen decisions

Before training, the study saves configuration, competence thresholds, full
scenario definitions, partition hashes, reward schema, and a protocol SHA-256.
The pump equation, failure rules, reward, PPO architecture and hyperparameters
are held constant across observation schemas.

The minimal observation is `[level,target,demand]` (`pump-minimal:1`). The calibrated
observation appends `[efficiency,capacity,leak]` (`pump-calibrated:1`). Calibration
is assumed available only for this comparison, not for industrial deployment.
Capacity and leak must be at most 10 for the calibrated space. Physical `dt`,
bounds and horizon remain fixed; remaining episode time is not observed. Old
three-input checkpoints remain three-input checkpoints. Schema/shape mismatch
and hash mismatch are explicit errors.

Each schema starts a fresh SB3 MLP `[64,64]` for each independent training seed.
That seed controls policy initialization, stochastic rollout sampling and training
scenario selection. The continuation seed resets sampling randomness for both
arms while preserving the common starting policy and optimizer. Scenario seeds
do not perturb the deterministic pump equation.
Successive curve checkpoints continue the same in-memory model and optimizer.
Both schemas use the same scenario pool and timestep budgets; differing episode
lengths can produce different sampled scenario sequences. This is an
episode-sampled comparison, not identical transition replay.

Competence requires zero validation violations, zero execution errors, every
episode reaching its full horizon, and scenario-weighted mean absolute tracking
error <= 0.10. Select the first qualifying checkpoint for each schema/seed. A
schema must qualify for every requested training seed. Among qualifying schemas,
choose the lowest validation tracking error. Final holdouts are not consulted.

## Partitions

`pump-partitions:1` produces training, nominal validation, nominal final holdout,
and held-out stress pools from a recorded NumPy generator seed. Pilot seed 3101
produces 12 scenarios per pool; standard seed 3102 produces 24. Pump trajectory
seeds are disjoint but are bookkeeping: this plant is deterministic. Different
physical parameter vectors establish the split.

All pools use `dt=.1`, horizon 200, and bounds `[.05,.95]`. Capacity is uniform
in `[.75,1]`, efficiency `[.8,1]`, initial level `[.3,.7]`, and target `[.4,.6]`.
Normal leakage is uniform `[0,.03]`, stress leakage `[0,.08]`. Demand is
`fraction*capacity*efficiency - .5*leak`, with fraction `[.45,.65]` for normal
pools and `[.75,.90]` for held-out stress. Every generated scenario must pass
target-feasibility analysis.

Cross-partition leakage checks reject shared seeds and parameter vectors whose
resolved physical values all differ by at most `1e-4` after five-decimal rounding.
Horizon is excluded from near-duplicate comparison. IDs hash normalized physical
parameters; partition hashes cover complete records. This tolerance is a concrete
local rule, not proof that all statistically related situations have been excluded.
Selected discoveries form a separate curriculum pool. Known-stress evaluation
deliberately reuses this pool and is never evidence of generalization.

## Matched continuation

Both arms load the exact selected SHA-256 and preserve the same reward, schema,
PPO settings, additional steps and continuation seed (`training_seed + 5000`).
Normal-only samples no discoveries; failure-informed samples eligible discoveries
with probability 0.30 at each episode reset. Actual sampled episode counts are
saved. The mixture refers to episodes, not transitions.

Normal-only and failure-informed arms receive equal additional rollout-multiple
budgets. Loaded PPO settings are checked against the requested configuration.
The unchanged baseline, both arms, and rule reference use the same fixed final
scenario sets. Checkpoint hashes, evaluation identity and actual steps must match
the protocol before comparison. A partial arm cannot be called a matched study.

If no schema passes competence or no eligible feasible discovery is available,
the study completes with an explicit stopping reason and inconclusive robustness.
No final holdout metrics are fabricated or substituted from validation results.

## Resources and persistence

| Profile | Curve steps per schema/seed | Seeds | Max wall time | Additional steps per arm | Scenarios/pool |
|---|---|---|---:|---:|---:|
| smoke | 128, 256 | 3 | 90 s | 256 | 4 |
| pilot | 10,240, 25,600 | 3 | 900 s | 10,240 | 12 |
| standard | 10,240, 25,600, 51,200 | 3, 17, 41 | 3,600 s | 25,600 | 24 |
| extended | 25,600, 51,200, 102,400 | 3, 17, 41 | 7,200 s | 51,200 | 32 |

Defaults are CPU, two Torch threads, one worker, rollout 128, minibatch 32,
four optimization epochs, learning rate .0003, gamma .99, GAE .95, entropy
coefficient 0, value coefficient .5. The selected profile imposes caps; overrides
can reduce work. Extended requires `--allow-extended`.

The wall deadline is cooperative: checked between stages and on rollout steps.
An ongoing optimizer update, filesystem write, or individual evaluation can
overrun it briefly; this is not OS process preemption. Timestep caps are exact
rollout multiples. Persisted manifests distinguish requested caps from actual
steps, wall time, throughput, available process memory and artifact size.

Runs live under `artifacts/training/<run-id>`. JSON is atomically replaced;
checkpoints are local trusted SB3 files with SHA-256 metadata. Interruptions
retain a marker, manifest, completed curves and the active partial model where
available. `sirfex train report <run-id>` reads complete or partial artifacts.
Abrupt process termination during model serialization may leave an incomplete
ZIP; only a successful hash-verified checkpoint is usable. `verify` checks hashes.
