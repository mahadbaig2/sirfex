# Phase 3 PPO Controller

Phase 3.2 introduces an opt-in normalized action coordinate and revised reward.
The historical commands below keep their legacy defaults. New checkpoints stamp
the policy action, reward and termination schemas and refuse incompatible model
loading or continuation. Raw Gaussian inspection lives in the isolated
`train diagnose-policy` command; physical recorded actions remain in experiment
traces. For the controlled measurements and selection rule see
[the competence study](PPO_COMPETENCE_STUDY.md), [action diagnostics](PPO_ACTION_DIAGNOSTICS.md),
and [reward study](REWARD_TERMINATION_STUDY.md).

```powershell
sirfex train diagnose-policy --config configs/training/action_diagnostics.json
sirfex train diagnose-reward --config configs/training/reward_diagnostics.json
sirfex train competence --config configs/training/pump_competence.json
sirfex train report <study-id>
```

For scientifically gated studies use [Phase 3.1's protocol](EXPERIMENT_PROTOCOL.md)
and [learning-curve results](PPO_LEARNING_CURVES.md). The original Phase 3 commands
below remain historical pipeline demonstrations; they do not enforce all Phase
3.1 study gates. New studies use two CPU threads and explicit minimal/calibrated
schemas, while old checkpoints retain their original three-input contract.

Phase 3 uses Stable-Baselines3 PPO as an optional CPU-only training dependency. SIRFEX owns scenario construction, evaluation, evidence, and provenance; SB3 owns rollout collection, policy updates, optimizer state, and PPO loss calculations.

The trainable pump adapter exposes a three-value observation `[level, target, demand]` and a one-value continuous command in `[0, 1]`. It delegates every physical transition to `PumpEnvironment`, so training and evaluation use the same tank dynamics. The training reward is the existing pump reward minus `10 * violation_magnitude` for a transparent constraint penalty. A reward value is never used to redefine a physical failure.

The default policy is a small MLP with hidden layers `[64, 64]`, CPU device, conservative rollout sizes, and four Torch threads. Profiles are `smoke`, `quick`, `standard`, and opt-in `extended`. Smoke and quick runs are pipeline checks, not claims of controller competence.

PPO checkpoints are trusted local files only. Each checkpoint has a run-scoped ID, SHA-256 digest, environment and space metadata, PPO configuration, seed, timestep budget, and dependency versions. Loading verifies the path and hash before Stable-Baselines3 deserialization. Remote URLs and unknown artifact directories are not loaded automatically.
