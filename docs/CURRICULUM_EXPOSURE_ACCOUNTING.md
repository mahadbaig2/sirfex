# Curriculum exposure accounting

`PumpGymEnv` records each physical transition in a versioned `pump-curriculum-exposure:1` ledger. Scenario assignment occurs on `reset()`, after an episode terminates or truncates. Stable-Baselines3 rollout boundaries do not assign a new scenario. The ledger counts starts, completions, active partial episode interactions, and actual nominal/stress interactions independently. It records per-scenario and per-discovery starts and interactions, distinct discoveries, completed stress episode lengths, sampler seed and explicit reseeds.

`configured_stress_probability` is the Bernoulli probability at episode reset. In `interaction-debt:1` mode it is `null`; `target_stress_interaction_fraction` is the requested intensity. The realized fraction is always `stress_interactions / actual_environment_interactions`. The scheduler selects stress at an **episode boundary** when accrued stress interactions fall below the target fraction of accrued total interactions. It completes every physical episode normally and does not change the PPO step budget.

Each PPO run reconciles `nominal_interactions + stress_interactions = actual_environment_interactions = model additional timesteps`. Completed episode lengths plus partial interactions reconcile separately for each mode. Unassigned scenarios, interrupted resets and mismatched model counts mark the ledger invalid. Historical artifacts retain their original fields; their missing transition counts cannot be reconstructed from episode counts.

The accounting implementation is in [exposure.py](../src/sirfex/training/exposure.py) and [pump_gym.py](../src/sirfex/training/pump_gym.py).
