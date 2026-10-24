# Reproducibility

An experiment records its scenario seed, environment version, controller configuration and metadata, evaluation configuration, dependency versions, SIRFEX version, and optional Git commit. Numerical trajectories are persisted as JSON Lines.

Replay reconstructs only controllers with an explicit factory. Phase 1 supports the random LunarLander controller and pump rule controller. Replay returns:

- `exact_match`: all recorded fields match exactly.
- `tolerance_match`: only configured numerical tolerance differences remain.
- `mismatch`: recorded evidence differs materially.
- `unavailable`: artifacts, dependencies, environment, or controller reconstruction are unavailable.

Replay is deterministic on the verified local Python 3.12 environment for the Phase 1 controllers. Exact cross-platform or cross-version reproducibility is not promised. Different hardware, dependency versions, or unrecorded controller state can change numerical traces.

Search reproducibility additionally records the bounded space, strategy identifier, strategy seed, objective direction, trial budget, and scenario seed policy. Random search owns a controller-independent NumPy generator; TPE records its Optuna seed and startup-trial caveat. Held-out seeds are evaluated after optimization and are not used to update the strategy.

```powershell
sirfex experiments replay <experiment-id>
```
## Phase 3 training

Phase 3 records Python/library versions, Git commit, PPO configuration, CPU device, Torch thread count, training seed, scenario curriculum, checkpoint hashes, and evaluation sets. PPO training is reproducible within documented dependency and platform limits, not guaranteed bitwise-identical across operating systems. The `smoke`, `quick`, `standard`, and opt-in `extended` profiles make runtime intent explicit.
