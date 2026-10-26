# Reproducibility

Phase 3.2 evidence stores policy action, observation, reward and termination
schemas in new checkpoint metadata and records physical commands separately
from raw PPO diagnostics. The action diagnostic reconstructs raw distributions
from saved models under a fixed seed; it never attributes them to historical
rollouts. New stochastic PPO replay uses a saved inference seed and private
Torch RNG state. The [Phase 3.2 report](PHASE_3_2_REPORT.md) identifies run,
checkpoint and search IDs and a compact public review export. Binary checkpoints,
complete trajectories and the immutable source studies remain in local stores;
cross-version bitwise identity is not promised.

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
