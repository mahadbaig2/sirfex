# Reproducibility

An experiment records its scenario seed, environment version, controller configuration and metadata, evaluation configuration, dependency versions, SIRFEX version, and optional Git commit. Numerical trajectories are persisted as JSON Lines.

Replay reconstructs only controllers with an explicit factory. Phase 1 supports the random LunarLander controller and pump rule controller. Replay returns:

- `exact_match`: all recorded fields match exactly.
- `tolerance_match`: only configured numerical tolerance differences remain.
- `mismatch`: recorded evidence differs materially.
- `unavailable`: artifacts, dependencies, environment, or controller reconstruction are unavailable.

Replay is deterministic on the verified local Python 3.12 environment for the Phase 1 controllers. Exact cross-platform or cross-version reproducibility is not promised. Different hardware, dependency versions, or unrecorded controller state can change numerical traces.

```powershell
sirfex experiments replay <experiment-id>
```
