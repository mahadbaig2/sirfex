# SIRFEX Phase 0 Architecture

SIRFEX is a local Python package with a `src/` layout. Phase 0 deliberately keeps the execution path in one process and does not introduce a database, web service, plugin registry, or distributed worker system.

## Package boundaries

- `sirfex.core`: serializable `Scenario` and `EpisodeResult` types, adapter protocols, and the environment-agnostic `EvaluationRunner`.
- `sirfex.environments`: environment-specific physics and API translation. `LunarLanderEnvironment` wraps stock Gymnasium; `PumpEnvironment` owns the simplified single-tank model.
- `sirfex.controllers`: controller adapters. Phase 0 provides a random controller and a pump hysteresis controller.
- `sirfex.benchmarks`: measured machine and simulation benchmark helpers.
- `sirfex.cli`: the small `sirfex` command-line entry point.
- `sirfex.experiments`: versioned manifests, atomic file storage, controller reconstruction, replay, and descriptive comparison.
- `sirfex.search`: bounded candidate spaces, risk objectives, strategies, trial orchestration, discovery ranking, search persistence, and fair comparison.
- `sirfex.visualization`: UI-independent, read-only evidence readers and chart/playback transformations. It has no Streamlit or Plotly dependency.
- `sirfex.training`: optional Gymnasium pump adapter, lazy Stable-Baselines3 PPO orchestration, trusted checkpoint metadata, curricula, fixed-set evaluation, regression criteria, and local training persistence.

The optional `studio/app.py` Streamlit application consumes `sirfex.visualization` and persisted artifacts. It is not part of the engine execution path and cannot create experiments, searches, or scenarios.

## Execution flow

1. A CLI command selects an environment and controller.
2. A `Scenario` supplies an environment identifier, parameters, version, and seed.
3. `EvaluationRunner` resets the adapter, resets the controller, validates spaces, and executes up to the configured step limit.
4. The adapter reports environment-specific outcome classification and metadata.
5. `EpisodeResult.to_dict()` provides a JSON-serializable record.

The runner does not branch on LunarLander or pump behavior. Physics, termination rules, and environment metrics stay in the adapters.

## Experiment workflow

The Phase 1 workflow is:

```text
Configure → Validate → Execute → Classify → Persist → Reload → Replay → Compare
```

The file store records a manifest, scenario, result summary, and JSONL trajectory. Replay uses explicit environment/controller factories and reports exact match, tolerance match, mismatch, or unavailable. Comparison is descriptive and only marks experiments comparable when their environment identifiers and versions match.

Phase 2.5 adds a local evidence-reading path:

```text
ExperimentStore/SearchStore → EvidenceReader → recorded transformations → optional Studio charts
                                             └→ existing replay verification
```

The Explorer never calls an environment's `step()` while browsing a stored trajectory and never changes authoritative ranking or failure classification.

Phase 3 composes training after failure evidence without making the evaluator depend on Stable-Baselines3:

```text
Phase 2 search artifacts → eligible discoveries → ScenarioCurriculum
                       → PumpGymEnv (delegates PumpEnvironment physics)
                       → optional Stable-Baselines3 PPO
                       → checkpoint integrity → EvaluationRunner metrics
                       → regression report and local training artifacts
```

## Phase 3.1 validation

Phase 3.1 adds scientific validation around the optional training workflow:

```text
Original discoveries → supporting experiment replay → feasibility interpretation
Frozen physical partitions → leakage check → matched observation learning curves
                          → validation-only competence/selection
                          → eligible feasible curriculum + competent checkpoint
                          → matched continuations → final held-out evaluation
                          → paired training-seed uncertainty and nominal guardrails
```

Absent competence or eligible discoveries stops continuation with an explicit
inconclusive result. `PumpObservationEnvironment` inherits unchanged pump dynamics;
the calibrated schema adds plant parameters without changing old checkpoints.
The shared `EvaluationRunner` supplies all physical metrics. Diagnostic policy
traces use the existing `ExperimentStore` and replay service and are viewable in
Studio. Training aggregates and hashes remain in `artifacts/training`.
Studio remains an evidence reader. See [protocol](EXPERIMENT_PROTOCOL.md) and
[validation interpretation](ROBUSTNESS_VALIDATION.md).

## Search workflow

Phase 2 composes the experiment layer rather than duplicating it:

```text
SearchSpace → Strategy → Scenario validation → EvaluationRunner
           → RiskObjective → ExperimentStore → Trial/Discovery persistence
           → Replay verification → Strategy comparison
```

Search orchestration does not mutate third-party simulators. Environment adapters remain responsible for supported physical bounds and evidence; objectives label observed measurements versus inferred proxies.

## Pump model assumptions

The pump adapter uses a normalized tank level `h` in `[0, 1]`, a normalized pump command `u` in `[0, 1]`, and a fixed time step `dt`:

```text
h[t+1] = h[t] + dt * (pump_capacity * efficiency * u[t]
                       - demand - leak * h[t])
```

The model terminates when the level crosses the configured low or high operating limits and truncates at the configured horizon. These assumptions make conservation and numerical behavior testable without claiming that the model is a validated industrial plant.

## Dependency decisions

Python 3.12 is the development target. NumPy and Gymnasium are runtime dependencies. Box2D and pygame use the `simulation` extra. Ruff and pytest use `dev`. Stable-Baselines3 and Torch remain optional in `training` (also included by `benchmark`); core evaluation, search, replay and Studio do not require them for non-PPO artifacts.

## Reproducibility

Scenarios carry an explicit seed and version. LunarLander resets and seeds its action space. Evaluation results include environment/controller identifiers, scenario configuration, seed, termination state, outcome, metrics, and optional numerical traces. Exact reproducibility remains best-effort and is subject to dependency and platform behavior.

## Known limitations

- Stock LunarLander runtime physics are not modified by Phase 0. Gravity, wind, and degradation stress axes require future tested wrappers or custom dynamics.
- The pump is a transparent numerical model, not a validated industrial plant.
- Controllers are trusted Python code and are not sandboxed.
- Benchmark output is local evidence for the current machine, not a universal performance guarantee.
- LunarLander Studio views use recorded observations and do not reconstruct terrain or claim confirmed landing/crash semantics without adapter evidence.
