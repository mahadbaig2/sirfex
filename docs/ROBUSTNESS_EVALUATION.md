# Robustness Evaluation

Phase 3 evaluates four references on fixed scenario sets: the rule-based pump controller, the baseline PPO checkpoint, normal PPO continuation, and failure-informed PPO continuation.

The sets are:

- **Nominal:** normal operating scenarios not used as the final evaluation set for training.
- **Known stress:** eligible Phase 2 discoveries selected for the curriculum.
- **Held-out stress:** independently defined stress scenarios not used in curriculum selection.

Reported metrics include episode count, constraint-violation frequency, maximum and mean violation magnitude, mean and worst tracking error, mean reward, episode length, and execution failures. These are descriptive metrics of the simplified pump model, not industrial safety certification.

The CLI persists all controller summaries in `evaluation.json`. Checkpoint replay can be verified through the existing experiment replay path when a PPO experiment is created with trusted checkpoint metadata.

## Phase 3.3 final evaluation

The frozen matched study evaluates the three original competent PPO baselines
and both continuations per seed, plus the rule and analytical references, on
the same 12 nominal final, three known-stress and 60 physically distinct
feasible held-out stress configurations. The final sets are opened only after
all six continuation checkpoints complete and pass hash verification. The
known set measures adaptation to encountered conditions; the held-out set is
the prespecified generalization endpoint. Per-scenario outcomes and aggregate
violation frequency, magnitudes, tracking, horizon, command, saturation and
execution failures are persisted. Representative runs use the existing
experiment replay service. See [held-out methodology](HELD_OUT_STRESS_EVALUATION.md)
and [Phase 3.3 results](PHASE_3_3_REPORT.md).
