# Robustness Evaluation

Phase 3 evaluates four references on fixed scenario sets: the rule-based pump controller, the baseline PPO checkpoint, normal PPO continuation, and failure-informed PPO continuation.

The sets are:

- **Nominal:** normal operating scenarios not used as the final evaluation set for training.
- **Known stress:** eligible Phase 2 discoveries selected for the curriculum.
- **Held-out stress:** independently defined stress scenarios not used in curriculum selection.

Reported metrics include episode count, constraint-violation frequency, maximum and mean violation magnitude, mean and worst tracking error, mean reward, episode length, and execution failures. These are descriptive metrics of the simplified pump model, not industrial safety certification.

The CLI persists all controller summaries in `evaluation.json`. Checkpoint replay can be verified through the existing experiment replay path when a PPO experiment is created with trusted checkpoint metadata.
