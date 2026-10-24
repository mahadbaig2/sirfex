# Failure Taxonomy

Phase 1 separates execution status, observed outcome, and failure classification.

## Execution status

`completed` means the runner completed an episode and produced a result. Exceptions remain visible to callers and are not silently converted into successful results. Future persistence layers may record failed execution manifests.

## Failure categories

- `nominal`: no failure event was observed.
- `constraint_violation`: a declared physical or operating constraint was violated.
- `objective_failure`: the environment reported failure of its declared objective.
- `invalid_action`: a controller produced an invalid action.
- `invalid_observation`: an environment produced an observation outside its declared space.
- `numerical_instability`: the model produced non-finite or otherwise invalid numerical state.
- `controller_exception`: controller code raised an exception.
- `environment_exception`: environment execution raised an exception.
- `timeout`: execution exceeded its configured step/time budget.
- `unknown`: the evidence is insufficient for a stronger classification.

Failure records include an explanation, optional step and measurements, evidence source, and evidence status (`observed`, `inferred`, or `unknown`). Truncation alone is not a failure. Stock LunarLander terminal outcomes are `unknown` unless an environment-provided outcome field is available.
