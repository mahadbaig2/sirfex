# Failure Discovery

Search runs are stored under `artifacts/searches/<search-id>/` with a manifest, search space, trial JSONL, discoveries, summary, and separate replay-verification files. Each discovery references the supporting experiment IDs instead of duplicating trajectories.

Discovery ranking is local to one objective and direction. Near-duplicates are grouped by a canonical parameter key that rounds floating-point values to five decimal places. Original trials remain preserved. Results from incompatible objectives or environments are not collapsed into a universal severity score.

An observed pump constraint violation is distinct from a tracking proxy, a LunarLander instability proxy, an invalid candidate, and an execution error. A discovery is only replay-verified when all supporting experiment replays report an acceptable verification status.

Held-out seeds are evaluated after search and are recorded as validation evidence, not fed back into optimization.
