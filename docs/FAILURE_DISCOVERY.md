# Failure Discovery

Phase 3.2 adds [avoidable-failure discovery](AVOIDABLE_FAILURE_DISCOVERY.md):
the existing pump constraint objective runs only after a finite-horizon
feasibility filter, and a controller violation qualifies only with a safe,
same-scenario, full-horizon reference trace plus matching replays. The additive
`avoidable.json` records witnesses, reference experiment IDs, exclusions and
separate counts for rejected proposals, simulation evaluations and replays.
Historical Phase 2 discoveries are not relabeled.

Search runs are stored under `artifacts/searches/<search-id>/` with a manifest, search space, trial JSONL, discoveries, summary, and separate replay-verification files. Each discovery references the supporting experiment IDs instead of duplicating trajectories.

Discovery ranking is local to one objective and direction. Near-duplicates are grouped by a canonical parameter key that rounds floating-point values to five decimal places. Original trials remain preserved. Results from incompatible objectives or environments are not collapsed into a universal severity score.

An observed pump constraint violation is distinct from a tracking proxy, a LunarLander instability proxy, an invalid candidate, and an execution error. A discovery is only replay-verified when all supporting experiment replays report an acceptable verification status.

Phase 3 consumes only observed, valid, replay-verified pump discoveries for failure-informed curricula. It preserves the Phase 2 ranking and records selected discovery and supporting experiment IDs in the training manifest.

Held-out seeds are evaluated after search and are recorded as validation evidence, not fed back into optimization.
