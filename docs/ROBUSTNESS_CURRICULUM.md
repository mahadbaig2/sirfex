# Failure-Informed Robustness Curriculum

The Phase 3.1 workflow supersedes the original selection protocol for scientific
studies: it replays supporting evidence, checks physical feasibility and fixed
observation assumptions, freezes disjoint partitions, and gates on baseline
competence. Its stress ratio is sampled on every episode reset and actual counts
are recorded. See [the protocol](EXPERIMENT_PROTOCOL.md). All 16 existing
discoveries were infeasible in the measured audit; no continuation was run.

Phase 3.3 uses the three new Phase 3.2 feasible, replay-verified avoidable
discoveries as a frozen known-stress pool. Each robust episode reset samples a
30% stress branch, then samples uniformly among these discoveries; the other
70% draws from the frozen 12-scenario nominal training pool. The normal arm
draws only from the same nominal pool. Each arm gets 25,600 additional steps
from an independent load of the identical baseline policy and optimizer state.
Actual scenario episode counts and checkpoint hashes are saved. The 60 held-out
physical configurations are excluded from the curriculum. See the
[matched protocol](MATCHED_ROBUSTNESS_STUDY.md).

The description below documents the original Phase 3 pipeline. Its finite
prefabricated mixture can differ from the requested probability, and its
historical artifacts are not retroactively relabeled.

Phase 3 consumes Phase 2 search artifacts; it does not implement a new search algorithm. A discovery is eligible only when it is a valid pump scenario, has observed evidence, has supporting experiments, and is replay-verified with `exact_match` or `tolerance_match`.

The curriculum is deterministic and fixed-mixture. Normal scenarios are sampled from the documented moderate-demand distribution. Selected discovered scenarios are the stress pool. `curriculum_ratio` controls the stress probability; the default is 30%. A seeded NumPy generator selects the pool and scenario, and the generated curriculum is persisted with discovery and experiment provenance.

Normal continuation and failure-informed continuation both start from the exact baseline checkpoint and use the same PPO configuration and additional timestep budget. Held-out stress scenarios are constructed independently and are never added to the training curriculum.
