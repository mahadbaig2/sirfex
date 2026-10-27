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

Phase 3.4 froze four additional avoidable discoveries from a bounded
feasibility-aware search and compared 5%/10% episode-reset stress probabilities
using either the original three-case pool or the seven-case expanded pool.
Every candidate forked the same competent checkpoint for 5,120 development
steps. Each low-ratio arm actually sampled only one stress episode, so the
realized 5% and 10% comparisons were identical within each pool. No candidate
passed both development stress-gain and nominal-preservation rules; the
workflow stopped before final matched continuation. See the
[screening record](CURRICULUM_SCREENING.md) and [Phase 3.4 report](PHASE_3_4_REPORT.md).

Phase 3.5 records actual transitions rather than inferring exposure from
episode-reset probabilities. Its 0% control and 5%/10% expanded-pool arms use
an episode-boundary interaction target, with measured exposure and completed
episodes reported separately. A candidate needs adequate stress interactions,
completed stress episodes and discovery diversity on both development seeds
before its performance can qualify for confirmation. See the
[accounting definition](CURRICULUM_EXPOSURE_ACCOUNTING.md),
[gate](EXPOSURE_ADEQUACY_PROTOCOL.md) and
[frozen protocol](PHASE_3_5_EXPERIMENT_PROTOCOL.md).

The description below documents the original Phase 3 pipeline. Its finite
prefabricated mixture can differ from the requested probability, and its
historical artifacts are not retroactively relabeled.

Phase 3 consumes Phase 2 search artifacts; it does not implement a new search algorithm. A discovery is eligible only when it is a valid pump scenario, has observed evidence, has supporting experiments, and is replay-verified with `exact_match` or `tolerance_match`.

The curriculum is deterministic and fixed-mixture. Normal scenarios are sampled from the documented moderate-demand distribution. Selected discovered scenarios are the stress pool. `curriculum_ratio` controls the stress probability; the default is 30%. A seeded NumPy generator selects the pool and scenario, and the generated curriculum is persisted with discovery and experiment provenance.

Normal continuation and failure-informed continuation both start from the exact baseline checkpoint and use the same PPO configuration and additional timestep budget. Held-out stress scenarios are constructed independently and are never added to the training curriculum.
