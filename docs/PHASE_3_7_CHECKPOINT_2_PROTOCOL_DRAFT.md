# Phase 3.7 Checkpoint 2 scientific protocol proposal

**DRAFT — NOT FROZEN — NOT AUTHORIZED FOR EXECUTION**

This document is for human review. The paired [machine draft](../configs/robustness/phase37_checkpoint2_protocol.draft.json) is deliberately non-executable (`executable: false`). Checkpoint 1 calibration accepts only `software_calibration_only` and cannot launch this study. A separate frozen protocol, commit, artifact budget, and explicit approval are required.

## Controller inventory and analyses

The proposed primary descriptive comparison pairs the Phase 3.5 **screening** normal policy with its 5% and 10% expanded-stress counterparts within seeds 3 and 17. The six local screening checkpoints have matching 61,440 total recorded training timesteps, exist, and their bytes match manifest SHA-256. The Phase 3.5 manifest status is `no_eligible_curriculum`; these are screening checkpoints, not a successful approved curriculum or final continuation. All six still need loadability checks before execution.

Secondary historical comparisons pair Phase 3.3 normal versus 30% robust continuations within seeds 3, 17 and 41. Their six files exist and match manifest SHA-256. The three Phase 3.2 selected starting policies (seeds 3, 17, 41) are context controllers. The rule-based and analytical controllers are references. These groups must not be pooled into one dose-response series: interventions, exposure and source study differ. Only the Phase 3.2 seed-3 checkpoint was loaded during Checkpoint 1.1 calibration; `sha256_verified` and `loadability_tested` are distinct inventory fields.

## Fixed development scenarios

Use only previously designated development nominal and screening stress scenarios, preserving physical parameters, scenario seeds and semantic identities. The safe, development-only identity export is **pending**. It must not load, iterate over, compare against or expose Phase 3.4/3.5 sealed final partitions. Before freeze, record an explicit identity/hash manifest for up to 12 anchors from authorized development-only files, plus a duplicate and seed-exclusion audit. Do not substitute scenarios silently.

## Disturbance measurement

Propose `delta_max=0.60` and `grid_step=0.025` (`25` points including zero) for all 16 declared directions. This is a finite resolution grid: `delta_first` is the first observed violating grid point, not a proven global minimum. Parameter bounds cause censoring, unknown feasibility remains unknown, and the restricted physical ceiling bracket is distinct from controller failure. Validate the selected anchor/direction domains before freezing; the full 25-point grid does not guarantee all generated scenarios are valid.

## Differential search

Propose exploratory Class A threshold `0.05` normalized tank-level fraction while preserving continuous signed margin gaps. Do not tune the threshold to obtain a positive result. Class B requires finite-horizon viability, X failure, Y survival, complete evidence, persisted original/repeat episodes and independent regeneration. Class A remains a safe near miss, with its own replay status. Prioritize verified Class B, then Class A. Propose seeded nonadaptive random search using the development-only domain in `configs/search/pump_avoidable.json`, at most 200 candidates per pair and five archived headlines per class. Proposed search seed is `3701`; new scenario seeds start at `37000000`. Confirm namespace and duplicate exclusions against development sources only, never final partition contents. Pair definitions and exact count remain to be approved.

## Execution controls

The worst-case grid has `17 × 12 × 16 × 25 = 81,600` episodes and `16.32 million` simulator transitions at horizon 200. Proposed guardrails are eight hours wall time, 30 minutes per PPO controller, two GB of artifacts, and a stop on any feasibility/simulator contradiction or corrupt evidence. Persist only compact grid candidates and full trajectories for selected findings. Verify every checkpoint SHA-256 before load. Use immutable completed cases, explicit failed-case recovery, deterministic IDs, atomic status writes, progress counts and a human-approved protocol hash. Any full-study command must refuse draft protocol status.

## Interpretation and open decisions

Results would be development-only within a simplified pump simulator. Zero observed violations is not a safety guarantee; rankings are descriptive; related PPO checkpoints are not independent experimental units; negative and null results are reportable. Open decisions are the safe scenario manifest, final controller selection, full loadability preflight, grid/domain check, differential budget and seeds, exclusion rules, wall/storage caps, and the protocol freeze commit. No controller-improvement claim follows automatically.
