# SIRFEX Phase 3.7 Checkpoint 2B: partial run

**Status:** `checkpoint_2b_partial_budget_exhausted`. The development-only experiment stopped at the first PPO controller's 30-minute evaluation budget. These observations do not answer the planned comparative research questions.

## Scope actually observed

The approved frozen protocol is `27a2b4ca8bb1552c22c8d0c3ad07931b14d0b676d28656a2af4796c7fb75bb04`, the deterministic plan is `ead9af1ef00824016e6dea102ef4b7a3f37b3eb6948cfff238b1dae6839947c2`, and the verified partial scientific JSON is `ba2c2cbe91d3acba7fb046015a4efb8abaf7049326b0132194b11d14156b5a68`. Full evidence is retained in the ignored local run directory identified in the [execution report](PHASE_3_7_CHECKPOINT_2B_EXECUTION_REPORT.md). The counts below are observations within one incomplete controller workload, not independent samples or a full-cohort estimate.

`ppo_3_10240` completed all 24 development stress scenarios and 146 of its 192 planned anchor/direction grid cells. Ten of 12 nominal validation anchors have grid-derived controller results. The other 16 controllers have no observations. The grid evaluated 3,650 points; 2,445 were independently classified physically viable and 1,205 physically infeasible. Among the viable evaluated points, 511 had a modeled physical violation and 1,934 survived. All 1,205 evaluated infeasible points had a violation and are kept separate from avoidable controller failures. There were no unknown-feasibility points in the processed cells. Each point can be correlated with other points on the same anchor and direction.

Of the 146 processed grid cells, 119 had a first observed failure on the finite grid and 27 were right-censored at the maximum tested delta. In 111 cells with both an observed failure and a finite physical-infeasibility threshold, the observed failure preceded that threshold. No processed cell had a nonmonotone failure sequence. These thresholds are observations on `0.000, 0.025, ..., 0.600`, not global minima.

The 24 completed stress episodes included six modeled physical violations. Their tracking MAE median was approximately 0.2633 and post-initial signed-margin median approximately 0.0870. The ten observed nominal episodes had no modeled violation; their tracking MAE median was approximately 0.0528 and post-initial signed-margin median approximately 0.3456. These are within-controller descriptive summaries over different scenario sets. The paired treatment and classical reference comparisons have **zero complete matched coverage**, and differential search and independent replay did not begin. No controller ranking, robustness improvement, or safety claim follows.

## Why the run stopped

The durable budget ledger records 1,833.832 PPO evaluation seconds against the approved 1,800-second scheduling cap, plus 1,998.938 cumulative study wall seconds. The runner completed the work item that crossed the cap, producing a 33.832-second boundary overshoot, then scheduled no further work. The run used about 3.47 MB of local artifacts against the 2-GiB artifact cap. The process exited cleanly with a closed invocation journal and no recovery charge. No budget was increased or reset.

The partial JSON and Markdown reports regenerate from verified stored evidence. The current compact grid evidence omits per-episode lengths, so an exact total transition count cannot be recovered. The approved experiment remains incomplete under its current authorization. Any additional evaluation would need a separately reviewed protocol and resource decision; these partial observations must remain visible in that review.
