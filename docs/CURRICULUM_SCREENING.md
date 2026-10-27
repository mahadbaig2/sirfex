# Phase 3.4 development-only curriculum screen

The principal control is 100% nominal episode sampling. The candidate ratios
are 5% and 10% known stress using the original three Phase 3.2 discoveries.
If bounded new search produces at least two additional replay-verified,
physically avoidable failures, 5% and 10% ratios using the expanded pool enter
the same screen. The historical 30% condition remains the measured Phase 3.3
comparator and is not retrained automatically. Ratio comparisons share a pool;
pool-diversity comparisons share a ratio.

A fixed seed-3 development checkpoint is reloaded independently for every
candidate and normal control. Every candidate receives 5,120 additional PPO
steps in the standard profile, identical optimizer starting state, PPO settings
and continuation seed. The screen evaluates the 12-case Phase 3.2 development
validation set and 24 newly generated development stress cases. Actual nominal
and stress episode exposures, checkpoint hashes, per-scenario outcomes and
training seconds are saved. Final nominal and stress holdouts are never opened.

A candidate is eligible only if its development nominal MAE is at most 110% of
the normal control MAE, it adds no nominal or development-stress violation,
adds no execution failure, and lowers development-stress violation frequency
or reduces stress MAE by at least `.005`. Among eligible candidates, select the
lowest stress violation frequency, then lowest stress MAE, then stable ID.
These are frozen experiment policy thresholds, not formal safety guarantees.
If none is eligible, record `no_eligible_curriculum` and stop without final
training or evaluation. Do not spend extra steps seeking a favorable result.

The benchmark, candidate list, rule, budgets and source hashes are published
before screening. After screening, the selection decision and final protocol
hash are published before any confirmatory training. The exploratory screening
seed is reused only as one of three independently initialized baseline seeds in
confirmation; selection bias for that seed is disclosed in the final report.
