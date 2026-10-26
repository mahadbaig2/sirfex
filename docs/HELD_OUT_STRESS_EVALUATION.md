# Held-out feasible-stress evaluation

The Phase 3.3 primary evaluation set consists of 60 distinct pump configurations:
12 unused Phase 3.2 final stress scenarios and 48 newly generated feasible
scenarios. All have dt .1, horizon 200 and operating limits [.05,.95]. The new
generator used seed 33103 and accepted 48 of 71 proposed parameter combinations
after mathematical finite-horizon viability and physical/seed leakage checks.
The frozen primary-set SHA-256 is
`7ad4b202d5410ab0c5dab145982884649bcfc5c1bdbf4092768d1716d97776dc`.
The exact parameters and provenance are in the
[preregistered evidence](evidence/phase_3_3_preregistered.json).

No held-out trajectory was run before the study protocol and partition hashes
were fixed. The existing Phase 3.2 training and validation partitions, three
known stress discoveries, and both final sets are disjoint under the project's
physical-key and seed rule. The three curriculum discoveries form a separate
known-stress set; its outcomes measure learning on encountered conditions, not
generalization.

For every baseline, normal continuation, robust continuation, rule reference and
analytical reference, the existing `EvaluationRunner` measures one deterministic
episode per physical configuration. Each learned controller sees identical
scenarios in both arms. The reported physical metrics are violation frequency,
maximum and mean violation magnitude, tracking error, full-horizon frequency,
time to first violation, mean command and boundary saturation. All 60
per-scenario outcomes are persisted so that aggregate changes can be inspected.
The mean violation magnitude is the mean per-scenario maximum, not a sum of
reward penalties.

A different scenario seed on identical deterministic pump physics would not add
independent information. Sixty physical conditions provide descriptive variation
within a model, while the independent policy comparison has only three baseline
training seeds. The rule and analytical controllers are references with different
assumptions; the analytical controller observes calibrated physical parameters.
Neither one is a randomized policy-training replicate.

## Observed Phase 3.3 result

The original PPO baselines violated limits in 13/60, 31/60 and 0/60 physical
cases for seeds 3, 17 and 41. Both the equal-budget normal continuations and
the failure-informed continuations had **0/60 violations in every seed**,
0 maximum/mean violation magnitude, 60/60 full horizons and zero execution
failures. The robust-minus-normal primary difference was therefore 0/60 for
each seed. The 60-case endpoint cannot distinguish the arms after normal
continuation already reaches zero violations.

Held-out mean tracking error (normal → robust) was .0938 → .1257 for seed 3,
.0754 → .0683 for seed 17 and .0689 → .0601 for seed 41. These mixed tracking
changes are secondary and do not reverse the preregistered nominal tracking
regression. The rule and calibrated analytical references also had zero
violations, with mean held-out tracking .0684 and .0163, respectively.

Original baselines completed 47/60, 29/60 and 60/60 full horizons; the earliest
observed physical violation occurred at 1.9 seconds for seed 3 and 1.0 second
for seed 17. Both continuation arms completed 60/60 horizons for each seed, so
they have no time to first violation. Mean physical command (normal → robust)
was .7441 → .7677, .7474 → .7514 and .7489 → .7548. The fraction of held-out
steps at the upper physical command boundary was 5.48% → 10.44%, 6.19% →
6.77% and 8.21% → 8.73%. These command statistics describe behavior; they
are not additional preregistered success criteria.

The [full report](PHASE_3_3_REPORT.md) gives paired outcomes and limitations;
the [public evidence](evidence/phase_3_3_results.json) contains all 60
scenario-level records for each policy, physical parameters, aggregate metrics,
action statistics, checkpoint hashes and replay IDs.
