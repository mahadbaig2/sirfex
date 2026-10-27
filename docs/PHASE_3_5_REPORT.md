# Phase 3.5 — Exposure-Controlled Curriculum Validation

## Executive summary

Phase 3.5 resolved the exposure ambiguity in Phase 3.4 and completed a fixed, two-seed development screen. Every stress arm met the preregistered exposure gate and the 5% and 10% targets produced different realized stress interaction fractions. Both candidates improved development feasible-stress tracking, but neither preserved nominal tracking on **both** development seeds. The frozen outcome is **`no_eligible_curriculum`**. No confirmatory continuation, final holdout evaluation, regression classification or final replay was run. This is a development result, not evidence of independent robustness superiority.

The machine-readable [screening evidence](evidence/phase_3_5_results.json) includes all six arms, exposure ledgers, checkpoint hashes and eligibility checks. Binary checkpoints and full episode traces remain local.

## Prior limitation and instrumentation

In Phase 3.4, each 5% and 10% Bernoulli curriculum arm sampled only one stress episode in 5,120 PPO interactions. Configured episode probabilities therefore did not establish distinct physical exposure. Phase 3.5 tracks every actual physical transition by nominal/stress mode, scenario and discovery, as well as completed and partial episodes. The single-environment invariant reconciled in all six runs:

`nominal_interactions + stress_interactions = actual_environment_interactions = PPO additional timesteps = 51,200`.

The new `pump-curriculum-exposure:1` schema, reset seed behavior and episode-boundary scheduler are described in [Curriculum Exposure Accounting](CURRICULUM_EXPOSURE_ACCOUNTING.md). Historical Phase 3.3/3.4 artifacts were not rewritten or backfilled.

## Frozen design, calibration and provenance

The [public protocol](evidence/phase_3_5_preregistered.json) was committed before any candidate outcome evaluation. Its content hash is `b4617b9342f95534464b63a1c14f1253fb5acf284f28cccccdc1884183b9bfe4`. The study ID is `20261027T143749_d88d2dd37f`. The main protocol was published at Git commit `7dd497b` after accounting was published at `4d19af0`.

The three arms were a normal-only control and expanded seven-discovery curricula targeting 5% or 10% of **interactions** with stress episodes. These are episode-boundary interaction targets, not episode-selection probabilities. The 4,096-step neutral pilot used fixed normalized policy action 0.5 (physical command 0.75), with no policy performance evaluation. It measured 229 stress interactions and 10 completed stress episodes at the 5% target and 413 interactions and 16 episodes at the 10% target. Neutral environment throughput was 8,555 and 8,628 steps/s respectively. PPO throughput in the actual screen was about 480–510 steps/s including optimization. Pilot and study hardware remained CPU-only with two Torch threads.

The frozen adequacy gate required, **per stress arm**, at least 10 completed stress episodes, 1,000 stress interactions, three discoveries, valid scenario assignments and exact transition/PPO reconciliation. Development seeds were 3 and 17; continuation seeds were 5003 and 5017. Each arm received 51,200 additional PPO transitions from a verified Phase 3.2 starting checkpoint and optimizer state. Normal control and stress arms used the same action, reward, termination, observation and PPO contracts. The screening total was 307,200 interactions. The wall cap was 5,400 seconds; the fixed confirmatory plan, if selected, was three matched pairs at 51,200 additional interactions per arm.

The source Phase 3.4 study ended at `no_eligible_curriculum` and has no final protocol, final evaluation, continuation runs or paired final analysis. Reused final feasible-stress and nominal partitions retain hashes `c992525e1f9e382432dfe3fceebef0a0e9339a16432a4767fb12455687816226` and `da86e081c624fe65009d89505c48c7760d0a2692070b6c7e7d463dfa886f1f94`. Physical and seed overlap checks among historical training, discoveries, development and final sets passed; Phase 3.4's sampled training scenario IDs did not include final IDs. Source checkpoint hashes for seeds 3, 17 and 41 and the discovery-pool hash are in the [protocol](evidence/phase_3_5_preregistered.json). The public scenario parameters alone did not establish blindness; the saved outcome-file and training-provenance checks were also performed. The final partitions remain unopened after this study.

## Actual development exposure and outcomes

| Baseline seed | Arm | Stress interactions | Realized fraction | Completed stress episodes | Partial stress interactions | Discoveries | Nominal MAE | Development stress MAE | Stress violation frequency |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | Normal | 0 | 0.00% | 0 | 0 | 0 | 0.017726 | 0.116625 | 0 |
| 3 | Target 5% | 2,635 | 5.15% | 37 | 80 | 7 | 0.029588 | 0.065390 | 0 |
| 3 | Target 10% | 5,165 | 10.09% | 47 | 0 | 7 | 0.050098 | 0.055472 | 0 |
| 17 | Normal | 0 | 0.00% | 0 | 0 | 0 | 0.023817 | 0.115226 | 0 |
| 17 | Target 5% | 2,672 | 5.22% | 30 | 0 | 7 | 0.020360 | 0.072207 | 0 |
| 17 | Target 10% | 5,230 | 10.21% | 64 | 0 | 7 | 0.025739 | 0.046509 | 0 |

All six arms completed exactly 51,200 interactions and passed checkpoint and exposure reconciliation. All four stress arms exceeded every exposure threshold. The [screening figure](evidence/phase_3_5_screening.png) shows target versus realized interaction fraction, per-discovery exposure, nominal and stress MAE, violation-frequency floor, difficulty-stratum tracking and action command statistics. Numerical source data remain in the [public evidence](evidence/phase_3_5_results.json).

## Eligibility and interpretation

The frozen nominal MAE guardrail allowed at most 10% degradation against the matched normal continuation on **each** development seed. Seed 3 degraded by **66.92%** for the 5% target and **182.63%** for the 10% target. Seed 17 improved nominal MAE by 14.51% at 5% and degraded by 8.07% at 10%, both within the guardrail. Both candidates improved development stress MAE on both seeds by more than the frozen 0.005 threshold, and neither added development nominal violations, stress violations or execution failures. Their per-seed exposure and equal-budget checks passed. Since both candidates failed seed 3's nominal guardrail, **neither qualified**, and `selected_candidate_id` is `null`.

All development stress violation frequencies were zero. This endpoint is saturated on these scenarios; the prespecified development MAE gain supplied the stress-gain check. The observed tracking association does not isolate a causal effect of intensity: seed, sampled scenario ordering and stochastic PPO optimization may also affect the result. The study establishes that exposure was genuinely different and adequate, and that the frozen nominal preservation rule still failed for both candidates. It does not establish a final held-out effect.

No confirmation pairs were trained. No final nominal or feasible-stress policy outcomes, paired seed intervals, regression classification, rule/analytical final evaluations or final replay verification exist for this study. Three-seed statistical intervals would have been exploratory if confirmation had opened; there are no final deltas to analyze. The known-discovery performance is not substituted for the independent final benchmark.

## Verification, resources and limitations

`sirfex train verify 20261027T143749_d88d2dd37f` verified the frozen protocol hash and all six stored checkpoint SHA-256 hashes. The full local suite finished with **122 passed, four existing warnings**. Six PPO continuations took 626.57 seconds of cumulative training time, approximately 104.43 seconds per arm. The final process RSS was 299,499,520 bytes and peak working set was 300,421,120 bytes, about 286 MiB. Full artifacts are in the ignored local `artifacts/training/<study-id>/` directory; the public export omits checkpoint binaries, local paths and long action traces. Reproducing PPO training from public JSON alone is not possible without the referenced source checkpoints and runtime.

The neutral pilot's policy action is not a model-driven controller; it calibrates reset and transition behavior only. Development selection has only two independent trained baseline seeds. The final partitions were preserved but did not produce outcomes. No claim of superior physical safety or nominal-preserving robustness follows from the development stress MAE gains. A future milestone should investigate why stress-targeted PPO continuation moves seed-3 nominal behavior, using new development data and a new frozen method before another independent final study. The Phase 3.4/3.5 final sets remain available for a single prospective confirmatory evaluation under a separately frozen, eligible protocol.
