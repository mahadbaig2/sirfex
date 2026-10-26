# Phase 3.3 matched robustness protocol

This protocol was fixed before opening final holdout outcomes. Its machine-readable
[preregistration](evidence/phase_3_3_preregistered.json) records the source
checkpoint hashes, exact physical scenario records, partition hashes, known
discovery witnesses, PPO configuration, curriculum, comparison policy, and
development-only cross-policy results.
The prepared study ID is `20261026T222151_846c8296a2`; protocol SHA-256 is
`159fd934853a4b40f2f54362f00fa4961f8daa4bf82b717a1cae18c21ed670ff`.

## Question and paired design

The primary question is whether continuing a competent pump PPO policy with 30%
SIRFEX-discovered avoidable stress reduces **held-out feasible-stress physical
violation frequency** compared with 100% normal continuation. The comparison is
failure-informed minus normal, paired within each preselected independent PPO
training seed. Seed 3 starts from `ppo_3_10240`, seed 17 from `ppo_17_10240`, and
seed 41 from `ppo_41_25600`. Their SHA-256 identities are fixed in the
preregistration and checked before training. No later 51,200-step checkpoint may
replace them.

Each seed forks the same saved policy and optimizer state into two independent
SB3 loads. Both arms use continuation seed `baseline_seed + 5000`, identical PPO
hyperparameters and **25,600 additional interaction steps**. The normal arm
samples 100% from the frozen 12-scenario nominal training pool; the robust arm
samples 70% normal and 30% stress. Each stress episode samples uniformly across
the three fixed Phase 3.2 discoveries. Actual episode and discovery exposure
counts are recorded. Sequential training avoids resource contention. A fixed
final checkpoint is evaluated; no best-checkpoint selection uses holdout results.

The six-input calibrated observation, normalized `[-1,1]` policy coordinate,
physical `[0,1]` pump command, `pump-survival-tracking:2` reward and
`pump-terminate-on-violation:1` termination are retained. The pump physical
equation and failure categories are unchanged. Torch runs on CPU with two threads,
one worker, a 153,600-step total continuation cap and a 2,400-second inclusive
wall deadline. Deadline interruptions retain partial evidence and cannot be
classified as a complete study.

## Frozen inputs and leakage rule

The 12 normal training, 12 competence validation, 12 nominal final holdout and
12 original held-out feasible-stress scenarios are loaded from the recorded
Phase 3.2 study, preserving their source hashes. A new final stress supplement
adds **48** feasible physical configurations generated with fixed seed `33103`
under the documented search bounds: initial level [.15,.85], demand [.35,.95],
capacity [.5,1.4], efficiency [.6,1], leakage [0,.12], target .5, dt .1,
operating limits [.05,.95] and horizon 200. It accepts only mathematically
finite-horizon viable configurations. This yields 60 held-out feasible-stress
scenarios, selected before any final policy outcome is inspected. A 60-case
frequency changes in units of 1/60; this is useful descriptive resolution but
does not multiply the number of independently trained controllers.

The existing physical-key leakage rule rejects configurations within absolute
1e-4 on every physical parameter other than horizon, and rejects seed overlap
across partitions. The three known discoveries are excluded from final stress.
A different random seed with identical pump physics does not count as an
independent stress example. The original final holdouts were unused in Phase 3.2;
their outcomes and those of the new 48 were sealed until six matched checkpoints
completed and their hashes verified.

The discovery evidence was reloaded and replayed against the original experiment
IDs. Both analytical and rule references remained safe over identical 200-step
scenarios. Before training, the three baseline PPO policies were measured on the
known set only: seed 3 failed 3/3, seed 17 failed 3/3, and seed 41 failed 0/3.
The curriculum still uses all three uniformly in both robust-arm seed pairs;
seed 41's known scenarios are challenges, not proven seed-41 failures.

## Evaluation and interpretation

After both arms complete for all three seeds, evaluate each original baseline,
normal continuation, and robust continuation on the **same** nominal final,
known stress, and held-out feasible stress sets. Evaluate the rule and calibrated
analytical references on the same sets. Deterministic PPO inference is used
throughout. The existing evaluation engine records physical violations,
magnitude, tracking, horizon, actions and time to first violation per scenario.
Representative experiment artifacts from each controller and set are replayed.

The frozen per-seed guardrails require nominal mean tracking error no more than
10% above the matched normal arm, no increase in nominal violation frequency,
held-out violation frequency or maximum held-out violation magnitude, and no
increase in execution failures. These are experimental policy choices, not safety
limits. Per-seed `improved` additionally requires an actual favorable held-out
frequency or maximum-magnitude signal; a known-stress gain alone cannot qualify.
The overall `improved` label is deliberately stricter: all three paired seeds
must lower the **primary** held-out violation frequency and pass guardrails.
Any complete guardrail failure gives `regressed`; incomplete or mixed evidence
is `inconclusive`. This rule was specified before final outcomes.

We report robust-minus-normal per-seed differences, mean, median, range,
direction counts and the existing paired training-seed percentile bootstrap.
With only three independent training seeds, that interval is exploratory and
cannot establish strong population-level significance. Repeated episodes under
the same learned controller are not independent policy replications. Known-stress
performance is reported separately from unseen physical generalization.

```powershell
sirfex train robustness-study --config configs/training/pump_phase_3_3.json --prepare-only
sirfex train robustness-study --study <prepared-study-id>
sirfex train report <study-id>
sirfex train verify <study-id>
```
