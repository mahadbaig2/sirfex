# Phase 3.6A policy regression diagnostics

The headless diagnostic run `20261027T191840_736f65993d` used only the 12 Phase 3.2 validation scenarios and 24 Phase 3.4 development stress scenarios. Their frozen hashes and all supporting checkpoint SHA-256 values appear in the [machine-readable evidence](evidence/phase_3_6_diagnostics.json). The Phase 3.4/3.5 final sets were not evaluated or sent to a model.

For each seed, deterministic baseline rollouts supplied every fifth observation. This yielded 1,239 shared observations for seed 3 and 849 for seed 17; the lower count reflects earlier termination in some seed-17 baseline stress trajectories. At each fixed six-input observation, the Phase 3.2 baseline, Phase 3.5 normal/5%/10% checkpoints, Phase 3.3 normal/robust checkpoints and the analytical controller produced physical pump commands. A fresh-reset hysteresis action was also recorded, with its missing prior command state explicitly noted. These are **action comparisons at fixed observations**, not simulated counterfactual trajectories. Independent development trajectory metrics come from the saved Phase 3.5 evaluations.

| Seed | Policy | Fixed-state mean physical command | Upper-bound saturation | Saved nominal MAE | Saved development stress MAE |
|---:|---|---:|---:|---:|---:|
| 3 | Phase 3.2 baseline | 0.669 | 0.0% | — | — |
| 3 | Phase 3.5 normal | 0.736 | 59.4% | 0.01773 | 0.11663 |
| 3 | Phase 3.5 target 5% | 0.807 | 71.6% | 0.02959 | 0.06539 |
| 3 | Phase 3.5 target 10% | 0.868 | 75.9% | 0.05010 | 0.05547 |
| 17 | Phase 3.2 baseline | 0.578 | 0.0% | — | — |
| 17 | Phase 3.5 normal | 0.619 | 48.1% | 0.02382 | 0.11523 |
| 17 | Phase 3.5 target 5% | 0.686 | 61.4% | 0.02036 | 0.07221 |
| 17 | Phase 3.5 target 10% | 0.687 | 53.7% | 0.02574 | 0.04651 |

Relative to the matched normal continuation on the **same baseline-observation sample**, seed-3 target-5 and target-10 actions rose by mean 0.0708 and 0.1314; neither decreased at any sampled state. Seed-17 mean signed changes were +0.0674 and +0.0676. On the nominal-validation subset, seed-3 mean physical command rose from 0.6192 for normal continuation to 0.7338 and 0.8075 for the target-5 and target-10 policies. Seed 17 also shifted up on nominal states, but target-5's independent nominal tracking improved. The action shift is therefore a measurable association, not a sufficient explanation of tracking regression.

The evidence also includes conditional summaries by target error, distance to each operating boundary, corrective authority, demand/effective-capacity ratio, initial level and development partition. Parameter L2 distances from the baseline are descriptive only. PPO optimizer randomness, sampled scenario order and changed state visitation were not separately manipulated; no causal root cause is established. There are no intermediate Phase 3.5 checkpoints, so no within-continuation learning trajectory is claimed.

The leading controlled hypothesis is that stress continuation can increase commands in ordinary nominal states and may overshoot the target for some starting policies. A second hypothesis is that reducing nominal practice contributes to forgetting; the current evidence does not isolate this factor. A bounded scheduled-exposure or nominal-rehearsal intervention could test these ideas on development data under unchanged physical and reward contracts. The agentic layer must treat these as hypotheses, not findings.

Run `sirfex research diagnose --config configs/research/pump_diagnostics.json` to create a new local diagnostic artifact. `sirfex research verify <run-id>` verifies its frozen protocol hash.
