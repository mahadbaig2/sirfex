# PPO competence study

The Phase 3.1 gate is unchanged: zero physical violations, zero execution errors,
every nominal validation episode reaches its 200-step horizon, and mean absolute
level error (MAE) ≤ .10. The frozen validation set has 12 scenarios generated with
dataset seed 3101. Training, validation, nominal holdout and held-out stress
partitions are disjoint under the Phase 3.1 physical-key check. Only validation
was used to choose checkpoints; final holdouts were not evaluated here.

Run `20261025T205339_64b622c146` used the calibrated six-value observation,
`pump-normalized:1` policy action, `pump-survival-tracking:2` reward and original
`pump-terminate-on-violation:1` termination. Only action coordinates and reward
changed relative to the Phase 3.1 calibrated study, in predeclared sequential
ablations. PPO remained learning rate .0003, rollout 128, batch 32, four epochs,
gamma .99, GAE .95, entropy 0, value coefficient .5 and MLP `[64,64]` on CPU with
two Torch threads. The standard profile capped training at 51,200 steps per seed
and total wall time at 900 seconds. All checkpoints were evaluated, including
those after the first passing one.

| Seed | Steps | Validation MAE | Violations / 12 | Full horizon | Gate |
|---:|---:|---:|---:|---:|---|
| 3 | 10,240 | .053288 | 0 | 12/12 | PASS, selected |
| 3 | 25,600 | .036859 | 0 | 12/12 | PASS |
| 3 | 51,200 | .016863 | 0 | 12/12 | PASS |
| 17 | 10,240 | .078340 | 0 | 12/12 | PASS, selected |
| 17 | 25,600 | .036893 | 0 | 12/12 | PASS |
| 17 | 51,200 | .028521 | 0 | 12/12 | PASS |
| 41 | 10,240 | .114120 | 0 | 12/12 | FAIL, tracking |
| 41 | 25,600 | .076652 | 0 | 12/12 | PASS, selected |
| 41 | 51,200 | .020125 | 0 | 12/12 | PASS |

All nine checkpoints had zero execution errors and full horizons. Each gate row
contains 2,400 recorded physical actions. The first passing checkpoint per seed
was selected before search, without cherry-picking the best 51,200-step metric.
At those selected checkpoints, lower/upper saturation fractions were 0/0 for
seeds 3 and 17 and .000417/0 for seed 41; mean applied commands were .5432,
.5426 and .5379, respectively. Contrast the historical deterministic all-zero
commands and the action-only ablation's all-one commands. Full action statistics,
per-scenario outcomes and raw Gaussian diagnostics are persisted in the study.

| Selected seed | Checkpoint ID | SHA-256 |
|---:|---|---|
| 3 | `ppo_3_10240` | `859c9244e83fce5cf31600d9f7fae0038e053fc8b3dc1663218ed616257e70f9` |
| 17 | `ppo_17_10240` | `05b5fe45c1f46b7eb28699bd4a1dc693e9c11d56fba3376211c5985168a9d333` |
| 41 | `ppo_41_25600` | `48fe432e373aef9f64e55e0092f9b09ba95ef407bcc6d81f76a1f61432fd83f0` |

The 153,600 training steps plus validation and diagnostics took 634.48 seconds
wall and 839.73 CPU seconds. Reported peak process working set was 294,662,144
bytes (281 MiB); completed ZIP checkpoints were 143,779–145,188 bytes. Training
segments ranged roughly 199–273 steps/s, with validation about 3.5–4.2 seconds
per 12-scenario checkpoint. This is a measurement on the current local CPU,
not a general throughput guarantee.

The sequential controlled comparison was:

| Study | Action | Reward | Seed 3 at 10,240 / 25,600 | Interpretation |
|---|---|---|---|---|
| A, Phase 3.1 calibrated | Direct `[0,1]` | Legacy | 12/12 violations at both | Negative historical baseline |
| B, `20261025T204620_eb148a7c0a` | Normalized `[-1,1]` | Legacy | 12/12 violations at both; all commands one | Action change alone inadequate |
| C, `20261025T204951_e773d5feaa` | Normalized | Revised | 0/12 at both; MAE .053288 / .036859 | Promising pilot |
| Standard, `20261025T205339_64b622c146` | Normalized | Revised | Three independent seeds pass by 25,600 | Replicated nominal gate |

The first 10,240-step pilot took about 52 seconds of training at 196 steps/s;
the action-only 25,600-step pilot took 117.84 seconds total, and the revised
pilot used a comparable sequential budget. No optimizer search or final-holdout
tuning occurred. Because B and C differ in reward and the legacy run differs
in action coordinates, the evidence supports the combined revised configuration;
it does not identify a unique causal mechanism or prove general competence beyond
this modeled nominal validation distribution. The standard study's `partially_ready`
status means avoidable failure evidence was still pending when training finished.
The [Phase 3.2 report](PHASE_3_2_REPORT.md) combines both gates.

```powershell
sirfex train competence --config configs/training/pump_competence.json
sirfex train report 20261025T205339_64b622c146
python scripts/inspect_validation_policies.py 20261025T205339_64b622c146 --selected-only
python scripts/inspect_selected_policy_distributions.py 20261025T205339_64b622c146
```
