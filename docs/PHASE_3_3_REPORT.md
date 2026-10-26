# SIRFEX Phase 3.3 report — matched failure-informed robustness study

Measured 26–27 October 2026 on local Windows 11, Python 3.12 and CPU. This is a
controlled experiment in the simplified, deterministic single-tank pump model.
The [protocol](MATCHED_ROBUSTNESS_STUDY.md) and
[machine-readable preregistration](evidence/phase_3_3_preregistered.json) were
published before the final holdouts were opened. The complete compact
[result evidence](evidence/phase_3_3_results.json) preserves per-scenario
outcomes, hashes, actual training exposure, paired comparisons and replay IDs.

## 1. Executive summary and primary hypothesis

The primary question was whether a competent PPO policy continued with 30%
replay-verified, physically avoidable failures would violate physical pump
limits less often on independent feasible stress than the same policy continued
for an equal budget on normal scenarios alone. The nominal holdout guardrails
guard against a stress gain that sacrifices normal operation. The independent
unit is a baseline training seed; this study has three matched pairs.

## 2. Experimental design and frozen inputs

The selected Phase 3.2 PPO checkpoints were the preregistered competent policies,
not the latest available checkpoint. Each seed's normal and robust arms loaded
the same saved SB3 policy and optimizer state independently, used continuation
seed `baseline seed + 5000`, the same PPO hyperparameters and exactly 25,600
additional interactions. Normal episodes sampled only from the 12 frozen
training cases; the robust arm sampled 70% normal and 30% uniformly over three
known discoveries at episode reset. Training was sequential and the fixed final
checkpoint of each arm was evaluated without holdout-based model selection.

| Baseline seed | Selected checkpoint | Selected steps | SHA-256 |
|---:|---|---:|---|
| 3 | `ppo_3_10240` | 10,240 | `859c9244e83fce5cf31600d9f7fae0038e053fc8b3dc1663218ed616257e70f9` |
| 17 | `ppo_17_10240` | 10,240 | `05b5fe45c1f46b7eb28699bd4a1dc693e9c11d56fba3376211c5985168a9d333` |
| 41 | `ppo_41_25600` | 25,600 | `48fe432e373aef9f64e55e0092f9b09ba95ef407bcc6d81f76a1f61432fd83f0` |

All three original checkpoint files and metadata were verified against these
identities. The baseline selected model and optimizer fingerprints were checked
against both newly loaded arms before training. The continuation retained the
six-input calibrated observation, normalized `[-1,1]` policy action mapped to
physical `[0,1]`, `pump-survival-tracking:2` reward and
`pump-terminate-on-violation:1`. The physical dynamics, constraint definitions
and failure categories did not change. Legacy Phase 3.1 direct-action/reward
continuation was not used.

The reward is `1 - 4 * tracking_error² - .01 * physical_command² -
10 * violation_magnitude - indicator(terminated)`. This is a training reward;
the primary result is measured from physical constraint evidence, not reward.

## 3. Partitions, leakage and discovery eligibility

The frozen Phase 3.2 nominal training, competence validation, nominal final and
original stress final partitions each contain 12 scenarios. A fixed-seed
generator (`33103`) accepted 48 physically distinct finite-horizon feasible
stress configurations from 71 proposed parameter combinations, making 60 final
held-out stress scenarios. The exact records and SHA-256 identities are in the
preregistration. The leakage check excludes any physical configuration within
absolute 1e-4 on every physical parameter across train, validation, known and
final groups, and excludes seed overlap. Re-seeding identical deterministic
physics is not an independent held-out case. Final scenarios were not sampled
for training, selection or diagnosis.

The known-stress pool comes from Phase 3.2 search
`20261026T090912_05c97a1dec`:

| Discovery | Original seed-3 failure step |
|---|---:|
| `disc_05c97a1dec_0000` | 32 |
| `disc_05c97a1dec_0002` | 22 |
| `disc_05c97a1dec_0001` | 29 |

Each original failing experiment replayed exactly. Rule and analytical
references completed the identical 200-step physical scenario without a
constraint violation, and witness command sequences were persisted. Pretraining
known-set checks found seed 3 failed 3/3, seed 17 failed 3/3 and seed 41 failed
0/3. The seed-41 known cases are challenges, not proven failures of that seed.
Known-set changes are reported separately from independent held-out changes.

## 4. Training configuration and budgets

The preregistered run is `20261026T222151_846c8296a2`, with protocol SHA-256
`159fd934853a4b40f2f54362f00fa4961f8daa4bf82b717a1cae18c21ed670ff`.
The six-arm plan caps total additional interactions at 153,600 and sets a
2,400-second inclusive wall deadline. Torch uses CPU with two threads and one
environment worker. PPO used learning rate 3e-4, 128 steps per rollout, batch
size 32, four epochs, gamma .99, GAE lambda .95, entropy coefficient 0,
value coefficient .5 and two 64-unit hidden layers. The exact parameters are
recorded in each arm's `ppo_config` and in the public evidence.
`continuation_runs.json` records actual
steps, seconds, sampled nominal and stress episodes, per-discovery exposure,
starting fingerprints, and final checkpoint ID and SHA-256. All six checkpoint
hashes must verify before any final evaluation is allowed.

## 5. Final-set evaluation

Every original baseline, normal continuation, robust continuation, rule
reference and calibrated analytical reference runs on the identical 12 nominal
final, three known-stress and 60 held-out feasible-stress configurations. PPO
inference is deterministic. The existing `EvaluationRunner` records physical
violation frequency and magnitude, tracking, horizon completion, command and
boundary saturation, and execution failures. One episode per physical case is
used; scenario diversity is descriptive, while only three independent policy
training seeds support the paired inference. The analytical reference observes
calibrated physical parameters and is not an equally trained PPO control.

## 6. Prespecified paired comparison and limits

Per-seed differences are robust minus normal. The primary endpoint is held-out
physical violation frequency; a known-discovery gain alone cannot qualify.
Nominal mean tracking error may not exceed the matched normal value by more
than 10%; nominal/held-out violation frequency, maximum held-out violation
magnitude and execution failures may not increase. Overall `improved` requires
all three seeds to reduce primary frequency while satisfying every guardrail.
Complete guardrail failure is `regressed`; mixed or incomplete evidence is
`inconclusive`. The paired percentile bootstrap over three training seeds is
exploratory and does not create industrial-safety or population guarantees.

The pump is a simplified simulator, the stress generator and discoveries share
its model assumptions, and three baseline seeds are insufficient for a strong
population-level inference. The 60 physically distinct holdouts do not become
60 independent trained policies. Reward changes, algorithm changes and
additional policy tuning are outside this frozen comparison.

## 7. Actual continuation exposures and checkpoint identities

All six arms completed exactly 25,600 additional steps, for 153,600 total.
Paired starting policy and optimizer fingerprints matched. Actual episode
counts differ because a stress episode can terminate earlier than a normal
episode; equality of interaction steps and PPO configuration is the matched
budget. The six continuations took 531.3 seconds of aggregate training time.

| Seed | Arm | Final checkpoint SHA-256 | Normal episodes | Stress episodes | Training seconds |
|---:|---|---|---:|---:|---:|
| 3 | normal | `cca14de90a83fa3205efb586e9c34f0b461edf12afb8cfa90e74d05659f75f27` | 129 | 0 | 65.1 |
| 3 | robust | `4b2dddc8fd2aa76b49fdffa40d3c7a6848d5a49b0a6b2949d82b53e484a0b2d1` | 94 | 49 | 70.9 |
| 17 | normal | `e0889617537e2278872303626f176cfad42dc4fbc4f1b6907d7c0764f6e89f54` | 130 | 0 | 92.4 |
| 17 | robust | `21f9ae78d4176d3a2817b22d641009a6a3204734c5d5a407b9b20ca59500477c` | 103 | 43 | 99.4 |
| 41 | normal | `d46f64ac798d83ff894eec784e3995e99d7ff80e795f860b2ca0eb4c07230fa8` | 129 | 0 | 103.5 |
| 41 | robust | `a04b792f77686627775bf5d421b7e4b2b92b70ed66c7e1675f7460644bd3c54f` | 85 | 51 | 100.0 |

Known-discovery episode exposures in ID order `0000`, `0001`, `0002` were
`14/13/22` for seed 3, `13/17/13` for seed 17 and `14/20/17` for seed 41.
Each normal arm had zero discovery exposure. The requested stress probability
is a reset-level sampling probability, not a guarantee of exact episode counts.

## 8. Nominal, known and held-out outcomes

Frequency is the count of physical violations divided by the number of tested
configurations. Tracking is mean absolute level error; maximum magnitude is the
largest amount outside a physical operating limit. All 12 nominal and all three
known scenarios per controller ran the full horizon except the original
seed-3/17 baselines on known stress. All six continuations completed every
episode on every set without execution failure.

| Seed | Policy | Nominal violations / 12 | Nominal tracking | Known violations / 3 | Known tracking | Held-out violations / 60 | Held-out tracking |
|---:|---|---:|---:|---:|---:|---:|
| 3 | original baseline | 0 | .0699 | 3 | .3399 | 13 | .2355 |
| 3 | normal continuation | 0 | .0331 | 0 | .1880 | 0 | .0938 |
| 3 | robust continuation | 0 | .1113 | 0 | .0715 | 0 | .1257 |
| 17 | original baseline | 0 | .1037 | 3 | .3255 | 31 | .2490 |
| 17 | normal continuation | 0 | .0304 | 0 | .1593 | 0 | .0754 |
| 17 | robust continuation | 0 | .0428 | 0 | .1348 | 0 | .0683 |
| 41 | original baseline | 0 | .1076 | 0 | .3429 | 0 | .2146 |
| 41 | normal continuation | 0 | .0256 | 0 | .1222 | 0 | .0689 |
| 41 | robust continuation | 0 | .0528 | 0 | .1121 | 0 | .0601 |

The original baselines violated the held-out physical limits in 13/60, 31/60
and 0/60 cases for seeds 3, 17 and 41. **Both** continuation arms reduced
those counts to zero for all three seeds. The failure-informed arm therefore
has **zero incremental held-out violation-frequency improvement over equal
normal continuation**; both arms also have zero maximum and mean held-out
violation magnitude. Known-discovery violations likewise dropped to zero in
both arms for all seeds. Failure-informed training improved known-set tracking
relative to normal continuation, but that is adaptation to encountered cases
and does not establish generalization. It worsened nominal tracking on every
seed. The rule and calibrated analytical references had zero violations and
full horizons on all three sets; their held-out tracking errors were .0684
and .0163, respectively.

The [held-out evaluation note](HELD_OUT_STRESS_EVALUATION.md) documents the
physical set and the [public result JSON](evidence/phase_3_3_results.json)
contains every scenario result, including first-violation time and command
statistics. There were no execution failures in the final evaluation.

## 9. Paired differences, uncertainty and regression

The differences below are robust minus normal, so a positive nominal tracking
difference is harmful. The nominal relative change is measured against the
normal continuation's tracking error; the preregistered maximum is +10%.

| Seed | Held-out violation-frequency difference | Held-out max-magnitude difference | Held-out tracking difference | Nominal tracking difference | Nominal relative change | Guardrail outcome |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 0/60 | 0 | +.03190 | +.07816 | +235.93% | regressed |
| 17 | 0/60 | 0 | −.00710 | +.01238 | +40.76% | regressed |
| 41 | 0/60 | 0 | −.00876 | +.02720 | +106.43% | regressed |

The three paired primary differences have mean 0, median 0, range [0,0], and
direction counts 0 improved / 3 unchanged / 0 worsened. The preregistered
5,000-draw paired training-seed bootstrap gives exploratory 95% percentile
interval [0,0] for both held-out frequency and maximum magnitude differences.
This degenerate interval records the three observed zeros; it does **not**
prove future policies equivalent or make the study adequately powered for
small physical risk differences. The smallest nonzero frequency increment on
60 cases is 1/60. The known-set tracking differences were −.11645, −.02449
and −.01007. Held-out tracking changed in opposite directions across seeds.

All three seeds failed the frozen nominal-tracking guardrail, while nominal
violation frequency, held-out violation frequency, held-out maximum magnitude
and execution-failure guardrails passed. The per-seed and overall classification
is therefore **`regressed`**. The specific regression is nominal tracking
accuracy, not a new physical constraint violation. The normal-only arm
already removed the measurable held-out violations at this study budget, so
the primary frequency endpoint has a ceiling/floor problem for detecting any
further benefit of stress sampling.

## 10. Replay, resources and verification

All 33 representative experiments—each original/continued PPO policy and both
references on each of the three sets—replayed with `exact_match`; none failed.
The six final continuation checkpoint SHA-256 hashes in section 7 were verified
again by `sirfex train verify`. Total study wall time, including preflight,
training, final evaluation, replay and persistence, was 726.5 seconds under the
2,400-second limit. Aggregate training time was 531.3 seconds; the completed
study artifact directory occupied 7,834,511 bytes. The end-of-run process RSS
and recorded working-set value were both 302,682,112 bytes. This is one
process-level measurement, not a sampled peak across the entire run; no GPU
was used.

Local verification passed: 105 pytest tests (four existing dependency/test
warnings), Ruff lint and format checks, `pip check`, and six checkpoint
integrity checks. The public CI training lane runs the matched-study
integration checks. The pump adapter and Phase 3.2 report are unchanged.

## 11. Scientific interpretation, defects and next milestone

The study rejects a beneficial incremental robustness claim for this protocol.
Normal continuation alone made all three policies safe on the frozen 60-case
feasible-stress set, while the 30% stress curriculum worsened normal tracking
on all three seeds beyond the prespecified tolerance. That is an observed
simulation result, not evidence that failure-informed training is generally
harmful or that either controller is safe on real plants. The analytical
reference receives calibrated parameters, and the held-out generator samples
only within one simplified physical model. Three independent baseline seeds
limit uncertainty assessment; the zero-frequency endpoint is uninformative
about smaller differences or different stress families. Full trajectories and
SB3 checkpoint binaries remain in ignored local artifacts; the public export
includes compact per-scenario and provenance data.

No known implementation defect invalidated the completed study. The measured
failure is a nominal-tracking policy regression under the frozen comparison.
The next research milestone should preregister a separate experiment with a
more discriminating independent stress family or severity range and a
curriculum that explicitly constrains nominal tracking. It should add more
independent policy-training seeds before making a population claim. The
present six checkpoints and final holdout outcomes must remain historical
evidence; any such new study needs newly frozen partitions and selection rules.

## 12. Acceptance record

| Engineering check | Evidence |
|---|---|
| Existing phases and unchanged pump dynamics | 105 tests passed; pump adapter diff and Phase 3.2 report diff are empty |
| Original baselines and Phase 3.2 artifacts preserved | Three preregistered starting SHA-256 hashes verified; new study has its own ID and directory |
| Action, reward, termination and observation contracts | Versioned identities checked at preflight and saved on each arm |
| Identical branch state and matched budgets | Starting model/optimizer fingerprints equal within seed; six arms each added 25,600 steps with identical PPO settings |
| Valid curriculum and leakage-free final sets | Three replay-backed feasible discoveries; frozen physical/seed leakage check; 12 nominal plus 60 held-out stress cases |
| Complete provenance and checkpoints | Six final hashes verified; config, partitions, exposure counts, evaluation and regression persisted |
| Same final scenarios and paired comparison | Shared set hashes for all controllers; per-seed robust-minus-normal results in public evidence |
| Representative reproducibility and CPU limits | 33 exact replays; 726.5 seconds within 2,400-second cap; CPU-only |
| Test and dependency isolation | 105 pytest tests, Ruff, `pip check` and optional training lane passed locally |

| Scientific check | Observation |
|---|---|
| Three independent matched pairs | All six complete |
| Nominal final holdout | 12 cases per controller, all completed |
| Known stress | Three cases per controller; no post-continuation violations in either arm |
| Independent feasible-stress final set | 60 cases per controller; zero post-continuation violations in either arm |
| Incremental benefit and nominal regression | Primary difference zero in each pair; nominal tracking worsened beyond tolerance in every pair |
| Variability, uncertainty and honest classification | Individual pairs, [0,0] exploratory primary bootstrap, overall `regressed` with limits stated |
