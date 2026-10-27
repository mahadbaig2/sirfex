# SIRFEX Phase 3.4 report — discriminating stress and nominal preservation

Measured 27 October 2026 on local Windows 11/Python 3.12 with CPU-only PPO.
The [benchmark and screening protocol](evidence/phase_3_4_preregistered.json)
was pushed before development screening. The [completed machine-readable
evidence](evidence/phase_3_4_results.json) records scenario parameters,
feasibility, hashes, search witnesses, actual exposure and per-scenario
development outcomes. Historical [Phase 3.3 results](PHASE_3_3_REPORT.md) were
not changed.

## Executive result

**Workflow outcome: `no_eligible_curriculum`.** The two original-pool 5%/10%
candidates stayed within the frozen nominal MAE guardrail but did not improve
development stress. The expanded-pool 5%/10% candidates reduced development
stress MAE by .00872 but raised nominal MAE by 74.37%, beyond the fixed 10%
limit. The study therefore stopped at selection. No matched confirmatory
continuation was run; the new final nominal and feasible-stress holdouts were
not evaluated. There is **no Phase 3.4 held-out robustness-improvement claim**.

This result is a measured screening rejection, not a controller regression
classification from final independent pairs. The final sets remain eligible
for a genuinely new preregistered study only if its selection and analysis
rules are frozen without using these sealed outcomes.

## 1. Historical Phase 3.3 nominal-regression diagnosis

The 30%-stress Phase 3.3 arms sampled 49, 43 and 51 stress episodes for seeds
3, 17 and 41. Their nominal MAE changed from `.03313 → .11128`,
`.03038 → .04276` and `.02556 → .05276` against matched normal continuation.
Nominal mean physical pump command changed `.56109 → .57182`,
`.56231 → .56329` and `.56141 → .56603`. These are measured changes, not an
identified causal mechanism. Nominal upper-bound action saturation was
`0 → .00417`, `0 → .00042` and `.00333 → .00292` as fractions of actions;
the saturation direction was not uniform. Tracking worsened in both lower-
and upper-demand halves and both lower- and upper-initial-level halves in all
three seeds. Known-discovery tracking improved by `.11645`, `.02449` and
`.01007`, while held-out tracking changed `+.03190`, `−.00710`, `−.00876`.

The saved aggregates support the association between changed policy behavior,
known-case adaptation and nominal regression. They do not isolate the 30%
ratio from repeated exposure to three scenarios or their severity. The
stress episode counts (49, 43, 51) do not order the nominal deterioration
(235.93%, 40.76%, 106.43%) monotonically across these three seeds. This small
descriptive comparison cannot establish or exclude an exposure relationship.
Per-step stress-versus-normal training interactions and near-target within-episode
tracking distributions were not recorded in Phase 3.3; they cannot be inferred
from episode counts. The full historical comparison, including action standard
deviation, lower saturation, error ranges, conditional means, horizon and
violation frequencies, is in `diagnostics.json` in the public evidence.

## 2. Stress benchmark construction and feasibility

The new controller-independent generator uses fixed seed `3401`, four
predeclared target-authority/initial-boundary strata and exact finite-horizon
reachability. See [benchmark design](STRESS_BENCHMARK_DESIGN.md). It accepted
24 development stress configurations (six per stratum), 64 sealed final stress
configurations (16 per stratum), and 12 sealed final nominal configurations.

| Set / stratum | Accepted | Proposals | Invalid | Proven infeasible | Mean authority ratio | Mean initial safe-bound margin | Finite-horizon only* |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development nominal-like | 6 | 6 | 0 | 0 | .5997 | .3463 | 0 |
| Development moderate | 6 | 8 | 2 | 0 | .8113 | .3481 | 0 |
| Development high | 6 | 10 | 4 | 0 | .9035 | .3388 | 0 |
| Development near-boundary | 6 | 14 | 8 | 0 | .9835 | .1092 | 2 |
| Final nominal-like stress | 16 | 16 | 0 | 0 | .6290 | .3434 | 0 |
| Final moderate | 16 | 18 | 2 | 0 | .7845 | .3207 | 0 |
| Final high | 16 | 22 | 6 | 0 | .9467 | .3473 | 0 |
| Final near-boundary | 16 | 31 | 14 | 1 | .9791 | .1025 | 4 |
| Separate final nominal | 12 | 12 | 0 | 0 | .6333 | .3650 | 0 |

*`bounded_operation_feasible`: a safe finite-horizon trajectory exists while
target equilibrium is not established. The remaining accepted cases are
`target_feasible`; none was accepted with undetermined viability. Invalid
proposals were rejected by the existing pump validator. No accepted near
duplicate or shared seed was found against historical training, validation,
discoveries, opened Phase 3.3 cases, or another new partition. The physical
and seed leakage check passed; exact records and SHA-256 partition hashes are
in the preregistration.

The Phase 3.3 held-out set had mean target-authority ratio `.7613` across 60
physical cases. The new development stress mean was `.8245`, and sealed final
stress mean `.8348`; this is a physical-distribution comparison, not a final
controller result. The designated original seed-3 baseline violated 6/24
development stress cases, concentrated at high (1/6) and near-boundary (5/6)
difficulty. Its development stress MAE was `.26539`; rule and analytical
references had no physical violations on these 24 cases. This demonstrates
that the *development* benchmark spans controller weaknesses. Whether the
sealed final benchmark discriminates confirmatory continuations remains
unmeasured because selection stopped.

Frozen SHA-256 identities: development stress
`08c845a1b56de5dce5e77dbc36d6d1d4f8145f2723acbc99c67164dfebf00ff5`,
final stress `c992525e1f9e382432dfe3fceebef0a0e9339a16432a4767fb12455687816226`,
final nominal `da86e081c624fe65009d89505c48c7760d0a2692070b6c7e7d463dfa886f1f94`.
The preregistered protocol SHA-256 is
`552b18608a625c2e1041de8843b11fb9b4966db03c8a0a034d161d3619af27f0`.

## 3. New avoidable-failure search and curriculum pool

The existing feasibility-aware random search ran 40 proposals against the
fixed competent seed-3 baseline under search ID
`20261027T135757_2ba26db7df`. It recorded 26 valid evaluated feasible
candidates, 13 rejected candidates and one invalid candidate. Four distinct
physical violations became new avoidable discoveries. Both rule and analytical
references completed each identical scenario safely; eight reference
evaluations and 16 witness/failure replay checks are recorded. The four new
discovery IDs are `disc_2ba26db7df_0000` through `_0003`. The expanded pool
contains these four plus the three unchanged Phase 3.2 discoveries. Search
excluded all new final and development benchmark configurations and historical
partitions. No final outcome contributed to discovery selection. The
[separate search evidence](evidence/phase_3_4_search.json) includes all 40
proposal dispositions, exact replay statuses, supporting experiment IDs and
witness metadata; full command traces remain in the local experiment store.

| New discovery suffix | Initial level | Demand | Effective max inflow | Leak | Feasibility |
|---|---:|---:|---:|---:|---|
| `_0000` | .5454 | .4662 | .4932 | .0640 | bounded-operation feasible |
| `_0001` | .2625 | .5794 | .5950 | .1189 | bounded-operation feasible |
| `_0002` | .2595 | .5997 | .6874 | .0400 | target feasible |
| `_0003` | .1730 | .8140 | .8987 | .0373 | target feasible |

The four physical configurations differ in demand, initial level, leakage and
available inflow. Two have no target-equilibrium guarantee but are proven
safe over the finite horizon by the feasibility analyzer and full-horizon
reference witnesses. The search did not produce a balanced set of every
conceivable stress category; no missing category was fabricated.

## 4. Frozen development-only curriculum screen

The standard profile used a seed-3 competent starting checkpoint, independently
reloaded PPO policy and optimizer state, identical PPO hyperparameters and
continuation seed, and **5,120 additional interactions per arm**. The five
arms totaled 25,600 screening interactions. The 12-case Phase 3.2 validation
set and new 24-case development stress set were the only screening evaluators.
The 30% Phase 3.3 result was a historical comparator and was not retrained.
The [screening rule](CURRICULUM_SCREENING.md) required nominal MAE no more than
110% of the normal control, no additional nominal or development-stress
violation or execution failure, and a development-stress violation reduction
or MAE reduction of at least `.005`.

| Candidate | Pool | Stress probability | Actual nominal / stress episodes | Nominal MAE | Nominal change vs normal | Development stress violations / 24 | Development stress MAE | Stress MAE change vs normal | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Normal control | none | 0% | 26 / 0 | .066287 | — | 0 | .183008 | — | control |
| Original 5% | 3 discoveries | 5% | 26 / 1 | .067069 | +1.18% | 0 | .187306 | +.004298 | no stress gain |
| Original 10% | 3 discoveries | 10% | 26 / 1 | .067069 | +1.18% | 0 | .187306 | +.004298 | no stress gain |
| Expanded 5% | 7 discoveries | 5% | 26 / 1 | .115585 | +74.37% | 0 | .174291 | −.008717 | nominal guardrail failed |
| Expanded 10% | 7 discoveries | 10% | 26 / 1 | .115585 | +74.37% | 0 | .174291 | −.008717 | nominal guardrail failed |

All five arms had zero nominal development violations and zero execution
errors. The original-pool arms satisfied nominal guards but did not improve
stress; the expanded-pool arms met the stress-gain criterion but failed nominal
preservation. The frozen selector recorded `no_eligible_curriculum`, with all
checks and rejection reasons in `selected_curriculum.json`.

**Screening limitation:** at 5,120 steps and these low reset probabilities,
each stress arm sampled only **one stress episode**. The 5% and 10% arms in a
given pool realized identical exposures and numerical outcomes under the fixed
seed. Thus this screen cannot estimate an intensity-response curve, and the
nominal/stress change for the expanded pool is especially sensitive to one
training trajectory. The protocol was not extended or tuned after this was
observed. It supports the prespecified rejection decision, not a general
claim that 5% and 10% stress training are equivalent or harmful.

The five saved screening checkpoint hashes were reverified by `sirfex train
verify`. They and per-scenario development metrics are in the evidence JSON.
The hashes are:

| Checkpoint ID | SHA-256 | Training seconds |
|---|---|---:|
| `screen_normal` | `5ba15c5a1436bc378414f3687724a175c5fa0dd8dbc4d5d90cd752ccf5d6d9d8` | 22.2 |
| `screen_original_5` | `27695da3aac9da7971b35aa7f177f28a2097594155c7a1f4d68dd31b8ded832b` | 17.8 |
| `screen_original_10` | `23813c36554c52e64bccde7255955d6581ec2fdf431183db87b66c7224bca739` | 21.6 |
| `screen_expanded_5` | `a4bb9a207fc3176942d9ed505bfc01d5694acaeea54c5f579ea6cc1fad6b3663` | 17.8 |
| `screen_expanded_10` | `bdbc8120f59464a23237ef54f7cb0f92dad47afa0b7e4c52ab2f7ad66e58c01a` | 15.1 |

There was no selected curriculum, no final protocol or confirmatory
checkpoint, and no final evaluation. Accordingly there are **no final nominal
metrics, held-out stress metrics, three-seed paired differences, bootstrap
intervals or controller regression classifications for Phase 3.4**. The
historical `regressed` label belongs to Phase 3.3 and is not repurposed here.

## 5. Replay, resources and verification

The new search's four discoveries have replay-verified failing experiments and
constructive safe reference witnesses. The Phase 3.4 development screen saved
five hash-verified PPO checkpoints; no final representative experiment replay
was performed because the confirmatory stage was not entered. Existing Phase
3.3 replay and evidence remain intact.

Preparation took 69.1 seconds, including the new search's 26.3 seconds with
references. Screening took 150.3 seconds, of which 94.5 seconds was aggregate
PPO training; the rest includes evaluation, checkpointing and persistence.
The screen process recorded 297,701,376 bytes RSS and 298,631,168 bytes
peak working set at completion; the local study artifact directory occupied
3,060,269 bytes. The separate search process recorded
306,581,504 bytes RSS and 307,773,440 bytes peak working set. These are
process-level observations, not a continuous machine-wide memory trace. No
GPU, cloud service or paid API was used. The total 25,600 screening steps
were far below the preregistered 179,200-step total cap and 3,600-second wall
cap. No confirmatory steps were spent.

The baseline suite passed 105 tests with four existing warnings. The expanded
suite passed **111 tests** with the same four warnings, including deterministic
benchmark, feasibility, selection, floor-rule and actual 128-step PPO
screening smoke tests. Ruff lint and format checks and `pip check` passed.
Core pump dynamics, corrected action/reward/
termination contracts, historical Phase 3.3 artifacts and Studio behavior
remain unchanged.

## 6. Interpretation, acceptance and next experiment

This milestone established a reproducible stratified feasible benchmark and
four new constructive avoidable-failure witnesses. It did **not** establish a
confirmed robustness improvement or regression for the new curricula. The
development screen found no candidate that simultaneously showed a stress
gain and preserved nominal tracking. The low realized stress exposure is a
major limitation; the numerical equality of 5% and 10% within each pool is
sampling coincidence, not evidence of a flat response. Three policy-training
seeds would still limit a future population-level inference even if a
candidate had qualified.

Engineering acceptance: previous-phase tests pass; pump physics and PPO
contracts are unchanged; historical artifacts are untouched; existing search,
feasibility, persistence, CLI and replay services are reused; the benchmark
and development selection are deterministic and hash frozen; physical/seed
leakage checks pass; budget, checkpoint and no-eligible gates work on local CPU.
Scientific acceptance: new development stress discriminated the designated
baseline, the final benchmark is constructed but sealed, four additional
avoidable failures were found, **zero** curricula passed development guards,
and no confirmatory matched inference or held-out improvement was observed.
Uncertainty is therefore an absence of confirmatory evidence, not a
statistically significant zero effect.

The next research milestone should preregister a **separate** staged design
with a minimum realized stress-exposure requirement in screening, sufficient
interaction budget to distinguish low ratios, an independent development
checkpoint or more development seeds, and unchanged sealed final holdouts
only if their isolation can be verified. It must freeze those changes before
training and avoid using this study's development outcomes to tune a claim
about the current final sets. No extended profile, Phase 4 implementation,
or further Phase 3.4 search was launched.
