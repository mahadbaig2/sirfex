# SIRFEX

Simulation-based Integrity, Robustness & Failure Exploration

SIRFEX is an open-source, local-first Physical AI testing engine for autonomous controllers. It is designed to evaluate controllers in simulated physical environments, discover reproducible failure scenarios, and help engineers measure and improve controller robustness.

> Find failure conditions before they find your controller.

Phase 3.7 Checkpoint 1 adds local, bounded software calibration for signed pump constraint margins, disturbance grid outcomes, same-scenario differential findings, and reproducible reports. Checkpoint 1.1 hardens evidence immutability, prepared-grid integrity and independent replay for selected findings. Checkpoint 2A has a guarded execution engine, deterministic recovery and scientific report regeneration, validated on a bounded synthetic pilot. Checkpoint 2A.1 separates unknown physical feasibility in reports, strengthens crash and budget recovery, and requires a separate human authorization record. The scientific Phase 3.7 comparison has not run; the protocol remains an unapproved, non-executable candidate. See the [Checkpoint 2A.1 report](docs/PHASE_3_7_CHECKPOINT_2A_1_REPORT.md), [Checkpoint 2A completion report](docs/PHASE_3_7_CHECKPOINT_2A_COMPLETION_REPORT.md), [study execution guide](docs/ROBUSTNESS_STUDY_EXECUTION.md), [authorization procedure](docs/ROBUSTNESS_STUDY_AUTHORIZATION.md), [Checkpoint 1 report](docs/PHASE_3_7_CHECKPOINT_1_REPORT.md), [Checkpoint 1.1 report](docs/PHASE_3_7_CHECKPOINT_1_1_REPORT.md), [margin specification](docs/ROBUSTNESS_MARGIN_SPECIFICATION.md), [disturbance protocol](docs/DISTURBANCE_TOLERANCE_PROTOCOL.md), and [differential testing guide](docs/DIFFERENTIAL_CONTROLLER_TESTING.md).

Phase 3.6 adds development-only numerical policy diagnostics and an optional, approval-gated agent research workflow. A matched seed-3 pilot compared a GPT-OSS 120B proposal with a deterministic non-LLM strategy at 51,200 PPO interactions each. Both improved development stress tracking but failed the nominal-preservation guardrail; the agent strategy had higher nominal and stress tracking error in this pilot. Final holdouts remain sealed. See the [Phase 3.6 report](docs/PHASE_3_6_REPORT.md), [comparison](docs/AGENT_VS_BASELINE_EVALUATION.md), and [research protocol](docs/AGENT_EXPERIMENT_PROTOCOL.md).

## Project status

SIRFEX implements evaluation, failure search, a local Evidence Explorer, optional CPU PPO workflows, and development-only policy research through Phase 3.6. Phase 3.3 found **no incremental held-out violation reduction** from 30% failure-informed training and **nominal tracking regression in all three seeds**. Phase 3.4 froze a more varied feasible-stress benchmark and found four new avoidable failures, but **no low-intensity curriculum passed its development screening rule**. Phase 3.5 measures actual curriculum interactions and runs a frozen exposure-controlled comparison. Its research result is recorded in the [Phase 3.5 report](docs/PHASE_3_5_REPORT.md). The technical requirements remain in [the PRD](SIRFEX_Technical_PRD_and_Architecture_v1.md).

The project is intentionally CPU-first and local-first. It will not require paid APIs, hosted simulation, cloud training, or a dedicated GPU.

## Implemented in Phase 0

- Typed environment and controller interfaces with a shared evaluation runner.
- Headless Gymnasium LunarLander-v3 evaluation with a random controller.
- Simplified single-tank pump simulation with a deterministic rule controller.
- JSON-serializable episode results and seeded evaluation.
- `sirfex` CLI for environment listing, evaluation, and quick benchmarks.
- CPU benchmark suite and a short CPU-only PPO save/load smoke test.

## Implemented in Phase 1

- Versioned scenario JSON with strict pump validation.
- Explicit failure taxonomy and evidence-based LunarLander outcome semantics.
- Atomic local experiment manifests, results, and JSONL trajectories.
- Pump and LunarLander deterministic replay with verification reports.
- Descriptive experiment comparison without a misleading universal score.
- Apache-2.0 licensing and dependency/version provenance in manifests.

See the [Phase 1 report](docs/PHASE_1_REPORT.md) for the verified status and limitations.

## Implemented in Phase 2

- Bounded, versioned search-space definitions for valid scenarios.
- Pump constraint-violation and tracking-degradation objectives.
- LunarLander observation-derived instability risk proxy.
- Deterministic random search and seeded Optuna TPE search.
- Multi-seed aggregation, held-out validation, discovery ranking, and near-duplicate grouping.
- Persisted search runs, discovery evidence, and replay verification.

See the [Phase 2 report](docs/PHASE_2_REPORT.md) and [search benchmark methodology](docs/SEARCH_BENCHMARKS.md). Search results are evidence within documented simulations, not real-world safety certification.

## Implemented in Phase 2.5

- Optional local Streamlit Evidence Explorer installed through the `studio` extra.
- Read-only experiment and search browsers over the existing artifact stores.
- Recorded pump trajectory charts and deterministic manual playback.
- Observation-based LunarLander trajectory inspection.
- Failure evidence, discovery references, search convergence, and compatible-run comparisons.
- Existing replay verification exposed as the Explorer's only simulation-related action.

See [Evidence Explorer usage](docs/EVIDENCE_EXPLORER.md) and the [Phase 2.5 report](docs/PHASE_2_5_REPORT.md).

## Implemented in Phase 3

- CPU-only Stable-Baselines3 PPO training for the existing pump environment.
- Hashed, provenance-rich PPO checkpoints and local training artifacts.
- Failure-informed curricula consuming eligible Phase 2 discoveries.
- Matched normal and failure-informed continuations with nominal, known-stress, and held-out-stress evaluation.
- Explicit `improved`, `regressed`, or `inconclusive` regression classification.

See the [Phase 3 report](docs/PHASE_3_REPORT.md) and [training commands](docs/PPO_CONTROLLER.md). Phase 3 smoke results do not claim PPO robustness improvement.

## Implemented in Phase 3.1

- Pump physical feasibility analysis, with target, finite-horizon, infeasible and undetermined outcomes.
- Versioned three-value and six-value observations over unchanged pump physics.
- Reward diagnostics, bounded PPO learning curves and a predeclared competence gate.
- Reproducible physical scenario partitions, leakage checks and checkpoint selection independent of final holdouts.
- Gated, budget-verified continuation and uncertainty across independent training seeds.
- Recoverable partial artifacts, policy replay evidence and reviewable public results.

The bounded pilot found all 16 existing discoveries physically infeasible and both PPO observation variants noncompetent at 25,600 steps. Continuation was skipped, and robustness remains **inconclusive**. See [learning curves](docs/PPO_LEARNING_CURVES.md), [feasibility](docs/PUMP_FEASIBILITY.md), and the [experiment protocol](docs/EXPERIMENT_PROTOCOL.md). These are simulation findings, not industrial safety claims.

## Implemented in Phase 3.2

- Diagnosed historical PPO Gaussian output before clipping and compared deterministic and stochastic commands.
- Added a versioned normalized policy action coordinate and a versioned survival/tracking reward, preserving the original pump physics and legacy checkpoint contract.
- Added an analytical pump reference, per-step reward analysis and a bounded three-seed nominal competence study.
- Added feasibility-filtered search with same-scenario reference controls, finite-horizon witnesses and replay verification.

The [competence study](docs/PPO_COMPETENCE_STUDY.md) met the unchanged nominal gate for seeds 3, 17 and 41. The [Phase 3.2 report](docs/PHASE_3_2_REPORT.md) explains the controlled comparison, the three verified avoidable failures, CPU costs and limits. Those were the fixed inputs to Phase 3.3; Phase 3.2 alone did not demonstrate robustness improvement.

## Implemented in Phase 3.3

- Froze three competent PPO baselines, three replay-verified challenges, disjoint physical partitions and a 60-configuration feasible held-out stress set before final evaluation.
- Continued each baseline through matched normal and failure-informed 25,600-step CPU PPO arms from independently loaded copies of the same model and optimizer state.
- Added a strict checkpoint and partition preflight, actual per-scenario curriculum exposure counts, and a final-only evaluation gate across all six continuations.
- Prespecified per-seed nominal and stress guardrails, paired training-seed differences and exploratory uncertainty.

The two arms each reached 0/60 held-out violations for all three seeds after continuation, leaving no incremental physical-violation gain. Failure-informed training degraded nominal mean tracking error by 235.93%, 40.76% and 106.43% relative to its normal arm; the overall frozen classification is **`regressed`**. See the [preregistered protocol](docs/MATCHED_ROBUSTNESS_STUDY.md), [held-out set](docs/HELD_OUT_STRESS_EVALUATION.md), [statistical analysis](docs/ROBUSTNESS_STATISTICAL_ANALYSIS.md), and [Phase 3.3 report](docs/PHASE_3_3_REPORT.md) for details and limits.

## Implemented in Phase 3.4

- Analyzed the saved Phase 3.3 nominal regression and action distributions without changing historical evidence.
- Froze a controller-independent, four-stratum feasible pump benchmark with 24 development stress, 64 sealed final stress and 12 sealed final nominal scenarios.
- Reused bounded feasibility-aware search and replay to add four avoidable failures to the three historical discoveries.
- Screened equal-step normal, 5% and 10% failure-informed candidates on development data under a fixed nominal-preservation rule.

The [Phase 3.4 report](docs/PHASE_3_4_REPORT.md) records **`no_eligible_curriculum`**. Original-pool candidates had no stress gain; expanded-pool candidates breached the 10% nominal tracking limit. As preregistered, no confirmatory training or final holdout evaluation followed. The [benchmark design](docs/STRESS_BENCHMARK_DESIGN.md), [screening rule](docs/CURRICULUM_SCREENING.md), [nominal-preservation policy](docs/NOMINAL_PRESERVATION.md), [study evidence](docs/evidence/phase_3_4_results.json) and [search evidence](docs/evidence/phase_3_4_search.json) preserve the inputs and results.

## Phase 3.5 exposure-controlled study

Training now records actual physical interactions, completed and partial episodes, per-discovery exposure, and PPO step reconciliation. The [exposure accounting](docs/CURRICULUM_EXPOSURE_ACCOUNTING.md), [adequacy gate](docs/EXPOSURE_ADEQUACY_PROTOCOL.md), [frozen protocol](docs/PHASE_3_5_EXPERIMENT_PROTOCOL.md), and [Phase 3.5 report](docs/PHASE_3_5_REPORT.md) explain the intervention and its measured outcome. The 5% and 10% arms are episode-boundary **interaction targets**, with actual exposure reported separately.

## Planned v1 scope

- Headless LunarLander and simplified industrial pump-control environments.
- Shared environment, controller, and evaluation interfaces.
- Reproducible scenario execution, trajectory evidence, and failure replay.
- CPU benchmarking and optional PPO workflows.
- A Python package and CLI using the `sirfex` namespace and command.

Bounded retraining and matched robustness evaluation are implemented; these studies have not demonstrated an incremental held-out robustness benefit. A remote API and cloud execution remain outside the current results. See the PRD for the complete requirements, constraints, architecture, and roadmap.

## Development

Development is being carried out incrementally in public. Each milestone will be verified locally before it is pushed, and this README plus the GitHub repository description will be kept aligned with the actual implementation status.

The supported Phase 0 interpreter is Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,simulation,benchmark,search]"
```

For Phase 3 PPO workflows, install the isolated optional training dependencies with `python -m pip install -e ".[training]"` (the broader `benchmark` extra also includes them).

Run the verified checks and commands:

```powershell
python -m ruff format --check src tests scripts
python -m ruff check src tests scripts
python -m pytest
sirfex environments
sirfex evaluate --env lunarlander --controller random
sirfex evaluate --env pump --controller rule-based
sirfex benchmark --profile quick
sirfex scenarios list --env pump
sirfex scenarios validate --file configs/pump_baseline.json
sirfex experiments run --scenario configs/pump_baseline.json --controller pump-rule
sirfex experiments list
sirfex experiments replay <experiment-id>
sirfex experiments compare <experiment-a> <experiment-b>
sirfex train baseline --config configs/training/pump_ppo.json
sirfex train improve --baseline <training-run-id> --search <search-id> --config configs/training/pump_robustness.json
sirfex train list
sirfex train show <training-run-id>
sirfex train evaluate <training-run-id>
sirfex train compare <training-run-id>
sirfex train verify <training-run-id>
sirfex train feasibility
sirfex train diagnose
sirfex train learning-curve --config configs/training/pump_learning_curve.json
sirfex train validate --config configs/training/pump_validation.json
sirfex train report <training-run-id>
sirfex train robustness-study --config configs/training/pump_phase_3_3.json --prepare-only
sirfex train robustness-study --study <prepared-study-id>
sirfex train report <matched-study-id>
sirfex train verify <matched-study-id>
sirfex train phase-3-4 --config configs/training/pump_phase_3_4.json --stage prepare
sirfex train phase-3-4 --study <study-id> --stage screen
sirfex train phase-3-4 --study <selected-study-id> --stage confirm
sirfex train phase-3-5 --config configs/training/pump_phase_3_5.json --stage prepare
sirfex train phase-3-5 --study <study-id> --stage screen
sirfex train phase-3-5 --study <selected-study-id> --stage confirm
python scripts/ppo_smoke_test.py --steps 256
python scripts/run_training_benchmark.py --profile smoke --seed 3
python scripts/run_search_benchmark.py --trials 20 --seed 11
pip install -e ".[studio]"
streamlit run studio/app.py
```

The Explorer reads `artifacts/experiments` and `artifacts/searches` by default. For alternate local stores, pass Streamlit arguments such as `streamlit run studio/app.py -- --experiment-store path/to/experiments --search-store path/to/searches`.

The benchmark suite records machine-specific measurements; it does not promise universal throughput. Phase 1 experiment artifacts are local and ignored by Git by default.

## License

SIRFEX is licensed under the [Apache License 2.0](LICENSE). The license does not cover third-party dependencies, which retain their own licenses and notices; see [dependency license notes](docs/DEPENDENCY_LICENSES.md).

## History

SIRFEX was formerly called AEGIS. The rename is recorded in [CHANGELOG.md](CHANGELOG.md).
