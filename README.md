# SIRFEX

Simulation-based Integrity, Robustness & Failure Exploration

SIRFEX is an open-source, local-first Physical AI testing engine for autonomous controllers. It is designed to evaluate controllers in simulated physical environments, discover reproducible failure scenarios, and help engineers measure and improve controller robustness.

> Find failure conditions before they find your controller.

## Project status

SIRFEX Phase 0 is implemented and verified on the development machine. The technical requirements and architecture are defined in [the technical PRD](SIRFEX_Technical_PRD_and_Architecture_v1.md), and measured results are recorded in [the Phase 0 report](docs/PHASE_0_REPORT.md).

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

## Planned v1 scope

- Headless LunarLander and simplified industrial pump-control environments.
- Shared environment, controller, and evaluation interfaces.
- Reproducible scenario execution, trajectory evidence, and failure replay.
- CPU benchmarking and optional PPO workflows.
- A Python package and CLI using the `sirfex` namespace and command.

Planned capabilities are not presented as implemented features. Retraining curricula, a remote API, and cloud execution remain outside the current implementation. See the PRD for the complete requirements, constraints, architecture, and roadmap.

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
