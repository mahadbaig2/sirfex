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

## Planned v1 scope

- Headless LunarLander and simplified industrial pump-control environments.
- Shared environment, controller, and evaluation interfaces.
- Reproducible scenario execution, trajectory evidence, and failure replay.
- CPU benchmarking and optional PPO workflows.
- A Python package and CLI using the `sirfex` namespace and command.

Planned capabilities are not presented as implemented features. Automated failure discovery, Optuna search, retraining curricula, persistence, API, and Studio remain future work. See the PRD for the complete requirements, constraints, architecture, and roadmap.

## Development

Development is being carried out incrementally in public. Each milestone will be verified locally before it is pushed, and this README plus the GitHub repository description will be kept aligned with the actual implementation status.

The supported Phase 0 interpreter is Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,simulation,benchmark,search]"
```

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
python scripts/ppo_smoke_test.py --steps 256
python scripts/run_search_benchmark.py --trials 20 --seed 11
```

The benchmark suite records machine-specific measurements; it does not promise universal throughput. Phase 1 experiment artifacts are local and ignored by Git by default.

## License

SIRFEX is licensed under the [Apache License 2.0](LICENSE). The license does not cover third-party dependencies, which retain their own licenses and notices; see [dependency license notes](docs/DEPENDENCY_LICENSES.md).

## History

SIRFEX was formerly called AEGIS. The rename is recorded in [CHANGELOG.md](CHANGELOG.md).
