# SIRFEX Phase 0 Report

## Status

Phase 0 implementation is complete for the core foundation, two environments, shared evaluation, CLI, tests, and simulation benchmark. PPO status is reported separately below and must not be inferred from the simulation results.

## Implemented components

- Python 3.12 `src/sirfex` package and editable installation.
- Typed environment/controller protocols, `Scenario`, `EpisodeResult`, and environment-agnostic `EvaluationRunner`.
- Headless Gymnasium LunarLander-v3 adapter with random controller.
- Documented single-tank pump model with hysteresis controller.
- `sirfex environments`, `sirfex evaluate`, and `sirfex benchmark` commands.
- Ruff formatting/linting, pytest tests, GitHub Actions CI, and contributor/security documentation.
- Quick benchmark JSON output and standalone PPO smoke-test script.

## Verification record

The following were executed locally on Windows 11 x64 using Python 3.12.10:

- Ruff format check: passed.
- Ruff lint: passed.
- Pytest: 6 passed, 0 failed.
- Pump CLI evaluation: passed.
- LunarLander CLI evaluation: passed headlessly for a 10-step seeded episode.
- Quick simulation benchmark: passed for 5 episodes per environment.
- PPO CPU smoke test: passed for 256 steps; model save/load verified in 4.6758 seconds.

## Measured quick benchmark

These values are from one run on the development machine and are not universal guarantees:

| Environment | Episodes/s | Steps/s | Mean reward |
|---|---:|---:|---:|
| Pump | 78.84 | 15,768.62 | -1.9892 |
| LunarLander random | 88.60 | 8,576.50 | -170.2441 |

Machine observations: Windows 11, Python 3.12.10, 4 physical / 8 logical CPUs, approximately 15.85 GiB visible RAM. The benchmark did not include rendering.

## PPO smoke test

The smoke-test command is:

```powershell
python scripts/ppo_smoke_test.py --steps 256
```

The executed result was `passed`: a CPU PPO model trained for 256 steps and was saved and loaded successfully in 4.6252 seconds. `passed` means a CPU PPO model completed the requested short run and save/load check; `unavailable` means the optional Stable-Baselines3/PyTorch dependencies were not installed.

## PRD deviations and risks

- Artifact persistence, controller registration, failure search, retraining curricula, API, and Studio remain outside Phase 0.
- Exact landing outcome information is limited by the stock Gymnasium API; Phase 0 classifies terminal LunarLander episodes using the documented reward threshold convention.
- The current repository intentionally has no license until the MIT versus Apache-2.0 decision is confirmed.

## Recommendation for Phase 1

Add durable experiment manifests and artifact persistence only after reviewing Phase 0 measurements and confirming the licensing choice. Keep scenario validation and failure taxonomy ahead of search optimization.
