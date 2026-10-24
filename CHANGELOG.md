# Changelog

## Unreleased

- Renamed the project from AEGIS to SIRFEX (Simulation-based Integrity, Robustness & Failure Exploration). The Python package, imports, CLI, and project structure now use `sirfex`.
- Started Phase 0 with the verified Python package foundation, shared evaluation runner, LunarLander and pump adapters, CLI, tests, benchmarks, and project documentation.
- Adopted the Apache License 2.0 for the project.
- Added Phase 1 scenario validation, failure taxonomy, local experiment persistence, trajectory recording, deterministic replay, comparison, and reproducibility documentation.
- Added Phase 2 bounded scenario search, risk objectives, deterministic random search, Optuna TPE, multi-seed evaluation, discovery ranking, search persistence, replay verification, and fair strategy comparison.
- Added Phase 2.5's optional local Evidence Explorer, headless evidence readers, recorded trajectory views, search-history inspection, compatibility checks, and replay verification integration.
- Added Phase 3 optional CPU PPO training, pump Gymnasium adaptation over shared dynamics, hashed checkpoint provenance, deterministic failure-informed curricula, matched continuation evaluation, training persistence, and explicit regression classification.
