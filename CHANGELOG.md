# Changelog

## Unreleased

- Added Phase 3.2 raw PPO action diagnostics, versioned normalized policy actions and survival/tracking reward, an analytical pump reference, bounded three-seed competence studies, and feasibility-filtered avoidable failure search. The nominal gate passed across three seeds; three same-scenario avoidable failures were replay-verified. A matched robustness improvement study has not been run.
- Added Phase 3.1 feasibility classification, versioned calibrated observations, reward diagnostics, bounded learning curves, competence gates, disjoint physical datasets, verified continuation budgets, and training-seed uncertainty. The measured pilot found no competent PPO baseline and no eligible feasible discoveries; robustness remains inconclusive.
- Preserved partial training evidence and added optional CPU-training CI. Corrected null metrics for failed execution, scenario weighting, actual checkpoint timestep metadata, and continuation integrity/configuration checks.

- Renamed the project from AEGIS to SIRFEX (Simulation-based Integrity, Robustness & Failure Exploration). The Python package, imports, CLI, and project structure now use `sirfex`.
- Started Phase 0 with the verified Python package foundation, shared evaluation runner, LunarLander and pump adapters, CLI, tests, benchmarks, and project documentation.
- Adopted the Apache License 2.0 for the project.
- Added Phase 1 scenario validation, failure taxonomy, local experiment persistence, trajectory recording, deterministic replay, comparison, and reproducibility documentation.
- Added Phase 2 bounded scenario search, risk objectives, deterministic random search, Optuna TPE, multi-seed evaluation, discovery ranking, search persistence, replay verification, and fair strategy comparison.
- Added Phase 2.5's optional local Evidence Explorer, headless evidence readers, recorded trajectory views, search-history inspection, compatibility checks, and replay verification integration.
- Added Phase 3 optional CPU PPO training, pump Gymnasium adaptation over shared dynamics, hashed checkpoint provenance, deterministic failure-informed curricula, matched continuation evaluation, training persistence, and explicit regression classification.
