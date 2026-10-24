# SIRFEX Phase 2 Report

## Executive summary

Phase 2 turns the Phase 1 evaluation layer into a local, bounded failure-discovery engine. It validates search spaces, generates reproducible candidates, evaluates them through the existing runner, scores objective-specific risk, persists trials and discoveries, supports random and Optuna TPE strategies, evaluates held-out seeds, and verifies discovery evidence by replay.

## Implemented modules

- `sirfex.search.space`: bounded JSON search-space definitions and deterministic sampling.
- `sirfex.search.objectives`: pump constraint/tracking objectives and LunarLander observation-risk proxy.
- `sirfex.search.strategies`: seeded random and Optuna TPE strategies.
- `sirfex.search.runner`: trial lifecycle, budget accounting, multi-seed aggregation, held-out validation, and discovery generation.
- `sirfex.search.persistence`: atomic search-run artifacts and separate replay verification files.
- `sirfex.search.ranking`: objective-local ranking and five-decimal near-duplicate grouping.
- `sirfex.search.comparison`: fair descriptive comparison of search runs.

## Verification

Executed locally on Windows 11 x64 with Python 3.12:

- 35 automated tests passed before final verification documentation; the Phase 2 search integration includes a real observed pump constraint violation.
- Random search reproducibility tests passed.
- Optuna TPE integration passed locally with Optuna 4.9.0.
- Pump search found valid scenarios with observed constraint violations.
- Pump search discoveries replayed with `exact_match`.
- Held-out pump seeds were evaluated and reported separately.
- The fair 20-trial random/TPE benchmark used 20 simulation evaluations per strategy.

## Evidence limits

The pump result is evidence within the documented simplified single-tank model, not a claim about a real industrial plant. Tracking degradation and LunarLander instability are proxy risks. The stock LunarLander adapter does not expose documented runtime physical disturbance parameters or reliable terminal landing outcomes, so its Phase 2 configuration searches seeded initial-condition behavior and reports observation-derived proxy risk.

## Budget and reproducibility

The `--trials` budget counts candidate proposals. `actual_simulation_evaluations` counts valid candidate/seed executions. Multi-seed evaluation makes the additional cost explicit. Random search owns its NumPy RNG; TPE uses a seeded Optuna sampler. Search manifests record strategy, seed, objective, space, trial budget, scenario seeds, dependencies, and supporting experiment IDs.

## Known limitations

- Search execution is sequential and local; no checkpoint/resume or parallel workers are included.
- Search persistence is single-process and does not coordinate concurrent writers.
- Failed execution evidence is persisted as experiments, but failed search trials do not yet include a richer partial search-level trajectory summary beyond experiment references.
- TPE startup trials make small comparisons noisy.
- The current LunarLander search has no physical parameter variation beyond seed policy.

## Acceptance summary

Implemented: Phase 1 compatibility, failed execution persistence, explicit controller RNG, validated search spaces, pump search, LunarLander risk configuration, mathematical objectives, random/TPE search, budget accounting, multi-seed aggregation, ranking/deduplication, search persistence, replay verification, fair comparison, held-out validation, CLI, tests, and CPU-only benchmarks.

Phase 3 is not started.
