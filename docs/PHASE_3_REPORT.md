# SIRFEX Phase 3 Report

## Executive summary

Phase 3 adds a local CPU PPO training and robustness-evaluation process focused on the existing pump environment. Stable-Baselines3 remains optional. The pump Gymnasium adapter delegates transitions to the existing `PumpEnvironment`; no second physics implementation was introduced.

The verified smoke experiment used Python 3.12, Stable-Baselines3 2.9.0, Torch 2.14.1, a `[64, 64]` MLP, CPU device, seed 3, and 256 timesteps. The baseline PPO evaluation had 100% nominal constraint-violation frequency and mean tracking error `0.2430`, while the rule-based reference had 0% and `0.0559` respectively. This intentionally demonstrates that a smoke run is not evidence of PPO competence.

A matched continuation run selected 8 observed, replay-verified Phase 2 discoveries. Normal and failure-informed continuations both started from the same baseline checkpoint and used matched 256-timestep budgets. Nominal, known-stress, and held-out-stress evaluations were persisted. The resulting classification was `inconclusive`; no robustness improvement claim is made.

## Architecture

```text
Phase 2 discoveries → eligibility/selection → deterministic curriculum
                                      ↓
                         PumpGymEnv → Stable-Baselines3 PPO
                                      ↓
                    hashed checkpoint + provenance manifest
                                      ↓
          fixed nominal / known-stress / held-out-stress evaluation
                                      ↓
                         explicit regression classification
```

## Verification status

- Existing Phase 2.5 functionality remains available.
- Pump transition-equivalence, spaces, action validation, curriculum determinism, hash validation, and regression classification are covered by automated tests.
- 46 tests passed in the Phase 3 implementation environment, with the existing three LunarLander SWIG warnings.
- CPU smoke baseline and matched continuation completed locally.
- Stable-Baselines3 and Torch are optional benchmark dependencies; core imports remain isolated.

## Limitations

Smoke-scale PPO is not a meaningful scientific performance sample. The current report has no confidence intervals because the scenario counts are intentionally small. The simplified pump is not a validated industrial plant. Bitwise deterministic learning is not claimed across platforms or dependency versions. Extended training is opt-in and should be benchmarked before drawing conclusions.
