# Reward and termination study

Both formulas below are training rewards in dimensionless units per discrete
step. They are not physical safety metrics. The original pump simulation reward,
equation, clipping, constraint events and termination remain unchanged.

| Identity | Formula |
|---|---|
| `pump-violation-magnitude:1` (legacy) | `-error² - .01*u² - 10*violation_magnitude` |
| `pump-survival-tracking:2` | `1 - 4*error² - .01*u² - 10*violation_magnitude - 1[terminated]` |

Here error is the post-step clipped tank level minus target; violation magnitude
is measured before clipping by the original adapter. The revised unit bonus
rewards sustained operation, with stronger dense tracking and a modest unit
terminal cost. Every coefficient is fixed by the version rather than optimized
against final holdouts. This deliberately changes the objective; it is not
policy-invariant shaping or a proof of safe optimality. It may favor survival
over precise tracking, which is why the physical/tracking competence gate remains
separate and unchanged.

## Same-trajectory measurements

The diagnostic compares both reward formulas on the same persisted physical
trajectories. Twelve fixed validation scenarios, 200-step horizons, gamma .99:

| Controller | Violations | Mean length | Legacy sum | Legacy discounted sum | Revised sum | Revised discounted sum |
|---|---:|---:|---:|---:|---:|---:|
| Hysteresis | 0/12 | 200 | -2.5169 | -1.0910 | 193.2148 | 83.6586 |
| Zero command | 12/12 | 11.75 | -1.1468 | -1.0536 | 6.8873 | 6.6530 |
| Full command | 12/12 | 14.0833 | -1.6607 | -1.5022 | 7.2499 | 6.9591 |
| Analytical | 0/12 | 200 | -.6356 | -.2923 | 199.2713 | 86.2179 |
| Historical final PPO, either schema | 12/12 | 11.75 | -1.1468 | -1.0536 | 6.8873 | 6.6530 |

The historical final PPO policies have the same results in both deterministic
and the measured seeded stochastic episodes. Ninety-six episodes were persisted:
four nonlearning references and two final PPO schemas in two inference modes,
each across 12 scenarios. Each episode retains per-step reward components,
component sums, undiscounted/discounted returns, termination reason, constraint
event, and reference to its original physical trace. Eight representative replays
matched exactly. Diagnostic run: `20261025T203735_dfaffe5808`.

## What this does and does not establish

Failing early avoids future negative legacy rewards. Relative to hysteresis,
the zero controller has a better mean undiscounted return and slightly better
mean finite-horizon discounted return. But the analytical controller has an even
better legacy return while remaining safe. These examples therefore demonstrate
an incentive concern and an exploration/optimization problem, not that early
failure is the unique or global optimum of the legacy objective.

The finite-horizon sums above are not identical to PPO's bootstrapped targets.
Installed SB3 marks time limits as `truncated and not terminated` and adds the
discounted value of the terminal observation when bootstrapping a time limit.
Physical failures set `terminated=True`, so that bootstrap is absent. An earlier
negative-return ranking alone could not establish PPO's learning objective;
the action-only and reward-only comparisons supply additional empirical evidence.

## Termination decision

`pump-terminate-on-violation:1` is retained for all training. Continuing after an
operating-limit violation was considered but not admitted as a second training
environment: the original adapter eventually clips volume at 0 or 1, so continued
evolution after those physical bounds would not preserve the affine mass balance
used by feasibility analysis. Introducing boundary physics would require a
separate modeling decision. The full-horizon alternative is rejected rather than
quietly erasing failures or substituting an absorbing state with different dynamics.

No penalty is applied merely because a successful episode reaches its horizon.
Reward identity and termination identity are saved in both checkpoint metadata
and new model contracts. Alternate rewards produce identical physical trajectories
under identical physical commands. Legacy coefficients and old experiments remain
reproducible. Evaluation reports simulation return and reconstructed training-form
return separately; neither determines physical failure classification.

```powershell
sirfex train diagnose-reward --config configs/training/reward_diagnostics.json
sirfex train report <diagnostic-run-id>
```
