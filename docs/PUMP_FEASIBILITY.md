# Pump feasibility

The simulation equation and constraints are unchanged (`sirfex-pump:1`). Let
`a = 1 - dt*leak`, `b = dt*capacity*efficiency`, and `c = -dt*demand`.
The raw next level is `a*h + b*u + c`, with `0 <= u <= 1`. The simulator checks
constraints before clipping the level to `[0,1]`.

At target `T`, an equilibrium command exists when
`demand + leak*T <= capacity*efficiency`. An equilibrium somewhere in the allowed
interval exists when `demand + leak*min_level <= capacity*efficiency`, because
demand and leakage are nonnegative. These equilibrium tests alone do not establish
safe reachability from the initial state, or prove failure during a short horizon.

Starting with reachable interval `[initial,initial]`, propagate:

```text
next_low  = max(min_level, min(a*low,a*high) + c)
next_high = min(max_level, max(a*low,a*high) + b + c)
```

The affine image is an interval because states and actions are continuous.
Intersecting with the safe interval at every step retains exactly the states
reachable through a safe history. The `min/max` form also handles `a < 0`.
An empty interval proves no command sequence can remain within bounds through
that step under this model. A nonempty interval through the requested horizon
proves finite-horizon existence, without claiming a particular controller achieves it.

| Classification | Meaning |
|---|---|
| `target_feasible` | Finite-horizon viability and a target equilibrium both hold; target reachability or convergence is not separately proved. |
| `bounded_operation_feasible` | A safe finite-horizon sequence exists, but the target equilibrium does not. The interval-equilibrium field distinguishes indefinitely sustainable operation from temporary use of stored tank volume. |
| `infeasible` | Reachable safe states become empty before or at the horizon. |
| `undetermined` | Reachability budget ends before the horizon; no infeasibility proof has been obtained. |

Inputs undergo the original scenario validation. Calculations use exact rational
representations of decimal input values, capped at 10,000 steps. This avoids
accumulated analyzer roundoff but does not model binary floating-point simulator
roundoff. Boundary equalities can consequently differ by machine precision.
Results concern this discrete model, constant parameters, continuous actions, and
the stated horizon—not continuous-time plant controllability or real-world safety.

The Phase 3.1 audit found **16 valid, replay-verified discoveries: 16 infeasible,
0 target-feasible, 0 bounded-operation-feasible, 0 undetermined**. All original
failure artifacts remain unchanged. They document operating-envelope limitations;
none enters the main robustness curriculum. Classification is additional analysis,
not a replacement for the observed failure taxonomy.

```powershell
sirfex train feasibility
sirfex train feasibility --search <search-id>
sirfex train diagnose
```

Every supporting experiment is loaded and replayed again. Missing evidence,
incompatible versions, execution errors, mismatched scenarios, unverified replay,
and duplicates exclude a discovery. The validation study also requires fixed
`dt=.1`, bounds `[.05,.95]`, horizon `200`, and supported calibrated observation
ranges; these exclusions are persisted without editing the source discovery.
