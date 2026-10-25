# Analytical pump reference

The diagnostic reference uses the existing equation
`h_next = h + dt*(C*u - demand - leak*h)`, where `C=capacity*efficiency`.
For positive effective capacity, the equilibrium command is
`u_eq=(demand+leak*target)/C`.

Choose a response fraction `alpha=.25` and gain
`Kp=(alpha/dt-leak)/C`. Apply `clip(u_eq + Kp*(target-h),0,1)`.
Without saturation or numerical rounding, substitution gives
`target-h_next=(1-alpha)*(target-h)`: a pole of .75 and monotone error reduction.
The gain is derived from the equation, not tuned on held-out outcomes. Saturation
can prevent the requested response; the reference is not guaranteed to succeed
on every scenario, especially infeasible or unusual discrete-time regimes.

The controller reads calibrated observations `[level,target,demand,efficiency,
capacity,leak]` and receives the scenario's dt explicitly in persisted controller
configuration. All seven quantities are assumed known. It is an informed model
reference, not an information-matched competitor for minimal-observation PPO.
Zero or negative effective capacity raises an explicit controller error; the
reference never pretends a division by zero yields a valid control proof.

On the 12 Phase 3.1 nominal validation cases, analytical control had zero violations,
full 200-step horizons and mean absolute tracking error .001904. Hysteresis also
had zero violations, with MAE .068919. Zero and full-capacity references each failed
all 12 cases. The analytical result establishes solvability of these particular
modeled cases under its calibration assumptions, not industrial controllability.

```powershell
sirfex experiments run --scenario configs/pump_baseline.json --controller pump-analytical
sirfex experiments replay <experiment-id>
```

`pump-fixed` supplies zero/full diagnostic controls via controller configuration
`{"command": 0}` or `{"command": 1}`. It does not alter the action contract.
The analytical and fixed references use the ordinary controller factories,
EvaluationRunner, experiment persistence and replay. Tests cover equilibrium,
feedback, changed demand, leakage, saturation, zero capacity, finite horizons and
replay under a nondefault timestep.
