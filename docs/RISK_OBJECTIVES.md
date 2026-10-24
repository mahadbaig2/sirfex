# Risk Objectives

## Pump constraint violation

For each trajectory step, the pump environment reports whether the level crossed its configured bounds and the violation magnitude:

```text
max(min_level - level, level - max_level, 0)
```

The objective value is the maximum observed violation magnitude. It is maximized, uses normalized level-fraction units, and is `observed` environment evidence. A positive value is a physical constraint violation in the simplified model.

## Pump tracking degradation

The tracking objective computes mean absolute error over recorded steps:

```text
mean(abs(level[t] - target_level[t]))
```

It is maximized as an `inferred` proxy. A threshold of `0.1` marks substantial degradation for ranking, but it is not treated as a physical safety violation.

## LunarLander terminal-instability proxy

For each recorded observation, the proxy combines supported observation components:

```text
abs(vertical_velocity) + 0.5 * abs(horizontal_velocity) + 0.5 * abs(angle)
```

The maximum trajectory value is maximized as an `inferred` proxy. Stock LunarLander terminal outcomes remain `unknown` unless explicit environment evidence is provided; the proxy is not a landing-success classifier.
