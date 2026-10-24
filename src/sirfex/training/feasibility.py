"""Exact interval reachability for the constant-parameter pump model.

This interpretation never changes a scenario or its recorded failure evidence.
Decimal input values are treated as exact rationals; simulator rounding remains
a separate limitation. Constraints are tested before the simulator's clipping.
"""

from fractions import Fraction

from sirfex.core import Scenario
from sirfex.environments.pump import PumpEnvironment


def analyze_feasibility(scenario: Scenario, *, max_horizon: int = 10000) -> dict:
    if type(max_horizon) is not int or max_horizon < 1:
        raise ValueError("Reachability budget must be a positive integer")
    p = PumpEnvironment().validate_scenario(scenario)
    f = {k: Fraction(str(v)) for k, v in p.items()}
    capacity = f["pump_capacity"] * f["efficiency"]
    target_flow = f["demand"] + f["leak"] * f["target_level"]
    target = target_flow <= capacity
    equilibrium = f["demand"] + f["leak"] * f["min_level"] <= capacity
    a, b, c = 1 - f["dt"] * f["leak"], f["dt"] * capacity, -f["dt"] * f["demand"]
    lo = hi = f["initial_level"]
    viable = None
    impossible_step = None
    for step in range(1, min(p["horizon"], max_horizon) + 1):
        lo, hi = (
            max(f["min_level"], min(a * lo, a * hi) + c),
            min(f["max_level"], max(a * lo, a * hi) + b + c),
        )
        if lo > hi:
            viable, impossible_step = False, step
            break
        if step == p["horizon"]:
            viable = True
    classification = (
        "undetermined"
        if viable is None
        else "infeasible"
        if not viable
        else "target_feasible"
        if target
        else "bounded_operation_feasible"
    )
    return {
        "schema_version": "pump-feasibility:1",
        "classification": classification,
        "target_equilibrium": bool(target),
        "interval_equilibrium": bool(equilibrium),
        "finite_horizon_viable": viable,
        "horizon": p["horizon"],
        "first_unavoidable_violation_step": impossible_step,
        "maximum_inflow": float(capacity),
        "target_required_inflow": float(target_flow),
        "target_equilibrium_command": float(target_flow / capacity) if capacity else None,
        "reachable_final_interval": [float(lo), float(hi)] if viable else None,
        "method": "exact-rational affine reachable intervals with per-step constraints",
        "parameters": p,
    }
