"""Recorded reward decomposition with the unchanged Phase 3 reward identity."""

import math

REWARD_SCHEMA = "pump-violation-magnitude:1"


def reward_components(level, target, command, violation_magnitude, violation_weight=10.0):
    if (
        not all(
            math.isfinite(x)
            for x in (level, target, command, violation_magnitude, violation_weight)
        )
        or violation_weight < 0
    ):
        raise ValueError("Reward inputs must be finite and penalty weight nonnegative")
    parts = {
        "tracking_penalty": -((level - target) ** 2),
        "actuator_penalty": -0.01 * command**2,
        "constraint_penalty": -violation_weight * violation_magnitude,
    }
    return {**parts, "total": sum(parts.values()), "reward_schema": REWARD_SCHEMA}
