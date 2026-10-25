"""Explicit reward formulations; physics and physical termination never change."""

from sirfex.training.diagnostics import REWARD_SCHEMA, reward_components

LEGACY = REWARD_SCHEMA
SURVIVAL = "pump-survival-tracking:2"
TERMINATION = "pump-terminate-on-violation:1"


def reward_spec(schema):
    if schema == LEGACY:
        return {
            "schema": schema,
            "tracking_weight": 1.0,
            "alive_bonus": 0.0,
            "actuator_weight": 0.01,
            "violation_weight": 10.0,
            "terminal_cost": 0.0,
            "termination_schema": TERMINATION,
        }
    if schema == SURVIVAL:
        return {
            "schema": schema,
            "tracking_weight": 4.0,
            "alive_bonus": 1.0,
            "actuator_weight": 0.01,
            "violation_weight": 10.0,
            "terminal_cost": 1.0,
            "termination_schema": TERMINATION,
        }
    raise ValueError(f"Unknown pump reward schema: {schema}")


def components(schema, level, target, command, violation, terminated):
    spec = reward_spec(schema)
    base = reward_components(level, target, command, violation)
    terms = {
        "tracking_penalty": spec["tracking_weight"] * base["tracking_penalty"],
        "actuator_penalty": base["actuator_penalty"],
        "constraint_penalty": base["constraint_penalty"],
        "alive_bonus": spec["alive_bonus"],
        "terminal_penalty": -spec["terminal_cost"] if terminated else 0.0,
    }
    return {**terms, "total": sum(terms.values()), "reward_schema": schema}
