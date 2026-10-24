"""Pure transformations of recorded trajectories for charts and playback."""

from __future__ import annotations

from typing import Any


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def pump_trajectory(
    trajectory: list[dict[str, Any]], scenario: dict[str, Any]
) -> list[dict[str, Any]]:
    parameters = scenario.get("parameters", {})
    result = []
    for item in trajectory:
        info = item.get("info", {})
        level = _number(info.get("level"))
        if level is None:
            observation = item.get("observation", [])
            level = _number(observation[0]) if observation else None
        action = item.get("action", [])
        command = _number(action[0]) if action else None
        result.append(
            {
                "step": item.get("step"),
                "time": item.get("step"),
                "level": level,
                "target_level": parameters.get("target_level", 0.5),
                "min_level": parameters.get("min_level", 0.0),
                "max_level": parameters.get("max_level", 1.0),
                "command": command,
                "constraint_violation": bool(info.get("constraint_violation", False)),
                "demand": _number(info.get("demand")),
                "info": info,
            }
        )
    return result


def lunarlander_trajectory(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in trajectory:
        observation = item.get("observation", [])
        values = list(observation) if isinstance(observation, list) else []
        padded = values + [None] * max(0, 8 - len(values))
        result.append(
            {
                "step": item.get("step"),
                "x": _number(padded[0]),
                "y": _number(padded[1]),
                "vx": _number(padded[2]),
                "vy": _number(padded[3]),
                "angle": _number(padded[4]),
                "angular_velocity": _number(padded[5]),
                "left_leg_contact": padded[6],
                "right_leg_contact": padded[7],
                "action": item.get("action"),
                "reward": item.get("reward"),
            }
        )
    return result


def failure_events(result: dict[str, Any]) -> list[dict[str, Any]]:
    return list(result.get("failures", []))


def search_history(trials: list[dict[str, Any]], direction: str) -> list[dict[str, Any]]:
    best = None
    history = []
    for trial in trials:
        aggregate = trial.get("aggregate", {})
        value = aggregate.get("mean")
        if trial.get("status") != "completed" or not isinstance(value, (int, float)):
            history.append({"trial_index": trial.get("trial_index"), "value": None, "best": best})
            continue
        best = (
            value
            if best is None
            else (max(best, value) if direction == "maximize" else min(best, value))
        )
        history.append({"trial_index": trial.get("trial_index"), "value": value, "best": best})
    return history
