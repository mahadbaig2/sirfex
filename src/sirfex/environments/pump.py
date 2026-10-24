"""Transparent single-tank pump model for Phase 0.

Model assumptions: level is normalized to a 0..1 tank fraction, pump inflow is
linear in command and efficiency, demand is a bounded normalized outflow, and
Euler integration uses a fixed time step. This is a numerical test environment,
not a validated representation of a real plant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from sirfex.core.interfaces import Scenario
from sirfex.core.scenarios import (
    ParameterSpec,
    ScenarioValidationError,
    validate_parameter_specs,
    validate_scenario_basics,
)


@dataclass(frozen=True)
class PumpParameters:
    dt: float = 0.1
    horizon: int = 200
    target_level: float = 0.5
    initial_level: float = 0.5
    demand: float = 0.5
    pump_capacity: float = 0.8
    efficiency: float = 1.0
    leak: float = 0.0
    min_level: float = 0.05
    max_level: float = 0.95


class PumpEnvironment:
    id = "pump"
    version = "sirfex-pump:1"

    _parameter_specs = (
        ParameterSpec("dt", "number", 0.1, "s", "Fixed Euler integration step", 1e-9, None),
        ParameterSpec("horizon", "integer", 200, "steps", "Maximum episode length", 1, None),
        ParameterSpec("target_level", "number", 0.5, "fraction", "Desired tank level", 0.0, 1.0),
        ParameterSpec("initial_level", "number", 0.5, "fraction", "Initial tank level", 0.0, 1.0),
        ParameterSpec("demand", "number", 0.5, "normalized flow", "Demand outflow", 0.0, 1.0),
        ParameterSpec(
            "pump_capacity", "number", 0.8, "normalized flow", "Maximum pump inflow", 0.0, None
        ),
        ParameterSpec("efficiency", "number", 1.0, "fraction", "Pump efficiency", 0.0, 1.0),
        ParameterSpec(
            "leak", "number", 0.0, "normalized flow", "Level-proportional leakage", 0.0, None
        ),
        ParameterSpec("min_level", "number", 0.05, "fraction", "Low operating boundary", 0.0, 1.0),
        ParameterSpec("max_level", "number", 0.95, "fraction", "High operating boundary", 0.0, 1.0),
    )

    def __init__(
        self, parameters: PumpParameters | None = None, *, allow_action_clipping: bool = False
    ) -> None:
        self.parameters = parameters or PumpParameters()
        self.allow_action_clipping = allow_action_clipping
        self._active_parameters = self.parameters
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._level = 0.0
        self._step_count = 0
        self._demand = 0.0

    def reset(self, scenario: Scenario) -> tuple[np.ndarray, dict[str, Any]]:
        values = self.validate_scenario(scenario)
        self._active_parameters = PumpParameters(
            **{key: values[key] for key in PumpParameters.__dataclass_fields__}
        )
        self._level = float(values["initial_level"])
        self._demand = float(values["demand"])
        self._step_count = 0
        return self._observation(values), {"demand": self._demand, "seed": scenario.seed}

    def scenario_spec(self) -> dict[str, Any]:
        return {
            "environment_id": self.id,
            "environment_version": self.version,
            "parameters": [spec.to_dict() for spec in self._parameter_specs],
        }

    def validate_scenario(self, scenario: Scenario) -> dict[str, Any]:
        validate_scenario_basics(scenario, self.id, self.version)
        defaults = {
            spec.name: getattr(self.parameters, spec.name) for spec in self._parameter_specs
        }
        validated = validate_parameter_specs(scenario.parameters, self._parameter_specs)
        resolved = {**defaults, **{name: validated[name] for name in scenario.parameters}}
        if resolved["min_level"] >= resolved["max_level"]:
            raise ScenarioValidationError("min_level must be less than max_level")
        if not resolved["min_level"] <= resolved["initial_level"] <= resolved["max_level"]:
            raise ScenarioValidationError("initial_level must be within [min_level, max_level]")
        if not resolved["min_level"] <= resolved["target_level"] <= resolved["max_level"]:
            raise ScenarioValidationError("target_level must be within [min_level, max_level]")
        return resolved

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        command = self._validate_action(action)
        p = self._active_parameters
        if self.allow_action_clipping:
            command = float(np.clip(command, 0.0, 1.0))
        inflow = p.pump_capacity * p.efficiency * command
        outflow = p.demand + p.leak * self._level
        self._level += p.dt * (inflow - outflow)
        self._step_count += 1
        low_violation = self._level < p.min_level
        high_violation = self._level > p.max_level
        violation_magnitude = max(p.min_level - self._level, self._level - p.max_level, 0.0)
        self._level = float(np.clip(self._level, 0.0, 1.0))
        error = self._level - p.target_level
        reward = -float(error * error + 0.01 * command * command)
        terminated = bool(low_violation or high_violation)
        truncated = self._step_count >= p.horizon
        info = {
            "level": self._level,
            "demand": p.demand,
            "inflow": inflow,
            "outflow": outflow,
            "constraint_violation": low_violation or high_violation,
            "violation_magnitude": violation_magnitude,
            "min_level": p.min_level,
            "max_level": p.max_level,
            "termination_reason": "level_constraint"
            if terminated
            else ("horizon" if truncated else None),
        }
        return self._observation(), reward, terminated, truncated, info

    def _observation(self, values: dict[str, Any] | None = None) -> np.ndarray:
        target = (
            self._active_parameters.target_level
            if values is None
            else float(values["target_level"])
        )
        demand = self._demand if values is None else float(values["demand"])
        return np.array([self._level, target, demand], dtype=np.float32)

    def _validate_action(self, action: Any) -> float:
        try:
            value = np.asarray(action)
        except Exception as exc:
            raise ValueError("Pump action must be a one-element numeric array") from exc
        if value.shape != (1,) or not np.issubdtype(value.dtype, np.number):
            raise ValueError("Pump action must have numeric shape (1,)")
        command = float(value[0])
        if not np.isfinite(command):
            raise ValueError("Pump action must be finite")
        if not self.allow_action_clipping and not 0.0 <= command <= 1.0:
            raise ValueError(
                "Pump action must be in [0, 1]; enable action clipping explicitly to clip"
            )
        return command

    def metadata(self) -> dict[str, Any]:
        return {"environment_version": self.version, "model": "single-tank mass balance"}

    def evaluate_failure(
        self, total_reward: float, terminated: bool, info: dict[str, Any]
    ) -> dict[str, Any]:
        del total_reward
        if terminated:
            return {
                "outcome": "failure",
                "observed_outcome": "constraint_violation",
                "failure_category": "constraint_violation",
                "explanation": "Tank level crossed a configured operating constraint.",
                "failure_step": self._step_count - 1,
                "measurements": {
                    "level": info.get("level"),
                    "violation_magnitude": info.get("violation_magnitude", 0.0),
                },
                "threshold": (
                    f"level outside [{self._active_parameters.min_level}, "
                    f"{self._active_parameters.max_level}]"
                ),
                "termination_reason": info.get("termination_reason"),
            }
        return {
            "outcome": "nominal",
            "termination_reason": info.get("termination_reason", "horizon"),
        }

    def close(self) -> None:
        return None
