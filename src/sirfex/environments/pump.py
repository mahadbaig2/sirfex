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

    def __init__(self, parameters: PumpParameters | None = None) -> None:
        self.parameters = parameters or PumpParameters()
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
        values = {**self.parameters.__dict__, **scenario.parameters}
        self._validate_parameters(values)
        self._active_parameters = PumpParameters(
            **{key: values[key] for key in PumpParameters.__dataclass_fields__}
        )
        self._level = float(values["initial_level"])
        self._demand = float(values["demand"])
        self._step_count = 0
        return self._observation(values), {"demand": self._demand, "seed": scenario.seed}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        command = float(np.asarray(action).reshape(-1)[0])
        p = self._active_parameters
        command = float(np.clip(command, 0.0, 1.0))
        inflow = p.pump_capacity * p.efficiency * command
        outflow = p.demand + p.leak * self._level
        self._level += p.dt * (inflow - outflow)
        self._step_count += 1
        low_violation = self._level < p.min_level
        high_violation = self._level > p.max_level
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

    @staticmethod
    def _validate_parameters(values: dict[str, Any]) -> None:
        if not 0.0 <= float(values["initial_level"]) <= 1.0:
            raise ValueError("initial_level must be in [0, 1]")
        if float(values["dt"]) <= 0 or int(values["horizon"]) < 1:
            raise ValueError("dt must be positive and horizon must be at least 1")
        if not 0.0 <= float(values["demand"]) <= 1.0:
            raise ValueError("demand must be in [0, 1]")
        if not 0.0 <= float(values["efficiency"]) <= 1.0:
            raise ValueError("efficiency must be in [0, 1]")
        if not 0.0 <= float(values["leak"]) <= 1.0:
            raise ValueError("leak must be in [0, 1]")

    def metadata(self) -> dict[str, Any]:
        return {"environment_version": self.version, "model": "single-tank mass balance"}

    def evaluate_failure(
        self, total_reward: float, terminated: bool, info: dict[str, Any]
    ) -> dict[str, Any]:
        del total_reward
        if terminated:
            return {"outcome": "failure", "termination_reason": info.get("termination_reason")}
        return {
            "outcome": "nominal",
            "termination_reason": info.get("termination_reason", "horizon"),
        }

    def close(self) -> None:
        return None
