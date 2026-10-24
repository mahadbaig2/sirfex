"""Gymnasium training adapter over the existing pump dynamics."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from sirfex.core.interfaces import Scenario
from sirfex.environments.pump import PumpEnvironment


class PumpGymEnv(gym.Env[np.ndarray, np.ndarray]):
    """Expose PumpEnvironment through Gymnasium without duplicating physics."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenarios: list[Scenario] | None = None,
        *,
        seed: int = 0,
        violation_weight: float = 10.0,
    ) -> None:
        super().__init__()
        self._adapter = PumpEnvironment()
        self.observation_space = self._adapter.observation_space
        self.action_space = self._adapter.action_space
        self.scenarios = list(scenarios or [Scenario("pump")])
        if not self.scenarios:
            raise ValueError("At least one pump scenario is required")
        if any(item.environment_id != "pump" for item in self.scenarios):
            raise ValueError("PumpGymEnv accepts pump scenarios only")
        self._rng = np.random.default_rng(seed)
        self.violation_weight = float(violation_weight)
        self._scenario = self.scenarios[0]

    @property
    def current_scenario(self) -> Scenario:
        return self._scenario

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and "scenario" in options:
            candidate = options["scenario"]
            self._scenario = (
                candidate if isinstance(candidate, Scenario) else Scenario.from_dict(candidate)
            )
        else:
            self._scenario = self.scenarios[int(self._rng.integers(len(self.scenarios)))]
        observation, info = self._adapter.reset(self._scenario)
        return observation, {**info, "scenario": self._scenario.to_dict()}

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self._adapter.step(action)
        violation = float(info.get("violation_magnitude", 0.0))
        shaped_reward = float(reward - self.violation_weight * violation)
        return observation, shaped_reward, terminated, truncated, {**info, "base_reward": reward}

    def close(self) -> None:
        self._adapter.close()
