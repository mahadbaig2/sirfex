from __future__ import annotations

from typing import Any

import gymnasium as gym

from sirfex.core.interfaces import Scenario


class LunarLanderEnvironment:
    """Headless adapter around Gymnasium's stock LunarLander-v3 environment."""

    id = "lunarlander"
    version = "gymnasium:LunarLander-v3"

    def __init__(self) -> None:
        self._env = gym.make("LunarLander-v3", render_mode=None)

    @property
    def observation_space(self) -> Any:
        return self._env.observation_space

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    def reset(self, scenario: Scenario) -> tuple[Any, dict[str, Any]]:
        if scenario.parameters:
            unsupported = set(scenario.parameters) - {"seed"}
            if unsupported:
                raise ValueError(
                    "Stock LunarLander-v3 does not support runtime physical changes; "
                    f"unsupported parameters: {sorted(unsupported)}"
                )
        observation, info = self._env.reset(seed=scenario.seed)
        self._env.action_space.seed(scenario.seed)
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return self._env.step(action)

    def metadata(self) -> dict[str, Any]:
        return {"environment_version": self.version, "render_mode": None}

    def evaluate_failure(
        self, total_reward: float, terminated: bool, info: dict[str, Any]
    ) -> dict[str, Any]:
        if not terminated:
            return {
                "outcome": "truncated",
                "termination_reason": info.get("termination_reason", "unknown"),
            }
        if total_reward >= 200.0:
            return {"outcome": "success", "termination_reason": "successful_landing"}
        return {"outcome": "failure", "termination_reason": "crash_or_unsuccessful_termination"}

    def close(self) -> None:
        self._env.close()
