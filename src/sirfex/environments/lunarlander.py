from __future__ import annotations

from typing import Any

import gymnasium as gym

from sirfex.core.interfaces import Scenario
from sirfex.core.scenarios import ScenarioValidationError, validate_scenario_basics


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
        self.validate_scenario(scenario)
        observation, info = self._env.reset(seed=scenario.seed)
        self._env.action_space.seed(scenario.seed)
        return observation, info

    def scenario_spec(self) -> dict[str, Any]:
        return {
            "environment_id": self.id,
            "environment_version": self.version,
            "parameters": [],
            "limitations": (
                "Stock LunarLander-v3 exposes no supported runtime disturbance parameters "
                "through this adapter."
            ),
        }

    def validate_scenario(self, scenario: Scenario) -> None:
        validate_scenario_basics(scenario, self.id, self.version)
        if scenario.parameters:
            raise ScenarioValidationError(
                "LunarLander scenario parameters are unsupported by the stock adapter; "
                "gravity, wind, thrust degradation, and sensor faults are not advertised"
            )

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return self._env.step(action)

    def metadata(self) -> dict[str, Any]:
        return {"environment_version": self.version, "render_mode": None}

    def evaluate_failure(
        self, total_reward: float, terminated: bool, info: dict[str, Any]
    ) -> dict[str, Any]:
        del total_reward
        if not terminated:
            return {
                "outcome": "truncated",
                "observed_outcome": "truncated",
                "termination_reason": info.get("termination_reason", "unknown"),
            }
        if info.get("landing_success") is True:
            return {
                "outcome": "success",
                "observed_outcome": "confirmed_success",
                "termination_reason": "landing_success_reported_by_environment",
            }
        if info.get("landing_success") is False:
            return {
                "outcome": "failure",
                "observed_outcome": "confirmed_failure",
                "termination_reason": "landing_failure_reported_by_environment",
            }
        return {
            "outcome": "unknown",
            "observed_outcome": "terminal_outcome_unavailable",
            "termination_reason": "stock_api_does_not_expose_landing_outcome",
            "evidence_source": "adapter_limitation",
        }

    def close(self) -> None:
        self._env.close()
