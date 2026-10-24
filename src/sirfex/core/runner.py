"""Shared episode execution engine."""

from __future__ import annotations

import logging
from typing import Any

from sirfex.core.interfaces import ControllerAdapter, EnvironmentAdapter, Scenario
from sirfex.core.results import EpisodeResult

LOGGER = logging.getLogger(__name__)


class EvaluationRunner:
    """Run complete episodes without knowing environment-specific physics."""

    def __init__(self, max_steps: int = 1_000) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps

    def run_episode(
        self,
        environment: EnvironmentAdapter,
        controller: ControllerAdapter,
        scenario: Scenario,
        *,
        deterministic: bool = True,
        record_trace: bool = True,
    ) -> EpisodeResult:
        if scenario.environment_id != environment.id:
            raise ValueError(
                f"Scenario targets {scenario.environment_id!r}, "
                f"but environment is {environment.id!r}"
            )

        observation, reset_info = environment.reset(scenario)
        controller.reset()
        if not environment.observation_space.contains(observation):
            raise ValueError("Environment returned an observation outside its observation space")

        trace: list[dict[str, Any]] = []
        total_reward = 0.0
        terminated = False
        truncated = False
        step_info = dict(reset_info)
        steps_executed = 0

        try:
            for step_index in range(self.max_steps):
                action = controller.act(observation, deterministic=deterministic)
                if not environment.action_space.contains(action):
                    raise ValueError(f"Controller returned action outside action space: {action!r}")
                next_observation, reward, terminated, env_truncated, step_info = environment.step(
                    action
                )
                steps_executed += 1
                total_reward += float(reward)
                if record_trace:
                    trace.append(
                        {
                            "step": step_index,
                            "observation": observation.tolist()
                            if hasattr(observation, "tolist")
                            else observation,
                            "action": action.tolist() if hasattr(action, "tolist") else action,
                            "reward": float(reward),
                            "terminated": terminated,
                            "truncated": env_truncated,
                            "info": dict(step_info),
                        }
                    )
                observation = next_observation
                truncated = bool(env_truncated)
                if terminated or truncated:
                    break
            else:
                truncated = True
                step_info = {**step_info, "termination_reason": "runner_max_steps"}

            failure = environment.evaluate_failure(total_reward, terminated, step_info)
            return EpisodeResult(
                environment_id=environment.id,
                controller_id=controller.id,
                scenario=scenario.to_dict(),
                seed=scenario.seed,
                episode_reward=total_reward,
                episode_length=steps_executed,
                terminated=terminated,
                truncated=truncated,
                outcome=str(failure.get("outcome", "unknown")),
                metrics={**environment.metadata(), **failure, "reset_info": reset_info},
                trace=trace,
            )
        finally:
            LOGGER.debug(
                "Finished episode for %s with controller %s", environment.id, controller.id
            )
