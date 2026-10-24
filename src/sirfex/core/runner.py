"""Shared episode execution engine."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from sirfex.core.failures import EvaluationError, FailureCategory, FailureRecord
from sirfex.core.interfaces import ControllerAdapter, EnvironmentAdapter, Scenario
from sirfex.core.results import EpisodeResult

LOGGER = logging.getLogger(__name__)


@contextmanager
def managed_environment(factory: Callable[[], EnvironmentAdapter]) -> Iterator[EnvironmentAdapter]:
    """Create and reliably close an environment owned by the caller's context."""
    environment = factory()
    try:
        yield environment
    finally:
        environment.close()


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

        trace: list[dict[str, Any]] = []
        total_reward = 0.0
        terminated = False
        truncated = False
        reset_info: dict[str, Any] = {}
        step_info: dict[str, Any] = {}
        steps_executed = 0

        try:
            try:
                observation, reset_info = environment.reset(scenario)
            except Exception as exc:
                raise EvaluationError(
                    FailureRecord(
                        FailureCategory.ENVIRONMENT_EXCEPTION,
                        f"Environment reset failed: {exc}",
                        evidence_source="runner",
                    )
                ) from exc
            try:
                controller.reset()
            except Exception as exc:
                raise EvaluationError(
                    FailureRecord(
                        FailureCategory.CONTROLLER_EXCEPTION,
                        f"Controller reset failed: {exc}",
                        evidence_source="runner",
                    )
                ) from exc
            if not environment.observation_space.contains(observation):
                raise EvaluationError(
                    FailureRecord(
                        FailureCategory.INVALID_OBSERVATION,
                        "Environment returned an observation outside its observation space",
                        evidence_source="runner",
                    )
                )
            step_info = dict(reset_info)
            for step_index in range(self.max_steps):
                try:
                    action = controller.act(observation, deterministic=deterministic)
                except Exception as exc:
                    raise EvaluationError(
                        FailureRecord(
                            FailureCategory.CONTROLLER_EXCEPTION,
                            f"Controller action generation failed at step {step_index}: {exc}",
                            step=step_index,
                            evidence_source="runner",
                        )
                    ) from exc
                if not environment.action_space.contains(action):
                    raise EvaluationError(
                        FailureRecord(
                            FailureCategory.INVALID_ACTION,
                            f"Controller returned action outside action space: {action!r}",
                            step=step_index,
                            evidence_source="runner",
                        )
                    )
                try:
                    next_observation, reward, terminated, env_truncated, step_info = (
                        environment.step(action)
                    )
                except Exception as exc:
                    raise EvaluationError(
                        FailureRecord(
                            FailureCategory.ENVIRONMENT_EXCEPTION,
                            f"Environment step failed at step {step_index}: {exc}",
                            step=step_index,
                            evidence_source="runner",
                        )
                    ) from exc
                if not environment.observation_space.contains(next_observation):
                    raise EvaluationError(
                        FailureRecord(
                            FailureCategory.INVALID_OBSERVATION,
                            f"Environment returned an invalid observation at step {step_index}",
                            step=step_index,
                            evidence_source="runner",
                        )
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
            failures = []
            if failure.get("failure_category"):
                failures.append(
                    FailureRecord(
                        category=FailureCategory(failure["failure_category"]),
                        explanation=str(failure.get("explanation", "")),
                        step=failure.get("failure_step"),
                        measurements=failure.get("measurements"),
                        threshold=failure.get("threshold"),
                        evidence_source=str(failure.get("evidence_source", "environment")),
                    )
                )
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
                observed_outcome=str(
                    failure.get("observed_outcome", failure.get("outcome", "unknown"))
                ),
                failures=failures,
            )
        except EvaluationError as exc:
            exc.partial_trace = trace
            raise
        finally:
            LOGGER.debug(
                "Finished episode for %s with controller %s", environment.id, controller.id
            )
