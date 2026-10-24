"""Fixed-set controller evaluation and aggregate robustness metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from sirfex.core import EvaluationRunner
from sirfex.environments.pump import PumpEnvironment


@dataclass(frozen=True)
class EvaluationSummary:
    set_name: str
    controller_id: str
    scenarios: int
    episodes: int
    constraint_violation_frequency: float
    maximum_violation_magnitude: float
    mean_violation_magnitude: float
    mean_tracking_error: float
    worst_tracking_error: float
    mean_reward: float
    mean_episode_length: float
    execution_failures: int
    experiment_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"experiment_ids": list(self.experiment_ids)}


def evaluate_controller(
    controller_factory: Callable[[], Any], scenarios: list[Any], set_name: str, *, max_steps: int
) -> EvaluationSummary:
    if not scenarios:
        raise ValueError(f"Evaluation set {set_name!r} is empty")
    violations = []
    violation_magnitudes = []
    tracking = []
    rewards = []
    lengths = []
    failures = 0
    controller_id = "unknown"
    for scenario in scenarios:
        environment = PumpEnvironment()
        try:
            controller = controller_factory()
            controller_id = controller.id
            result = EvaluationRunner(max_steps=max_steps).run_episode(
                environment, controller, scenario, record_trace=True
            )
            infos = [item.get("info", {}) for item in result.trace]
            violations.append(any(item.get("constraint_violation", False) for item in infos))
            violation_magnitudes.append(
                max(
                    (float(item.get("violation_magnitude", 0.0)) for item in infos),
                    default=0.0,
                )
            )
            target = float(scenario.parameters.get("target_level", 0.5))
            tracking.extend(abs(float(item["level"]) - target) for item in infos if "level" in item)
            rewards.append(result.episode_reward)
            lengths.append(result.episode_length)
        except Exception:
            failures += 1
        finally:
            environment.close()
    completed = len(scenarios) - failures
    denominator = max(completed, 1)
    return EvaluationSummary(
        set_name,
        controller_id,
        len(scenarios),
        completed,
        sum(violations) / denominator if completed else 0.0,
        max(violation_magnitudes, default=0.0),
        sum(violation_magnitudes) / denominator if completed else 0.0,
        sum(tracking) / max(len(tracking), 1),
        max(tracking, default=0.0),
        sum(rewards) / denominator if completed else 0.0,
        sum(lengths) / denominator if completed else 0.0,
        failures,
    )
