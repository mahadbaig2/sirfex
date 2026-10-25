"""Fixed-set controller evaluation and aggregate robustness metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from sirfex.core import EvaluationRunner
from sirfex.environments.pump_observations import MINIMAL, PumpObservationEnvironment
from sirfex.training.diagnostics import reward_components
from sirfex.training.rewards import LEGACY, components, reward_spec


@dataclass(frozen=True)
class EvaluationSummary:
    set_name: str
    controller_id: str
    scenarios: int
    episodes: int
    constraint_violation_frequency: float | None
    maximum_violation_magnitude: float | None
    mean_violation_magnitude: float | None
    mean_tracking_error: float | None
    worst_tracking_error: float | None
    mean_reward: float | None
    mean_episode_length: float | None
    execution_failures: int
    experiment_ids: tuple[str, ...] = ()
    full_horizon_frequency: float | None = None
    mean_duration_seconds: float | None = None
    episode_results: list[dict] = field(default_factory=list)
    action_statistics: dict = field(default_factory=dict)
    inference_mode: str = "deterministic"
    reward_schema: str = LEGACY

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"experiment_ids": list(self.experiment_ids)}


def evaluate_controller(
    controller_factory: Callable[[], Any],
    scenarios: list[Any],
    set_name: str,
    *,
    max_steps: int,
    observation_schema: str = MINIMAL,
    reward_schema: str = LEGACY,
    deterministic: bool = True,
) -> EvaluationSummary:
    reward_spec(reward_schema)
    if not scenarios:
        raise ValueError(f"Evaluation set {set_name!r} is empty")
    violations = []
    violation_magnitudes = []
    tracking = []
    rewards = []
    lengths = []
    failures = 0
    controller_id = "unknown"
    episode_results = []
    full_horizons, durations = [], []
    for scenario in scenarios:
        environment = PumpObservationEnvironment(observation_schema)
        row = {"scenario": scenario.to_dict(), "status": "execution_failed"}
        episode_results.append(row)
        try:
            controller = controller_factory()
            controller_id = controller.id
            result = EvaluationRunner(max_steps=max_steps).run_episode(
                environment, controller, scenario, record_trace=True, deterministic=deterministic
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
            errors = [abs(float(item["level"]) - target) for item in infos]
            tracking.append(sum(errors) / len(errors))
            rewards.append(result.episode_reward)
            lengths.append(result.episode_length)
            p = environment.validate_scenario(scenario)
            full_horizons.append(result.episode_length == p["horizon"] and not result.terminated)
            durations.append(result.episode_length * p["dt"])
            first = next(
                (x["step"] + 1 for x in result.trace if x["info"].get("constraint_violation")), None
            )
            parts = [
                reward_components(
                    x["info"]["level"],
                    target,
                    x["action"][0],
                    x["info"].get("violation_magnitude", 0),
                )
                for x in result.trace
            ]
            if reward_schema != LEGACY:
                parts = [
                    components(
                        reward_schema,
                        x["info"]["level"],
                        target,
                        x["action"][0],
                        x["info"].get("violation_magnitude", 0),
                        x["terminated"],
                    )
                    for x in result.trace
                ]
            row.update(
                status="completed",
                constraint_violation=violations[-1],
                maximum_violation_magnitude=violation_magnitudes[-1],
                mean_tracking_error=tracking[-1],
                maximum_tracking_error=max(errors),
                time_to_first_violation_seconds=first * p["dt"] if first else None,
                episode_reward=result.episode_reward,
                episode_length=result.episode_length,
                duration_seconds=durations[-1],
                full_horizon=full_horizons[-1],
                applied_actions=[x["action"][0] for x in result.trace],
                reward_components={
                    key: sum(x[key] for x in parts) for key in parts[0] if key != "reward_schema"
                },
            )
        except Exception as exc:
            failures += 1
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            environment.close()
    completed = len(scenarios) - failures
    successful = [row for row in episode_results if row["status"] == "completed"]
    violations = [r["constraint_violation"] for r in successful]
    violation_magnitudes = [r["maximum_violation_magnitude"] for r in successful]
    tracking = [r["mean_tracking_error"] for r in successful]
    rewards = [r["episode_reward"] for r in successful]
    lengths = [r["episode_length"] for r in successful]
    full_horizons = [r["full_horizon"] for r in successful]
    durations = [r["duration_seconds"] for r in successful]
    import numpy as np

    actions = np.array([a for r in successful for a in r["applied_actions"]])
    denominator = max(completed, 1)
    return EvaluationSummary(
        set_name,
        controller_id,
        len(scenarios),
        completed,
        sum(violations) / denominator if completed else None,
        max(violation_magnitudes, default=None),
        sum(violation_magnitudes) / denominator if completed else None,
        sum(tracking) / len(tracking) if tracking else None,
        max(
            (r["maximum_tracking_error"] for r in episode_results if r["status"] == "completed"),
            default=None,
        ),
        sum(rewards) / denominator if completed else None,
        sum(lengths) / denominator if completed else None,
        failures,
        full_horizon_frequency=sum(full_horizons) / denominator if completed else None,
        mean_duration_seconds=sum(durations) / denominator if completed else None,
        episode_results=episode_results,
        action_statistics={
            "count": len(actions),
            "lower_bound_fraction": float(np.mean(actions == 0)),
            "upper_bound_fraction": float(np.mean(actions == 1)),
            "interior_fraction": float(np.mean((actions > 0) & (actions < 1))),
            "mean": float(actions.mean()),
            "std": float(actions.std()),
        }
        if len(actions)
        else {},
        inference_mode="deterministic" if deterministic else "stochastic",
        reward_schema=reward_schema,
    )
