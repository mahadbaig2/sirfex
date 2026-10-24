"""Minimal, honest comparison of two saved experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from sirfex.experiments.storage import ExperimentStore


@dataclass
class ComparisonReport:
    comparable: bool
    explanation: str
    environment: str
    controller_a: str
    controller_b: str
    differences: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_experiments(store: ExperimentStore, first_id: str, second_id: str) -> ComparisonReport:
    first = store.load(first_id)
    second = store.load(second_id)
    same_environment = (
        first.manifest.environment_id == second.manifest.environment_id
        and first.manifest.environment_version == second.manifest.environment_version
    )
    differences = {
        "scenario": {
            "first": first.manifest.scenario,
            "second": second.manifest.scenario,
        },
        "seed": {
            "first": first.manifest.scenario.get("seed"),
            "second": second.manifest.scenario.get("seed"),
        },
        "episode_reward": {
            "first": first.result.get("episode_reward"),
            "second": second.result.get("episode_reward"),
            "delta": (
                second.result.get("episode_reward", 0.0) - first.result.get("episode_reward", 0.0)
            ),
        },
        "episode_length": {
            "first": first.result.get("episode_length"),
            "second": second.result.get("episode_length"),
            "delta": second.result.get("episode_length", 0) - first.result.get("episode_length", 0),
        },
        "outcome": {
            "first": first.result.get("outcome"),
            "second": second.result.get("outcome"),
        },
        "failures": {
            "first": first.result.get("failures", []),
            "second": second.result.get("failures", []),
        },
    }
    return ComparisonReport(
        comparable=same_environment,
        explanation=(
            "Experiments share an environment and version; differences are descriptive, "
            "not a universal robustness score."
            if same_environment
            else "Experiments use different environment versions and should not be treated "
            "as directly comparable."
        ),
        environment=first.manifest.environment_id,
        controller_a=first.manifest.controller_id,
        controller_b=second.manifest.controller_id,
        differences=differences,
    )
