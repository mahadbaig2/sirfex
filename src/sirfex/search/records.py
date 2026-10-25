"""Serializable search trial and discovery records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from sirfex.search.objectives import RiskScore


@dataclass
class TrialRecord:
    trial_index: int
    status: str
    candidate_parameters: dict[str, Any]
    scenario_seeds: list[int]
    experiment_ids: list[str] = field(default_factory=list)
    risk_scores: list[dict[str, Any]] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    error: dict[str, str] | None = None
    elapsed_seconds: float = 0.0
    candidate_analysis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiscoveryRecord:
    discovery_id: str
    search_id: str
    environment_id: str
    controller_id: str
    scenario: dict[str, Any]
    objective: dict[str, Any]
    observed_failure: bool
    evidence_status: str
    experiment_ids: list[str]
    replay_status: str = "unverified"
    discovered_at_utc: str = ""
    similarity_key: str = ""
    held_out: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_scores(scores: list[RiskScore]) -> dict[str, Any]:
    if not scores:
        return {}
    values = [score.value for score in scores]
    direction = scores[0].direction
    mean = sum(values) / len(values)
    worst = max(values) if direction == "maximize" else min(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": mean,
        "worst_case": worst,
        "stddev": variance**0.5,
        "failure_frequency": sum(score.is_failure for score in scores) / len(scores),
        "direction": direction,
        "objective_id": scores[0].objective_id,
    }
