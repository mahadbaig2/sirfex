"""Serializable evaluation result structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from sirfex.core.failures import ExecutionStatus, FailureRecord


@dataclass
class EpisodeResult:
    environment_id: str
    controller_id: str
    scenario: dict[str, Any]
    seed: int
    episode_reward: float
    episode_length: int
    terminated: bool
    truncated: bool
    outcome: str
    metrics: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    execution_status: ExecutionStatus = ExecutionStatus.COMPLETED
    observed_outcome: str = "unknown"
    failures: list[FailureRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["execution_status"] = self.execution_status.value
        value["failures"] = [failure.to_dict() for failure in self.failures]
        return value
