"""Serializable evaluation result structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
