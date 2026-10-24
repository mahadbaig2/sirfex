"""Deterministic selection and mixing of normal and discovered scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sirfex.core.interfaces import Scenario
from sirfex.environments.pump import PumpEnvironment
from sirfex.search.persistence import SearchStorageError, SearchStore


@dataclass(frozen=True)
class SelectedDiscovery:
    discovery_id: str
    experiment_ids: tuple[str, ...]
    scenario: dict[str, Any]
    evidence_status: str
    replay_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"experiment_ids": list(self.experiment_ids)}


def select_discoveries(
    search_store: SearchStore, search_id: str, limit: int
) -> list[SelectedDiscovery]:
    if limit < 0:
        raise ValueError("discovery limit cannot be negative")
    try:
        loaded = search_store.load(search_id)
    except SearchStorageError as exc:
        raise ValueError(str(exc)) from exc
    selected = []
    environment = PumpEnvironment()
    try:
        for value in loaded["discoveries"]:
            if len(selected) >= limit:
                break
            if value.get("environment_id") != "pump":
                continue
            if value.get("evidence_status") != "observed":
                continue
            if value.get("replay_status") not in {"exact_match", "tolerance_match"}:
                continue
            scenario = Scenario.from_dict(value["scenario"])
            try:
                environment.validate_scenario(scenario)
            except Exception:
                continue
            selected.append(
                SelectedDiscovery(
                    value["discovery_id"],
                    tuple(value.get("experiment_ids", [])),
                    value["scenario"],
                    value["evidence_status"],
                    value["replay_status"],
                )
            )
    finally:
        environment.close()
    return selected


class ScenarioCurriculum:
    def __init__(
        self, normal: list[Scenario], stress: list[Scenario], ratio: float, seed: int
    ) -> None:
        if not normal:
            raise ValueError("Curriculum requires normal scenarios")
        if not 0 <= ratio <= 1:
            raise ValueError("stress ratio must be between 0 and 1")
        self.normal = normal
        self.stress = stress
        self.ratio = ratio
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def scenarios(self, count: int) -> list[Scenario]:
        if count < 1:
            raise ValueError("scenario count must be positive")
        result = []
        for _ in range(count):
            use_stress = bool(self.stress) and float(self._rng.random()) < self.ratio
            pool = self.stress if use_stress else self.normal
            result.append(pool[int(self._rng.integers(len(pool)))])
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "ratio": self.ratio,
            "seed": self.seed,
            "normal_scenarios": [item.to_dict() for item in self.normal],
            "stress_scenarios": [item.to_dict() for item in self.stress],
        }
