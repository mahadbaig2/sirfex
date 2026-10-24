"""Environment-specific risk objectives with explicit evidence semantics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from sirfex.core import EpisodeResult


@dataclass(frozen=True)
class RiskScore:
    objective_id: str
    value: float
    direction: str
    units: str
    evidence_source: str
    evidence_status: str
    is_failure: bool = False
    threshold: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RiskObjective(Protocol):
    identifier: str
    direction: str

    def evaluate(self, result: EpisodeResult) -> RiskScore: ...


class PumpConstraintObjective:
    identifier = "pump.constraint_violation"
    direction = "maximize"

    def evaluate(self, result: EpisodeResult) -> RiskScore:
        violations = [
            step.get("info", {})
            for step in result.trace
            if step.get("info", {}).get("constraint_violation")
        ]
        magnitudes = [float(item.get("violation_magnitude", 0.0)) for item in violations]
        value = max(magnitudes, default=0.0)
        return RiskScore(
            self.identifier,
            value,
            self.direction,
            "normalized level fraction",
            "pump environment constraint evidence",
            "observed",
            bool(violations),
            0.0,
            {"violation_count": len(violations), "max_violation_magnitude": value},
        )


class PumpTrackingObjective:
    identifier = "pump.tracking_degradation"
    direction = "maximize"

    def evaluate(self, result: EpisodeResult) -> RiskScore:
        errors = []
        for step in result.trace:
            observation = step.get("observation", [])
            if len(observation) >= 2:
                errors.append(abs(float(observation[0]) - float(observation[1])))
        value = sum(errors) / len(errors) if errors else 0.0
        return RiskScore(
            self.identifier,
            value,
            self.direction,
            "normalized level fraction",
            "pump trajectory tracking metric",
            "inferred",
            value > 0.1,
            0.1,
            {"mean_absolute_error": value, "max_absolute_error": max(errors, default=0.0)},
        )


class LunarLanderRiskObjective:
    identifier = "lunarlander.terminal_instability_proxy"
    direction = "maximize"

    def evaluate(self, result: EpisodeResult) -> RiskScore:
        vertical = []
        horizontal = []
        angle = []
        for step in result.trace:
            observation = step.get("observation", [])
            if len(observation) >= 6:
                horizontal.append(abs(float(observation[2])))
                vertical.append(abs(float(observation[3])))
                angle.append(abs(float(observation[4])))
        value = max(
            (v + 0.5 * h + 0.5 * a for v, h, a in zip(vertical, horizontal, angle, strict=False)),
            default=0.0,
        )
        if not math.isfinite(value):
            value = 0.0
        return RiskScore(
            self.identifier,
            value,
            self.direction,
            "observation-derived proxy units",
            "LunarLander observation risk proxy",
            "inferred",
            False,
            None,
            {
                "max_vertical_speed": max(vertical, default=0.0),
                "max_horizontal_speed": max(horizontal, default=0.0),
                "max_abs_angle": max(angle, default=0.0),
            },
        )


def make_objective(identifier: str) -> RiskObjective:
    objectives = {
        PumpConstraintObjective.identifier: PumpConstraintObjective,
        PumpTrackingObjective.identifier: PumpTrackingObjective,
        LunarLanderRiskObjective.identifier: LunarLanderRiskObjective,
    }
    try:
        return objectives[identifier]()
    except KeyError as exc:
        raise ValueError(f"Unsupported risk objective: {identifier}") from exc
