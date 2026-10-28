"""Signed tank-constraint margins from authoritative recorded transitions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from sirfex.core import EpisodeResult, Scenario
from sirfex.environments.pump import PumpEnvironment
from sirfex.robustness.identity import scenario_identity

SCHEMA = "pump-signed-margin:1"
LEVEL_TOLERANCE = 1e-6  # float32 observations, float64 simulator info


class MarginEvidenceError(ValueError):
    """Trace is insufficient or internally inconsistent for a signed margin."""


@dataclass(frozen=True)
class PumpMargin:
    schema_version: str
    environment_id: str
    environment_version: str
    scenario_id: str
    controller_id: str
    initial_margin: float
    post_initial_minimum_margin: float
    full_episode_minimum_margin: float
    signed_change_from_initial: float
    critical_timestep: int
    post_initial_critical_timestep: int
    critical_boundary: str
    constraint_violated: bool
    maximum_violation_magnitude: float
    evaluated_transitions: int
    evidence_status: str
    warnings: tuple[str, ...]
    units: str = "normalized_tank_level_fraction"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["warnings"] = list(self.warnings)
        return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MarginEvidenceError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise MarginEvidenceError(f"{label} must be finite")
    return number


def _boundary(level: float, lo: float, hi: float) -> str:
    lower = abs(level - lo)
    upper = abs(hi - level)
    if math.isclose(lower, upper, rel_tol=0, abs_tol=1e-12):
        return "both"
    return "lower" if lower < upper else "upper"


def evaluate_margin(result: EpisodeResult | dict[str, Any]) -> PumpMargin:
    """Calculate M0, M_post and M_min; reject corrupt or missing evidence.

    A runner step-cap truncation receives `incomplete` status and cannot be used
    as a full-horizon survival witness. The margin itself still describes the
    observed prefix.
    """
    value = result.to_dict() if isinstance(result, EpisodeResult) else result
    if not isinstance(value, dict) or value.get("environment_id") != "pump":
        raise MarginEvidenceError("Expected a pump EpisodeResult")
    try:
        scenario = Scenario.from_dict(value["scenario"])
        params = PumpEnvironment().validate_scenario(scenario)
        trace = value["trace"]
        controller_id = value["controller_id"]
        episode_length = value["episode_length"]
    except (KeyError, TypeError, ValueError) as exc:
        raise MarginEvidenceError("Missing or invalid pump episode metadata") from exc
    if (
        not isinstance(trace, list)
        or not trace
        or type(episode_length) is not int
        or len(trace) != episode_length
        or value.get("execution_status", "completed") not in ("completed",)
        or not isinstance(controller_id, str)
        or not controller_id
    ):
        raise MarginEvidenceError("Trace is empty, incomplete, or not a completed execution")
    lo, hi = float(params["min_level"]), float(params["max_level"])
    first = trace[0]
    if not isinstance(first, dict) or not isinstance(first.get("observation"), list):
        raise MarginEvidenceError("Initial observation is missing")
    try:
        observed_initial = _number(first["observation"][0], "initial level")
    except (IndexError, KeyError) as exc:
        raise MarginEvidenceError("Initial level is missing") from exc
    initial = float(params["initial_level"])
    if abs(observed_initial - initial) > LEVEL_TOLERANCE:
        raise MarginEvidenceError("Initial observation conflicts with the scenario")
    if not lo - LEVEL_TOLERANCE <= initial <= hi + LEVEL_TOLERANCE:
        raise MarginEvidenceError("Initial state is outside operating limits")
    initial_margin = min(initial - lo, hi - initial)
    margins: list[tuple[float, int, str]] = [(initial_margin, 0, _boundary(initial, lo, hi))]
    violated = False
    maximum_violation = 0.0
    prior_post_level = None
    for index, row in enumerate(trace):
        if not isinstance(row, dict) or row.get("step") != index:
            raise MarginEvidenceError("Trace steps must be contiguous and ordered")
        observation = row.get("observation")
        info = row.get("info")
        if not isinstance(observation, list) or not observation or not isinstance(info, dict):
            raise MarginEvidenceError("Trace observation or physical info is missing")
        before = _number(observation[0], "pre-step level")
        if prior_post_level is not None and abs(before - prior_post_level) > LEVEL_TOLERANCE:
            raise MarginEvidenceError("Recorded levels are not transition-contiguous")
        try:
            level = _number(info["level"], "post-step level")
            recorded_lo = _number(info["min_level"], "lower limit")
            recorded_hi = _number(info["max_level"], "upper limit")
            magnitude = _number(info["violation_magnitude"], "violation magnitude")
            indicator = info["constraint_violation"]
        except KeyError as exc:
            raise MarginEvidenceError("Physical constraint evidence is missing") from exc
        if (
            abs(recorded_lo - lo) > 1e-12
            or abs(recorded_hi - hi) > 1e-12
            or not 0 <= level <= 1
            or magnitude < 0
            or type(indicator) is not bool
        ):
            raise MarginEvidenceError("Physical limits, level, or violation evidence conflict")
        observed_margin = min(level - lo, hi - level)
        if indicator:
            if magnitude <= 0 or level > lo + LEVEL_TOLERANCE and level < hi - LEVEL_TOLERANCE:
                raise MarginEvidenceError("Violation indicator lacks a plausible boundary crossing")
            boundary = "lower" if level <= lo + LEVEL_TOLERANCE else "upper"
            if magnitude + LEVEL_TOLERANCE < -observed_margin:
                raise MarginEvidenceError(
                    "Violation magnitude is smaller than the clipped evidence"
                )
            margin = -magnitude  # authoritative pre-clipping measurement
            violated = True
            maximum_violation = max(maximum_violation, magnitude)
        else:
            if magnitude != 0 or observed_margin < -LEVEL_TOLERANCE:
                raise MarginEvidenceError("Safe indicator conflicts with the physical measurement")
            margin = observed_margin
            boundary = _boundary(level, lo, hi)
        margins.append((margin, index + 1, boundary))
        prior_post_level = level
    if bool(value.get("terminated")) != violated:
        raise MarginEvidenceError("Episode termination conflicts with pump violations")
    post = min(margins[1:], key=lambda item: (item[0], item[1]))
    full = min(margins, key=lambda item: (item[0], item[1]))
    last_info = trace[-1]["info"]
    incomplete = last_info.get("termination_reason") == "runner_max_steps" or (
        not violated and episode_length < params["horizon"]
    )
    if not violated and not bool(value.get("truncated")) and not incomplete:
        raise MarginEvidenceError("Episode lacks terminal or truncation evidence")
    return PumpMargin(
        SCHEMA,
        "pump",
        PumpEnvironment.version,
        scenario_identity(scenario),
        controller_id,
        initial_margin,
        post[0],
        full[0],
        post[0] - initial_margin,
        full[1],
        post[1],
        full[2],
        violated,
        maximum_violation,
        len(trace),
        "incomplete" if incomplete else "valid",
        ("runner_step_cap_or_partial_horizon",) if incomplete else (),
    )
