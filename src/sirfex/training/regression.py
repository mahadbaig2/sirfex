"""Explicit, configurable robustness regression classification."""

from __future__ import annotations

from typing import Any


def _relative_change(base: float, candidate: float) -> float:
    return (candidate - base) / max(abs(base), 1e-12)


def classify_regression(
    normal_control: dict[str, Any],
    robust: dict[str, Any],
    held_out_control: dict[str, Any],
    held_out_robust: dict[str, Any],
    known_control: dict[str, Any],
    known_robust: dict[str, Any],
    tolerances: dict[str, float],
) -> dict[str, Any]:
    checks = {
        "nominal_tracking": _relative_change(
            normal_control["mean_tracking_error"], robust["mean_tracking_error"]
        )
        <= tolerances.get("nominal_tracking_error_degradation", 0.10),
        "held_out_frequency": held_out_robust["constraint_violation_frequency"]
        <= held_out_control["constraint_violation_frequency"]
        + tolerances.get("held_out_violation_frequency_degradation", 0.0),
        "held_out_magnitude": held_out_robust["maximum_violation_magnitude"]
        <= held_out_control["maximum_violation_magnitude"]
        + tolerances.get("held_out_violation_magnitude_degradation", 0.0),
        "known_stress": known_robust["maximum_violation_magnitude"]
        < known_control["maximum_violation_magnitude"],
    }
    if all(checks.values()) and checks["known_stress"]:
        classification = "improved"
    elif (
        checks["nominal_tracking"] and checks["held_out_frequency"] and checks["held_out_magnitude"]
    ):
        classification = "inconclusive"
    else:
        classification = "regressed"
    return {"classification": classification, "checks": checks, "tolerances": tolerances}
