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
    for summary in (
        normal_control,
        robust,
        held_out_control,
        held_out_robust,
        known_control,
        known_robust,
    ):
        if summary.get("execution_failures", 0) or any(
            summary.get(k) is None
            for k in (
                "mean_tracking_error",
                "constraint_violation_frequency",
                "maximum_violation_magnitude",
            )
        ):
            return {"classification": "inconclusive", "reason": "Incomplete execution evidence"}
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


def classify_matched_robustness(
    normal: dict[str, dict[str, Any]],
    robust: dict[str, dict[str, Any]],
    tolerances: dict[str, float],
) -> dict[str, Any]:
    """Phase 3.3 policy over the existing physical metrics, fixed before final outcomes."""
    required = ("nominal_holdout", "known_stress", "held_out_stress")
    if any(name not in normal or name not in robust for name in required):
        return {
            "classification": "inconclusive",
            "guardrails_pass": False,
            "reason": "Missing evaluation set",
        }
    for name in required:
        for result in (normal[name], robust[name]):
            if (
                result.get("episodes", 0) == 0
                or result.get("execution_failures", 0)
                or any(
                    result.get(key) is None
                    for key in (
                        "constraint_violation_frequency",
                        "maximum_violation_magnitude",
                        "mean_tracking_error",
                    )
                )
            ):
                return {
                    "classification": "inconclusive",
                    "guardrails_pass": False,
                    "reason": "Incomplete or failed controller evaluation",
                }
        if normal[name]["episodes"] != robust[name]["episodes"]:
            return {
                "classification": "inconclusive",
                "guardrails_pass": False,
                "reason": "Matched arms used different evaluation counts",
            }
    n_nom, r_nom = normal["nominal_holdout"], robust["nominal_holdout"]
    n_stress, r_stress = normal["held_out_stress"], robust["held_out_stress"]
    checks = {
        "nominal_tracking": _relative_change(
            n_nom["mean_tracking_error"], r_nom["mean_tracking_error"]
        )
        <= tolerances["nominal_tracking_error_degradation"],
        "nominal_violation_frequency": r_nom["constraint_violation_frequency"]
        <= n_nom["constraint_violation_frequency"]
        + tolerances["nominal_violation_frequency_degradation"],
        "held_out_frequency": r_stress["constraint_violation_frequency"]
        <= n_stress["constraint_violation_frequency"]
        + tolerances["held_out_violation_frequency_degradation"],
        "held_out_max_magnitude": r_stress["maximum_violation_magnitude"]
        <= n_stress["maximum_violation_magnitude"]
        + tolerances["held_out_violation_magnitude_degradation"],
        "execution_failures": all(
            robust[name]["execution_failures"]
            <= normal[name]["execution_failures"] + tolerances["execution_failure_increase"]
            for name in required
        ),
    }
    passed = all(checks.values())
    if not passed:
        classification = "regressed"
    elif (
        r_stress["constraint_violation_frequency"] < n_stress["constraint_violation_frequency"]
        or r_stress["maximum_violation_magnitude"] < n_stress["maximum_violation_magnitude"]
    ):
        classification = "improved"
    else:
        classification = "inconclusive"
    return {
        "classification": classification,
        "guardrails_pass": passed,
        "checks": checks,
        "tolerances": tolerances,
        "note": "Per-seed descriptive policy; overall inference uses training-seed pairs",
    }
