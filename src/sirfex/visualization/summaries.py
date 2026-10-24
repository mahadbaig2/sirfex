"""Comparison and display-oriented evidence summaries."""

from __future__ import annotations

from typing import Any


def search_comparison(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    a, b = first["manifest"], second["manifest"]
    checks = {
        "environment": a.get("environment_id") == b.get("environment_id"),
        "controller": a.get("controller_id") == b.get("controller_id"),
        "objective": a.get("objective_id") == b.get("objective_id"),
        "space": first.get("space") == second.get("space"),
        "seed_policy": a.get("scenario_seeds") == b.get("scenario_seeds"),
    }
    return {
        "comparable": all(checks.values()),
        "checks": checks,
        "first": first["summary"],
        "second": second["summary"],
    }


def replay_label(status: str | None) -> str:
    return {
        "exact_match": "Exact match",
        "tolerance_match": "Tolerance match",
        "mismatch": "Mismatch",
        "unavailable": "Unavailable",
    }.get(status or "", "Not verified")
