"""Fair, descriptive comparison of persisted search runs."""

from __future__ import annotations

from typing import Any

from sirfex.search.persistence import SearchStore


def compare_searches(store: SearchStore, first_id: str, second_id: str) -> dict[str, Any]:
    first = store.load(first_id)
    second = store.load(second_id)
    first_manifest = first["manifest"]
    second_manifest = second["manifest"]
    comparable = (
        first_manifest.get("environment_id") == second_manifest.get("environment_id")
        and first_manifest.get("objective_id") == second_manifest.get("objective_id")
        and first_manifest.get("trial_budget") == second_manifest.get("trial_budget")
        and first["space"] == second["space"]
    )
    return {
        "comparable": comparable,
        "explanation": (
            "Strategies use the same environment, objective, space, and budget."
            if comparable
            else "Search runs differ in environment, objective, space, or budget."
        ),
        "strategies": {
            "first": first_manifest.get("strategy"),
            "second": second_manifest.get("strategy"),
        },
        "first": first["summary"],
        "second": second["summary"],
    }
