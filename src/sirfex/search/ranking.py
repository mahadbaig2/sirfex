"""Objective-local discovery ranking and near-duplicate grouping."""

from __future__ import annotations

from typing import Any

from sirfex.search.records import DiscoveryRecord


def similarity_key(parameters: dict[str, Any], precision: int = 5) -> str:
    normalized = {}
    for key in sorted(parameters):
        value = parameters[key]
        normalized[key] = round(value, precision) if isinstance(value, float) else value
    return repr(normalized)


def rank_discoveries(discoveries: list[DiscoveryRecord], direction: str) -> list[DiscoveryRecord]:
    unique: dict[str, DiscoveryRecord] = {}
    for discovery in discoveries:
        current = unique.get(discovery.similarity_key)
        if current is None:
            unique[discovery.similarity_key] = discovery
        else:
            current_value = current.objective.get("value", 0.0)
            new_value = discovery.objective.get("value", 0.0)
            if (direction == "maximize" and new_value > current_value) or (
                direction == "minimize" and new_value < current_value
            ):
                unique[discovery.similarity_key] = discovery
    return sorted(
        unique.values(),
        key=lambda item: item.objective.get("value", 0.0),
        reverse=direction == "maximize",
    )
