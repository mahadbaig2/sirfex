from pathlib import Path

import pytest

from sirfex.visualization.readers import EvidenceReader, EvidenceReadError
from sirfex.visualization.summaries import search_comparison
from sirfex.visualization.trajectories import (
    failure_events,
    lunarlander_trajectory,
    pump_trajectory,
    search_history,
)


def test_pump_mapping_uses_recorded_info_and_constraint_markers() -> None:
    points = pump_trajectory(
        [
            {
                "step": 0,
                "observation": [0.5, 0.5, 0.5],
                "action": [1.0],
                "info": {"level": 0.6, "constraint_violation": True},
            },
        ],
        {"parameters": {"target_level": 0.55, "min_level": 0.2, "max_level": 0.8}},
    )
    assert points[0]["level"] == 0.6
    assert points[0]["command"] == 1.0
    assert points[0]["constraint_violation"] is True
    assert points[0]["max_level"] == 0.8


def test_lunarlander_mapping_preserves_observation_fields() -> None:
    point = lunarlander_trajectory([{"step": 2, "observation": [1, 2, 3, 4, 5, 6, 1, 0]}])[0]
    assert point["x"] == 1.0
    assert point["y"] == 2.0
    assert point["vy"] == 4.0
    assert point["left_leg_contact"] == 1


def test_search_history_respects_minimization() -> None:
    history = search_history(
        [
            {"trial_index": 0, "status": "completed", "aggregate": {"mean": 4}},
            {"trial_index": 1, "status": "completed", "aggregate": {"mean": 2}},
            {"trial_index": 2, "status": "invalid", "aggregate": {}},
        ],
        "minimize",
    )
    assert [item["best"] for item in history] == [4, 2, 2]
    assert history[2]["value"] is None


def test_reader_loads_real_artifacts_and_rejects_unsafe_ids(tmp_path: Path) -> None:
    reader = EvidenceReader(tmp_path / "experiments", tmp_path / "searches")
    assert reader.list_experiments() == []
    with pytest.raises(EvidenceReadError):
        reader.load_experiment("../escape")


def test_failure_events_and_comparison_are_display_only() -> None:
    assert (
        failure_events({"failures": [{"category": "constraint_violation"}]})[0]["category"]
        == "constraint_violation"
    )
    first = {
        "manifest": {
            "environment_id": "pump",
            "controller_id": "x",
            "objective_id": "o",
            "scenario_seeds": [0],
        },
        "space": {},
        "summary": {},
    }
    second = {
        "manifest": {
            "environment_id": "pump",
            "controller_id": "x",
            "objective_id": "o",
            "scenario_seeds": [0],
        },
        "space": {},
        "summary": {},
    }
    assert search_comparison(first, second)["comparable"] is True
