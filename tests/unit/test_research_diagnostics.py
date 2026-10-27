"""Headless research diagnostics and storage boundaries."""

import numpy as np
import pytest

from sirfex.research.diagnostics import _action_comparison, _bin, _check_contract
from sirfex.research.store import ResearchStore


class ConstantController:
    def __init__(self, command):
        self.command = command

    def act(self, observation):
        return np.array([self.command], dtype=np.float32)


def test_fixed_state_comparison_distinguishes_normal_and_stress():
    record = {
        "observation": [0.4, 0.5, 0.6, 1.0, 0.8, 0.0],
        "partition": "development_nominal",
        "target_error": 0.1,
        "distance_to_min": 0.35,
        "distance_to_max": 0.55,
        "corrective_authority": 0.2,
        "demand": 0.6,
        "capacity": 0.8,
        "efficiency": 1.0,
        "initial_level": 0.4,
    }
    controllers = {
        "baseline": ConstantController(0.5),
        "normal": ConstantController(0.6),
        "expanded_5": ConstantController(0.7),
        "expanded_10": ConstantController(0.8),
    }
    result = _action_comparison([record, record], controllers)
    shift = result["shifts_from_baseline"]["normal_to_expanded_5"]
    assert shift["mean_signed_physical_action_shift"] == pytest.approx(0.1)
    assert result["state_conditioned"]["expanded_5:partition"]["development_nominal"]["count"] == 2
    assert _bin(record)["tracking_error"] == "below_target"


def test_store_rejects_unsafe_artifact_names_and_paths(tmp_path):
    store = ResearchStore(tmp_path)
    store.create("study_123", {"manifest.json": {"status": "prepared"}})
    assert store.read("study_123", "manifest.json")["status"] == "prepared"
    assert len(store.list()) == 1
    with pytest.raises(ValueError):
        store.directory("../escape")
    with pytest.raises(ValueError):
        store.write("study_123", "../escape.json", {})
    with pytest.raises(FileExistsError):
        store.create("study_123", {})


def test_checkpoint_contract_validation_rejects_mismatch():
    class Metadata:
        environment_id = "pump"
        observation_schema = "pump-calibrated:1"
        policy_action_schema = "pump-normalized:1"
        reward_schema = "pump-survival-tracking:2"
        termination_schema = "pump-terminate-on-violation:1"
        observation_shape = (6,)
        action_shape = (1,)

    _check_contract(Metadata())
    Metadata.reward_schema = "unsupported"
    with pytest.raises(ValueError):
        _check_contract(Metadata())
