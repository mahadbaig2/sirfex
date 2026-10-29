"""Selected differential findings must have durable, independently checked episodes."""

import copy
import json

import pytest

from sirfex.core import EvaluationRunner, Scenario
from sirfex.experiments.factories import make_controller, make_environment
from sirfex.experiments.replay import compare_results, replay_experiment
from sirfex.experiments.storage import ExperimentStore
from sirfex.robustness.differential import classify_pair
from sirfex.robustness.evidence import (
    EvidenceArchiveError,
    _validate_stored,
    archive_selected_findings,
)
from sirfex.robustness.identity import scenario_identity
from sirfex.robustness.margins import evaluate_margin
from sirfex.training.checkpoints import verify_checkpoint

SCENARIO = Scenario(
    "pump",
    parameters={
        "dt": 0.1,
        "horizon": 3,
        "initial_level": 0.11,
        "target_level": 0.5,
        "demand": 0.5,
        "pump_capacity": 1.0,
        "efficiency": 1.0,
        "leak": 0.02,
        "min_level": 0.05,
        "max_level": 0.95,
    },
    seed=370111,
)
SPECS = [{"id": "pump-fixed", "config": {"command": 0.0}}, {"id": "pump-rule", "config": {}}]


def evaluate(controller_id, scenario):
    environment = make_environment("pump")
    spec = next(value for value in SPECS if value["id"] == controller_id)
    try:
        controller = make_controller(controller_id, environment, spec["config"])
        return EvaluationRunner(scenario.parameters["horizon"]).run_episode(
            environment, controller, scenario, record_trace=True
        )
    finally:
        environment.close()


def candidate():
    x = evaluate("pump-fixed", SCENARIO)
    y = evaluate("pump-rule", SCENARIO)
    mx, my = evaluate_margin(x), evaluate_margin(y)
    finding = classify_pair(
        mx, my, finite_horizon_viable=True, replay_statuses=("exact_match", "exact_match")
    )
    assert finding.classification == "class_b_unverified"
    row = {
        "status": "evaluated",
        "scenario_id": scenario_identity(SCENARIO),
        "scenario": SCENARIO.to_dict(),
        "margin_x": mx.to_dict(),
        "margin_y": my.to_dict(),
        "finding": finding.to_dict(),
    }
    return {
        "records": [row],
        "controller_x": "pump-fixed",
        "controller_y": "pump-rule",
        "findings": {"class_b": [], "class_a": [], "class_b_unverified": [finding.to_dict()]},
    }


def test_selected_class_b_has_four_persisted_trajectories_and_independent_replay(tmp_path):
    result = archive_selected_findings(candidate(), SPECS, evaluate, tmp_path, "a" * 64)
    finding = result["findings"]["class_b"][0]
    evidence = finding["durable_evidence"]
    assert evidence["verification_status"] == "independently_replayed"
    assert evidence["candidate_margin_match"]
    assert set(evidence["experiments"]) == {"x_original", "x_repeat", "y_original", "y_repeat"}
    assert set(evidence["persisted_repeat_statuses"].values()) == {"exact_match"}
    assert set(evidence["independent_regeneration_statuses"].values()) == {"exact_match"}
    store = ExperimentStore(tmp_path / "experiments")
    for saved in evidence["experiments"].values():
        stored = store.load(saved["experiment_id"])
        assert len(stored.trajectory) == stored.result["episode_length"]
        assert replay_experiment(store, saved["experiment_id"]).status == "exact_match"
    # Rerunning the archive is idempotent and retains stable artifact IDs.
    repeated = archive_selected_findings(candidate(), SPECS, evaluate, tmp_path, "a" * 64)
    assert repeated["findings"]["class_b"][0]["durable_evidence"] == evidence


def test_class_a_remains_safe_near_miss_with_separate_durable_status(tmp_path):
    scenario = Scenario(
        "pump", parameters={**SCENARIO.parameters, "initial_level": 0.35}, seed=371112
    )
    x, y = evaluate("pump-fixed", scenario), evaluate("pump-rule", scenario)
    mx, my = evaluate_margin(x), evaluate_margin(y)
    finding = classify_pair(mx, my, finite_horizon_viable=True, near_miss_delta=0.05)
    assert finding.classification == "class_a"
    assert not mx.constraint_violated and not my.constraint_violated
    row = {
        "status": "evaluated",
        "scenario_id": scenario_identity(scenario),
        "scenario": scenario.to_dict(),
        "margin_x": mx.to_dict(),
        "margin_y": my.to_dict(),
        "finding": finding.to_dict(),
    }
    result = {
        "records": [row],
        "controller_x": "pump-fixed",
        "controller_y": "pump-rule",
        "findings": {"class_b": [], "class_a": [finding.to_dict()], "class_b_unverified": []},
    }
    archived = archive_selected_findings(result, SPECS, evaluate, tmp_path, "e" * 64)
    saved = archived["findings"]["class_a"][0]
    assert saved["classification"] == "class_a"
    assert saved["durable_evidence"]["verification_status"] == "independently_replayed"
    assert len(saved["durable_evidence"]["experiments"]) == 4


def test_corrupt_archived_scenario_or_trace_is_rejected(tmp_path):
    result = archive_selected_findings(candidate(), SPECS, evaluate, tmp_path, "b" * 64)
    saved = result["findings"]["class_b"][0]["durable_evidence"]["experiments"]["x_original"]
    store = ExperimentStore(tmp_path / "experiments")
    directory = store._experiment_dir(saved["experiment_id"])
    scenario_path = directory / "scenario.json"
    original = scenario_path.read_text()
    changed = json.loads(original)
    changed["parameters"]["demand"] = 0.4
    scenario_path.write_text(json.dumps(changed))
    with pytest.raises(EvidenceArchiveError, match="identity"):
        _validate_stored(
            store,
            saved["experiment_id"],
            scenario_identity(SCENARIO),
            "pump-fixed",
            saved["episode_sha256"],
        )
    scenario_path.write_text(original)
    trace_path = directory / "trajectory.jsonl"
    trace_path.write_text(trace_path.read_text() + "{}\n")
    with pytest.raises(EvidenceArchiveError, match="hash"):
        _validate_stored(
            store,
            saved["experiment_id"],
            scenario_identity(SCENARIO),
            "pump-fixed",
            saved["episode_sha256"],
        )
    trace_path.unlink()
    with pytest.raises(EvidenceArchiveError, match="missing or corrupt"):
        _validate_stored(
            store,
            saved["experiment_id"],
            scenario_identity(SCENARIO),
            "pump-fixed",
            saved["episode_sha256"],
        )


def test_repeat_mismatch_downgrades_class_b(tmp_path):
    calls = 0

    def changing(controller_id, scenario):
        nonlocal calls
        calls += 1
        episode = evaluate(controller_id, scenario)
        if calls == 2:  # X repeat differs while keeping valid physical trace.
            episode.episode_reward += 1
        return episode

    result = archive_selected_findings(candidate(), SPECS, changing, tmp_path, "c" * 64)
    assert result["findings"]["class_b"] == []
    assert (
        result["findings"]["class_b_unverified"][0]["durable_evidence"]["verification_status"]
        == "unverified"
    )


def test_missing_or_wrong_checkpoint_hash_fails_verification(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_checkpoint(tmp_path / "absent.zip", "0" * 64)
    path = tmp_path / "checkpoint.zip"
    path.write_bytes(b"not a real checkpoint")
    with pytest.raises(ValueError, match="hash"):
        verify_checkpoint(path, "0" * 64)


def test_compare_results_distinguishes_exact_tolerance_and_mismatch():
    original = evaluate("pump-rule", SCENARIO).to_dict()
    assert compare_results(original, original).status == "exact_match"
    close = copy.deepcopy(original)
    close["episode_reward"] += 1e-6
    assert compare_results(original, close).status == "tolerance_match"
    far = copy.deepcopy(original)
    far["episode_reward"] += 1
    assert compare_results(original, far).status == "mismatch"


def test_interrupted_archive_is_not_a_valid_experiment(tmp_path, monkeypatch):
    original_write = ExperimentStore._atomic_write

    def interrupt(path, content):
        if path.name == "trajectory.jsonl":
            raise OSError("simulated archive interruption")
        return original_write(path, content)

    monkeypatch.setattr(ExperimentStore, "_atomic_write", staticmethod(interrupt))
    with pytest.raises(OSError, match="interruption"):
        archive_selected_findings(candidate(), SPECS, evaluate, tmp_path, "d" * 64)
    directories = list((tmp_path / "experiments").iterdir())
    assert len(directories) == 1
    assert (directories[0] / ".incomplete").exists()
    store = ExperimentStore(tmp_path / "experiments")
    with pytest.raises(Exception, match="incomplete"):
        store.load(directories[0].name)
