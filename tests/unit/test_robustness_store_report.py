"""Semantic evidence hashing, interruption recovery and report regeneration."""

import copy
import json
import subprocess
import sys

import pytest

from sirfex.core import Scenario
from sirfex.robustness.identity import semantic_hash
from sirfex.robustness.report import build_report, render_markdown, verify_report
from sirfex.robustness.store import RobustnessStorageError, RobustnessStore, run_items


def protocol():
    return {
        "schema_version": "pump-robustness-protocol:1",
        "profile": "software_calibration_only",
        "grid": [0, 0.1],
    }


def items():
    spec = {"id": "pump-fixed", "config": {"command": 0.5}}
    return [
        (spec, Scenario("pump", parameters={"horizon": 2, "demand": demand}, seed=i))
        for i, demand in enumerate((0.3, 0.4, 0.5), start=1)
    ]


def calculation(spec, scenario):
    return {"censored": False, "demand": scenario.parameters["demand"], "controller": spec["id"]}


def test_forced_interruption_resumes_to_uninterrupted_semantic_hash(tmp_path):
    interrupted = RobustnessStore(tmp_path / "interrupted")
    run_id = interrupted.open_run(protocol(), run_id="checkpoint1_test")
    calls = 0

    def abort_once(spec, scenario):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("forced interruption")
        return calculation(spec, scenario)

    with pytest.raises(RuntimeError, match="forced interruption"):
        run_items(interrupted, run_id, items(), abort_once)
    status = json.loads((interrupted.directory(run_id) / "status.json").read_text())
    assert status["counts"]["completed"] == 1
    assert status["counts"]["failed"] == 1
    assert status["counts"]["pending"] == 1
    remaining_calls = 0

    def continue_run(spec, scenario):
        nonlocal remaining_calls
        remaining_calls += 1
        return calculation(spec, scenario)

    resumed = run_items(interrupted, run_id, items(), continue_run)
    assert remaining_calls == 2
    assert resumed["reused_completed"] == 1
    assert resumed["completed"] == 3
    uninterrupted = RobustnessStore(tmp_path / "uninterrupted")
    other_id = uninterrupted.open_run(protocol(), run_id="checkpoint1_test")
    fresh = run_items(uninterrupted, other_id, items(), calculation)
    assert resumed["semantic_results_sha256"] == fresh["semantic_results_sha256"]


def test_protocol_mismatch_duplicate_and_corrupt_case_are_rejected(tmp_path):
    store = RobustnessStore(tmp_path)
    run_id = store.open_run(protocol(), run_id="checkpoint1_test")
    changed = {**protocol(), "grid": [0, 0.2]}
    with pytest.raises(RobustnessStorageError, match="incompatible"):
        store.open_run(changed, run_id=run_id)
    with pytest.raises(RobustnessStorageError, match="Duplicate"):
        run_items(store, run_id, items()[:1] * 2, calculation)
    run_items(store, run_id, items()[:1], calculation)
    case_id = store.case_id(run_id, *items()[0])
    path = store.directory(run_id) / "cases" / f"{case_id}.json"
    value = json.loads(path.read_text())
    value["result"]["demand"] = 999
    path.write_text(json.dumps(value))
    with pytest.raises(RobustnessStorageError, match="hash changed"):
        store.read_case(run_id, case_id)


def sample_report():
    margin = {
        "schema_version": "pump-signed-margin:1",
        "environment_id": "pump",
        "controller_id": "pump-fixed",
        "scenario_id": "scenario-id",
        "initial_margin": 0.3,
        "full_episode_minimum_margin": 0.2,
        "post_initial_minimum_margin": 0.2,
        "signed_change_from_initial": -0.1,
        "critical_timestep": 1,
        "critical_boundary": "lower",
        "constraint_violated": False,
        "evidence_status": "valid",
    }
    cell = {
        "schema_version": "pump-disturbance-cell:1",
        "controller_id": "pump-fixed",
        "direction": ["demand"],
        "status": "censored_max",
        "delta_first": None,
        "delta_infeasible_grid": None,
        "relative_failure_threshold": None,
        "nonmonotone": False,
        "physics_nonmonotone": False,
        "requested_grid_points": 2,
        "evaluated_grid_points": 2,
    }
    return build_report(
        profile="synthetic",
        protocol=protocol(),
        controllers=[{"id": "pump-fixed", "checkpoint_sha256": None}],
        scenarios=[{"scenario_id": "scenario-id", "scenario": {"seed": 1}}],
        margin_results=[margin],
        disturbance_cells=[cell],
        differential_results=[],
        reproducibility={"git_commit": None, "command": "test fixture", "replay_status": "not_run"},
    )


def test_report_is_deterministic_json_authoritative_and_null_explicit():
    report = sample_report()
    assert report == sample_report()
    markdown = render_markdown(report)
    assert markdown == render_markdown(report)
    assert "censored_max" in markdown
    assert "—" in markdown
    assert "No qualifying differential finding" in markdown
    assert report["semantic_sha256"] == semantic_hash(
        {key: value for key, value in report.items() if key != "semantic_sha256"}
    )
    tampered = copy.deepcopy(report)
    tampered["margin_results"][0]["initial_margin"] = 0.99
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_report(tampered)


def test_core_margin_import_does_not_load_optional_ml_or_agent_packages():
    code = (
        "import sys; import sirfex.robustness.margins; "
        "assert not {'torch','stable_baselines3','groq','langgraph','streamlit'} "
        "& set(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
