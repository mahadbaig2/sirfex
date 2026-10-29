"""Semantic evidence hashing, interruption recovery and report regeneration."""

import copy
import json
import subprocess
import sys

import pytest

from sirfex.core import Scenario
from sirfex.robustness.identity import semantic_hash
from sirfex.robustness.report import build_report, render_markdown, verify_report
from sirfex.robustness.store import (
    RobustnessStorageError,
    RobustnessStore,
    _exclusive_lock,
    run_items,
)


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


def test_core_numerical_imports_do_not_load_optional_ml_or_agent_packages():
    code = (
        "import sys; import sirfex.robustness.margins, "
        "sirfex.robustness.disturbance, sirfex.robustness.differential; "
        "assert not {'torch','stable_baselines3','groq','langgraph','streamlit'} "
        "& set(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_completed_case_is_write_once_with_idempotent_retry_and_identity_checks(tmp_path):
    store = RobustnessStore(tmp_path)
    spec, anchor = items()[0]
    from sirfex.robustness.identity import scenario_identity
    from sirfex.robustness.store import controller_identity

    p = {
        **protocol(),
        "controllers": [{"identity_sha256": controller_identity(spec)}],
        "anchors": [scenario_identity(anchor)],
    }
    run_id = store.open_run(p)
    case_id = store.case_id(run_id, spec, anchor)
    record = {
        "schema_version": "pump-robustness-case:1",
        "case_id": case_id,
        "controller_identity": controller_identity(spec),
        "anchor_id": scenario_identity(anchor),
        "status": "completed",
        "result": {"x": 1},
    }
    first = store.write_case(run_id, case_id, record)
    assert store.write_case(run_id, case_id, record) == first
    assert (
        store.write_case(
            run_id, case_id, {**record, "result": {"x": 1}, "semantic_hash": "ignored"}
        )
        == first
    )
    with pytest.raises(RobustnessStorageError, match="immutable"):
        store.write_case(run_id, case_id, {**record, "result": {"x": 2}})
    assert store.read_case(run_id, case_id)["semantic_hash"] == first["semantic_hash"]
    for changed in (
        {**record, "controller_identity": "other"},
        {**record, "anchor_id": "other"},
    ):
        with pytest.raises(RobustnessStorageError, match="identity"):
            store.write_case(run_id, case_id, changed)
    with pytest.raises(RobustnessStorageError, match="incompatible"):
        store.open_run({**p, "grid": [0, 0.2]}, run_id=run_id)
    assert store.read_case(run_id, case_id)["semantic_hash"] == first["semantic_hash"]


def test_corrupt_case_and_interrupted_write_refuse_replacement(tmp_path, monkeypatch):
    store = RobustnessStore(tmp_path)
    spec, anchor = items()[0]
    run_id = store.open_run(protocol())
    case_id = store.case_id(run_id, spec, anchor)
    from sirfex.robustness.identity import scenario_identity
    from sirfex.robustness.store import controller_identity

    record = {
        "case_id": case_id,
        "controller_identity": controller_identity(spec),
        "anchor_id": scenario_identity(anchor),
        "status": "completed",
        "result": {"x": 1},
    }
    import sirfex.robustness.store as module

    original = module._atomic_json

    def interrupt(path, value):
        if path.name == f"{case_id}.json":
            raise OSError("interrupted before atomic replacement")
        return original(path, value)

    monkeypatch.setattr(module, "_atomic_json", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        store.write_case(run_id, case_id, record)
    assert store.read_case(run_id, case_id) is None
    monkeypatch.setattr(module, "_atomic_json", original)
    store.write_case(run_id, case_id, record)
    path = store.directory(run_id) / "cases" / f"{case_id}.json"
    path.write_text(path.read_text().replace('"x": 1', '"x": 2'))
    with pytest.raises(RobustnessStorageError, match="hash changed"):
        store.write_case(run_id, case_id, record)


def test_explicit_failed_recovery_and_concurrent_lock_refusal(tmp_path):
    store = RobustnessStore(tmp_path)
    spec, anchor = items()[0]
    run_id = store.open_run(protocol())
    case_id = store.case_id(run_id, spec, anchor)
    from sirfex.robustness.identity import scenario_identity
    from sirfex.robustness.store import controller_identity

    base = {
        "case_id": case_id,
        "controller_identity": controller_identity(spec),
        "anchor_id": scenario_identity(anchor),
        "status": "failed",
        "error_type": "RuntimeError",
    }
    store.write_case(run_id, case_id, base)
    finished = {**base, "status": "completed", "result": {"ok": True}}
    with pytest.raises(RobustnessStorageError, match="explicit recovery"):
        store.write_case(run_id, case_id, finished)
    store.write_case(run_id, case_id, finished, recovery=True)
    with _exclusive_lock(store.directory(run_id) / ".run.lock"):
        with pytest.raises(RobustnessStorageError, match="writer lock"):
            run_items(store, run_id, items()[:1], calculation)
    lock_path = store.directory(run_id) / "cases" / f"{case_id}.lock"
    with _exclusive_lock(lock_path):
        with pytest.raises(RobustnessStorageError, match="writer lock"):
            store.write_case(run_id, case_id, finished)
    import threading

    errors = []
    with _exclusive_lock(store.directory(run_id) / ".run.lock"):

        def competing_write():
            try:
                store.write_case(run_id, case_id, finished)
            except RobustnessStorageError as exc:
                errors.append(str(exc))

        competitor = threading.Thread(target=competing_write)
        competitor.start()
        competitor.join()
    assert len(errors) == 1 and "writer lock" in errors[0]


def test_checkpoint_scenario_and_protocol_change_case_identity(tmp_path):
    from sirfex.robustness.store import controller_identity

    store = RobustnessStore(tmp_path)
    first = store.open_run(protocol())
    spec, anchor = items()[0]
    ppo_one = {
        "id": "ppo",
        "config": {"checkpoint": {"sha256": "a" * 64, "path": "machine-one.zip"}},
    }
    ppo_two = {
        "id": "ppo",
        "config": {"checkpoint": {"sha256": "b" * 64, "path": "machine-one.zip"}},
    }
    moved = {"id": "ppo", "config": {"checkpoint": {"sha256": "a" * 64, "path": "machine-two.zip"}}}
    assert controller_identity(ppo_one) != controller_identity(ppo_two)
    assert controller_identity(ppo_one) == controller_identity(moved)
    assert store.case_id(first, ppo_one, anchor) != store.case_id(first, ppo_two, anchor)
    other_anchor = Scenario(
        "pump", parameters={**anchor.parameters, "demand": 0.6}, seed=anchor.seed
    )
    assert store.case_id(first, spec, anchor) != store.case_id(first, spec, other_anchor)
    changed = store.open_run({**protocol(), "grid": [0, 0.2]})
    assert store.case_id(first, spec, anchor) != store.case_id(changed, spec, anchor)
