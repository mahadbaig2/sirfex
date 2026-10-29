"""Bounded synthetic fixed-scenario and shared-grid integration."""

from __future__ import annotations

import json

from sirfex.core import Scenario
from sirfex.robustness.identity import scenario_identity, semantic_hash
from sirfex.robustness.study import build_plan
from sirfex.robustness.study_execute import StudyEngine
from sirfex.robustness.study_store import StudyBudgetExhausted


def synthetic_package():
    nominal = Scenario(
        "pump",
        {
            "dt": 0.1,
            "horizon": 20,
            "min_level": 0.05,
            "max_level": 0.95,
            "target_level": 0.5,
            "initial_level": 0.5,
            "demand": 0.45,
            "pump_capacity": 0.8,
            "efficiency": 0.95,
            "leak": 0.01,
        },
        seed=912001,
        environment_version="sirfex-pump:1",
        provenance={"partition": "synthetic_nominal"},
    )
    stress = Scenario(
        "pump",
        {
            "dt": 0.1,
            "horizon": 20,
            "min_level": 0.05,
            "max_level": 0.95,
            "target_level": 0.5,
            "initial_level": 0.5,
            "demand": 0.55,
            "pump_capacity": 0.8,
            "efficiency": 0.95,
            "leak": 0.02,
        },
        seed=912002,
        environment_version="sirfex-pump:1",
        provenance={"partition": "synthetic_stress"},
    )
    manifest = {
        "schema_version": "synthetic-study-manifest:1",
        "scope": "synthetic",
        "scenarios": [
            {"partition": name, "scenario_id": scenario_identity(s), "scenario": s.to_dict()}
            for name, s in (("synthetic_nominal", nominal), ("synthetic_stress", stress))
        ],
    }
    manifest["semantic_sha256"] = semantic_hash(manifest)
    protocol = {
        "schema_version": "pump-robustness-synthetic:1",
        "profile": "synthetic_test_only",
        "scope": "synthetic",
        "status": "SYNTHETIC TEST FIXTURE",
        "executable": False,
        "scenario_manifest_sha256": manifest["semantic_sha256"],
        "controller_inventory": [
            {"controller_id": "pump-rule", "config": {}},
            {"controller_id": "pump-analytical", "config": {}},
        ],
        "anchors": [scenario_identity(nominal)],
        "disturbance": {"directions": [["demand"]], "grid": [0.0, 0.05, 0.1]},
        "differential_pairs": [
            {"id": "synthetic_pair", "x": "pump-rule", "y": "pump-analytical", "class": "synthetic"}
        ],
        "search": {
            "search_seed": 991,
            "scenario_seed_start": 912100,
            "proposals_per_pair": 1,
            "class_a_margin_gap": 0.05,
            "top_per_class": 1,
        },
        "budget": {
            "wall_seconds_cap": 300,
            "per_ppo_seconds_cap": 30,
            "artifact_bytes_cap": 10_000_000,
        },
    }
    protocol["semantic_sha256"] = semantic_hash(protocol)
    return protocol, manifest


def test_synthetic_fixed_grid_resume_and_prepared_identity(tmp_path):
    protocol, manifest = synthetic_package()
    plan = build_plan(protocol, manifest, tmp_path, synthetic=True)
    assert plan["work_counts"]["fixed_stress"] == 2
    assert plan["work_counts"]["grid"] == 2
    engine = StudyEngine(protocol, manifest, plan, tmp_path, tmp_path, synthetic=True)
    try:
        try:
            engine.execute_fixed_and_grid(interrupt_after=3)
        except StudyBudgetExhausted as exc:
            assert str(exc) == "synthetic_interruption"
        else:
            raise AssertionError("Expected bounded synthetic interruption")
        assert engine.store.status()["stages"]["fixed_stress"]["completed"] == 2
    finally:
        engine.close()
    resumed = StudyEngine(protocol, manifest, plan, tmp_path, tmp_path, synthetic=True)
    try:
        status = resumed.execute_fixed_and_grid(resume=True)
        assert status["stages"]["grid"]["completed"] + status["stages"]["grid"]["censored"] == 2
        for item in plan["work_items"]:
            if item["stage"] == "grid":
                evidence = resumed.store.read(item["work_id"])["result"]
                assert evidence["prepared_grid_sha256"]
                assert evidence["nominal"] is not None
                assert evidence["cell"]["requested_grid_points"] == 3
        prepared = json.loads((resumed.store.directory / "prepared_grids.json").read_text())
        assert prepared["plan_sha256"] == plan["semantic_sha256"]
    finally:
        resumed.close()
