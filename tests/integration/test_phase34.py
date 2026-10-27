"""Scientific gates for the staged Phase 3.4 workflow."""

import pytest

from sirfex.core import Scenario
from sirfex.training.feasibility import analyze_feasibility
from sirfex.training.phase34 import (
    STRATA,
    classify_final,
    configuration,
    generate_benchmark,
    physical_difficulty,
    select_candidate,
)
from sirfex.training.protocol import leakage_check


def _summary(mae, frequency=0.0, failures=0):
    return {
        "mean_tracking_error": mae,
        "constraint_violation_frequency": frequency,
        "execution_failures": failures,
        "episodes": 4,
        "scenarios": 4,
    }


def test_controller_independent_benchmark_is_feasible_and_disjoint():
    config = configuration({"profile": "smoke", "benchmark_seed": 77})
    config.update(
        development_stress_per_stratum=2,
        final_stress_per_stratum=2,
        final_nominal_count=2,
    )
    groups, evidence = generate_benchmark(config, [])
    second, duplicate = generate_benchmark(config, [])
    assert evidence == duplicate
    assert groups == second
    assert {key: len(value) for key, value in groups.items()} == {
        "screening_stress": 8,
        "final_stress": 8,
        "final_nominal": 2,
    }
    assert leakage_check(groups)["passed"]
    assert set(row["stratum"] for row in evidence["groups"]["final_stress"]) == set(STRATA)
    for rows in evidence["groups"].values():
        for row in rows:
            assert row["feasibility"]["finite_horizon_viable"] is True
            assert row["difficulty"]["initial_boundary_margin"] > 0
            assert row["difficulty"] == physical_difficulty(Scenario.from_dict(row["scenario"]))


def test_infeasible_and_unknown_are_not_claimed_feasible():
    config = configuration({"profile": "smoke"})
    config.update(
        development_stress_per_stratum=1, final_stress_per_stratum=1, final_nominal_count=1
    )
    groups, _ = generate_benchmark(config, [])
    scenario = groups["final_stress"][0]
    assert analyze_feasibility(scenario, max_horizon=1)["finite_horizon_viable"] is None
    assert analyze_feasibility(scenario)["finite_horizon_viable"] is True
    impossible = Scenario(
        "pump",
        {
            **scenario.parameters,
            "initial_level": 0.1,
            "demand": 1.0,
            "pump_capacity": 0.5,
            "efficiency": 0.6,
        },
        seed=99,
        environment_version="sirfex-pump:1",
    )
    assert analyze_feasibility(impossible)["classification"] == "infeasible"


def test_development_selection_requires_nominal_preservation_and_stress_gain():
    policy = {
        "nominal_relative_mae_max": 0.1,
        "nominal_violation_increase_max": 0,
        "stress_violation_increase_max": 0,
        "execution_failure_increase_max": 0,
        "development_stress_mae_gain_min": 0.005,
    }
    rows = [
        {
            "candidate_id": "normal",
            "development_nominal": _summary(0.04),
            "development_stress": _summary(0.12),
        },
        {
            "candidate_id": "worse_nominal",
            "development_nominal": _summary(0.06),
            "development_stress": _summary(0.08),
        },
        {
            "candidate_id": "no_stress_gain",
            "development_nominal": _summary(0.041),
            "development_stress": _summary(0.119),
        },
    ]
    assert select_candidate(rows, policy)["status"] == "no_eligible_curriculum"
    rows.append(
        {
            "candidate_id": "eligible",
            "development_nominal": _summary(0.043),
            "development_stress": _summary(0.10),
        }
    )
    result = select_candidate(rows, policy)
    assert result["status"] == "selected"
    assert result["selected_candidate_id"] == "eligible"
    assert [r["eligible"] for r in result["decisions"]] == [False, False, True]


def _final_eval(normal_mae, selected_mae, normal_stress, selected_stress):
    def rows(mae):
        return [{"constraint_violation": False, "mean_tracking_error": mae} for _ in range(4)]

    evaluation = {"seeds": {}}
    for seed in ("3", "17", "41"):
        evaluation["seeds"][seed] = {
            "normal": {
                "final_nominal": _summary(normal_mae),
                "final_stress": {**_summary(normal_stress), "episode_results": rows(normal_stress)},
            },
            "selected": {
                "final_nominal": _summary(selected_mae),
                "final_stress": {
                    **_summary(selected_stress),
                    "episode_results": rows(selected_stress),
                },
            },
        }
    return evaluation


def test_final_rule_uses_tracking_at_zero_frequency_floor_and_nominal_guardrail():
    policy = {
        "nominal_tracking_relative_max": 0.1,
        "nominal_violation_increase_max": 0,
        "execution_failure_increase_max": 0,
    }
    strata = list(STRATA)
    result = classify_final(_final_eval(0.04, 0.042, 0.12, 0.10), strata, policy)
    assert result["primary_frequency_floor"] is True
    assert result["classification"] == "improved"
    assert result["uncertainty"]["frequency"]["paired_differences"] == [0, 0, 0]
    assert (
        classify_final(_final_eval(0.04, 0.05, 0.12, 0.10), strata, policy)["classification"]
        == "regressed"
    )
    assert (
        classify_final(_final_eval(0.04, 0.042, 0.12, 0.12), strata, policy)["classification"]
        == "inconclusive"
    )


def test_extended_profile_requires_opt_in():
    with pytest.raises(ValueError):
        configuration({"profile": "extended"})


def test_tiny_actual_ppo_screen_persists_equal_start_and_stops_before_final(tmp_path, monkeypatch):
    pytest.importorskip("stable_baselines3")
    from sirfex.training import phase34
    from sirfex.training.config import PPOConfig
    from sirfex.training.matched_study import model_fingerprint
    from sirfex.training.persistence import TrainingStore
    from sirfex.training.ppo import train_ppo
    from sirfex.training.protocol import content_hash, scenario_record
    from sirfex.training.pump_gym import PumpGymEnv
    from sirfex.training.validation import _checkpoint

    store = TrainingStore(tmp_path / "training")
    normal = Scenario("pump", {"horizon": 200, "demand": 0.48}, seed=101)
    validation = Scenario("pump", {"horizon": 200, "demand": 0.52}, seed=102)
    stress = Scenario("pump", {"horizon": 200, "demand": 0.68}, seed=103)
    final_nominal = Scenario("pump", {"horizon": 200, "demand": 0.54}, seed=104)
    final_stress = Scenario("pump", {"horizon": 200, "demand": 0.72}, seed=105)
    ppo = PPOConfig(n_steps=128, batch_size=32, torch_threads=2)
    env = PumpGymEnv(
        [normal],
        observation_schema="pump-calibrated:1",
        policy_action_schema="pump-normalized:1",
        reward_schema="pump-survival-tracking:2",
    )
    model = train_ppo(env, ppo, seed=3, timesteps=128)
    env.close()
    source_id, study_id = "smokesource34", "smokestudy34"
    source_groups = {
        "training": [scenario_record(normal)],
        "validation": [scenario_record(validation)],
    }
    store.save(
        source_id,
        {
            "manifest.json": {
                "schema_version": "2",
                "training_run_id": source_id,
                "status": "completed",
            },
            "dataset_partitions.json": {
                "groups": source_groups,
                "hashes": {k: content_hash(v) for k, v in source_groups.items()},
            },
        },
    )
    source_dir = store.directory(source_id)
    meta = _checkpoint(
        model,
        source_dir,
        source_id,
        "baseline",
        {
            "ppo": ppo.to_dict(),
            "policy_action_schema": "pump-normalized:1",
            "reward_schema": "pump-survival-tracking:2",
        },
        3,
        "pump-calibrated:1",
    )
    monkeypatch.setattr(phase34, "SOURCE_32", source_id)
    benchmark_groups = {
        "screening_stress": [scenario_record(stress)],
        "final_nominal": [scenario_record(final_nominal)],
        "final_stress": [scenario_record(final_stress)],
    }
    benchmark = {
        "groups": benchmark_groups,
        "hashes": {k: content_hash(v) for k, v in benchmark_groups.items()},
    }
    policy = {
        "nominal_relative_mae_max": 0.1,
        "nominal_violation_increase_max": 0,
        "stress_violation_increase_max": 0,
        "execution_failure_increase_max": 0,
        "development_stress_mae_gain_min": 1000.0,
    }
    candidates = [
        {"id": "normal", "pool": "none", "stress_ratio": 0.0},
        {"id": "original_5", "pool": "original", "stress_ratio": 0.05},
    ]
    protocol = {
        "benchmark_hashes": benchmark["hashes"],
        "development_validation_hash": content_hash(source_groups["validation"]),
        "training_hash": content_hash(source_groups["training"]),
    }
    store.save(
        study_id,
        {
            "manifest.json": {
                "schema_version": "2",
                "phase": "3.4",
                "training_run_id": study_id,
                "status": "prepared",
                "protocol_sha256": content_hash(protocol),
                "preparation_seconds": 0,
                "checkpoints": {},
            },
            "config.json": {"wall_seconds": 300, "screen_steps": 128},
            "protocol.json": protocol,
            "stress_benchmark.json": benchmark,
            "baseline_verification.json": {
                "3": {"checkpoint": meta.to_dict(), "fingerprint": model_fingerprint(model)}
            },
            "candidate_curricula.json": {
                "screening": {"candidates": candidates, "policy": policy},
                "original": {"old": {"scenario": stress.to_dict()}},
                "new": {},
            },
            "development_results.json": {"calibration": {}},
        },
    )
    result = phase34.screen(study_id, store)
    assert result["status"] == "no_eligible_curriculum"
    assert result["actual_screening_steps"] == 256
    saved = store.load(study_id)
    runs = saved["development_results"]["screening_runs"]
    assert len(runs) == 2
    assert runs[0]["starting_fingerprint"] == runs[1]["starting_fingerprint"]
    assert all(r["actual_additional_steps"] == 128 for r in runs)
    assert "final_evaluation" not in saved
    assert "continuation_runs" not in saved
    with pytest.raises(ValueError, match="selected"):
        phase34.confirm(study_id, store, None)
