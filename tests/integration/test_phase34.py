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
