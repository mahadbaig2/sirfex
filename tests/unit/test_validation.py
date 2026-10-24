from dataclasses import replace

import numpy as np
import pytest

from sirfex.core import Scenario
from sirfex.environments.pump import PumpEnvironment
from sirfex.environments.pump_observations import EXTENDED, MINIMAL, PumpObservationEnvironment
from sirfex.training.diagnostics import reward_components
from sirfex.training.evaluation import evaluate_controller
from sirfex.training.feasibility import analyze_feasibility
from sirfex.training.protocol import (
    compare_paired,
    competence,
    content_hash,
    datasets,
    leakage_check,
    matched_budget,
    paired_interval,
    scenario_record,
)
from sirfex.training.pump_gym import PumpGymEnv
from sirfex.training.validation import validation_config


@pytest.mark.parametrize(
    "parameters,classification",
    [
        ({}, "target_feasible"),
        ({"demand": 0.8, "pump_capacity": 0.8}, "target_feasible"),
        ({"demand": 0.9, "pump_capacity": 0.8, "horizon": 2}, "bounded_operation_feasible"),
        ({"demand": 0.9, "pump_capacity": 0.8}, "infeasible"),
        ({"demand": 0.3, "pump_capacity": 0.4, "leak": 0.5}, "bounded_operation_feasible"),
        ({"dt": 1.0, "leak": 3.0, "pump_capacity": 3.0, "demand": 0.1}, "target_feasible"),
        ({"dt": 1.0, "leak": 3.0, "pump_capacity": 0.1}, "infeasible"),
    ],
)
def test_feasibility_distinguishes_target_interval_and_finite_horizon(parameters, classification):
    assert analyze_feasibility(Scenario("pump", parameters))["classification"] == classification


def test_feasibility_cap_invalid_and_exact_boundary():
    assert analyze_feasibility(Scenario("pump"), max_horizon=1)["classification"] == "undetermined"
    result = analyze_feasibility(Scenario("pump", {"demand": 0.9, "pump_capacity": 0.8}))
    assert result["first_unavoidable_violation_step"] == 46
    with pytest.raises(ValueError):
        analyze_feasibility(Scenario("pump", {"demand": float("nan")}))
    with pytest.raises(ValueError):
        analyze_feasibility(Scenario("pump"), max_horizon=0)


@pytest.mark.parametrize("schema,size", [(MINIMAL, 3), (EXTENDED, 6)])
def test_observation_changes_no_physics(schema, size):
    s = Scenario("pump", {"horizon": 30, "efficiency": 0.8, "leak": 0.01})
    original, adapted = PumpEnvironment(), PumpObservationEnvironment(schema)
    original.reset(s)
    obs, _ = adapted.reset(s)
    assert obs.shape == (size,)
    for _ in range(30):
        action = np.array([0.8], dtype=np.float32)
        before, after = original.step(action), adapted.step(action)
        np.testing.assert_array_equal(before[0], after[0][:3])
        assert before[1:] == after[1:]
        if before[2] or before[3]:
            break


def test_reward_decomposition_and_equal_episode_weighting():
    parts = reward_components(0.6, 0.5, 0.8, 0.02)
    assert parts["total"] == pytest.approx(-0.01 - 0.0064 - 0.2)
    from sirfex.controllers import PumpRuleController

    result = evaluate_controller(
        PumpRuleController,
        [Scenario("pump", {"horizon": 2}), Scenario("pump", {"horizon": 20})],
        "different-length",
        max_steps=20,
    ).to_dict()
    assert result["mean_tracking_error"] == pytest.approx(
        np.mean([r["mean_tracking_error"] for r in result["episode_results"]])
    )


def test_execution_failure_is_not_zero_physical_failure():
    def broken():
        raise ValueError("unavailable controller")

    result = evaluate_controller(broken, [Scenario("pump")], "test", max_steps=10).to_dict()
    assert result["constraint_violation_frequency"] is None
    assert result["mean_tracking_error"] is None
    assert result["execution_failures"] == 1
    assert not competence(result)["passed"]


def test_reproducible_disjoint_physical_partitions():
    first, second = datasets(count=3), datasets(count=3)
    assert content_hash({k: [scenario_record(s) for s in v] for k, v in first.items()}) == (
        content_hash({k: [scenario_record(s) for s in v] for k, v in second.items()})
    )
    assert leakage_check(first)["passed"]
    s = first["training"][0]
    near = replace(
        s,
        parameters={**s.parameters, "demand": s.parameters["demand"] + 1e-6, "horizon": 100},
        seed=s.seed + 1,
    )
    assert not leakage_check({"train": [s], "test": [near]})["passed"]


def test_curriculum_probability_applies_to_each_reset():
    normal, stress = Scenario("pump", seed=1), Scenario("pump", seed=2)
    env = PumpGymEnv([normal], stress_scenarios=[stress], stress_ratio=0.3, seed=10)
    for _ in range(1000):
        env.reset()
    assert 260 <= env.sample_counts["stress"] <= 340


def test_unmatched_or_partial_budgets_rejected():
    record = dict(
        starting_sha256="a",
        observation_schema=MINIMAL,
        reward_schema="r",
        ppo_config={},
        additional_timesteps=128,
        continuation_seed=3,
        evaluation_hash="e",
        status="completed",
    )
    assert matched_budget(record, record)["matched"]
    assert not matched_budget(record, {**record, "additional_timesteps": 129})["matched"]
    assert not matched_budget(record, {**record, "status": "partial"})["matched"]


def test_uncertainty_uses_training_seeds_and_guardrails():
    assert paired_interval([-0.1])["interval_95"] is None
    assert paired_interval([-0.1] * 3)["interval_95"] == pytest.approx([-0.1, -0.1])

    def metrics(violation, tracking=0.05):
        return dict(
            episodes=10,
            execution_failures=0,
            mean_tracking_error=tracking,
            constraint_violation_frequency=violation,
            maximum_violation_magnitude=violation,
        )

    pair = {
        "normal": {"nominal_holdout": metrics(0), "held_out_stress": metrics(0.5)},
        "robust": {"nominal_holdout": metrics(0), "held_out_stress": metrics(0.2)},
    }
    assert compare_paired([{**pair, "seed": 3}])["classification"] == "inconclusive"
    assert (
        compare_paired([{**pair, "seed": s} for s in (3, 17, 41)])["classification"] == "improved"
    )
    assert compare_paired([{**pair, "seed": 3}] * 3)["classification"] == "inconclusive"
    pair["robust"]["nominal_holdout"] = metrics(0, 0.2)
    assert (
        compare_paired([{**pair, "seed": s} for s in (3, 17, 41)])["classification"] == "regressed"
    )


@pytest.mark.parametrize(
    "config",
    [
        {"budgets": [129]},
        {"seeds": [3, 3]},
        {"profile": "extended"},
        {"wall_seconds": float("nan")},
        {"ppo": {"learning_rate": float("nan")}},
    ],
)
def test_reject_invalid_study_configuration(config):
    with pytest.raises(ValueError):
        validation_config(config)
