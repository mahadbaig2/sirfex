"""Hand-computable signed-margin checks and one genuine local development trace."""

from pathlib import Path

import pytest

from sirfex.controllers.analytical_pump import FixedPumpController
from sirfex.core import EvaluationRunner, Scenario
from sirfex.environments.pump import PumpEnvironment
from sirfex.experiments.storage import ExperimentStore
from sirfex.robustness import MarginEvidenceError, evaluate_margin


def fixed_episode(*, command=0.0, **parameters):
    values = {
        "horizon": 1,
        "initial_level": 0.5,
        "target_level": 0.5,
        "demand": 0.0,
        "pump_capacity": 0.8,
        **parameters,
    }
    scenario = Scenario("pump", parameters=values, seed=11)
    environment = PumpEnvironment()
    try:
        return EvaluationRunner(values["horizon"]).run_episode(
            environment, FixedPumpController(command), scenario
        )
    finally:
        environment.close()


def test_comfortable_and_tied_initial_margin():
    m = evaluate_margin(fixed_episode(pump_capacity=0.0))
    assert m.initial_margin == pytest.approx(0.45)
    assert m.post_initial_minimum_margin == pytest.approx(0.45)
    assert m.full_episode_minimum_margin == pytest.approx(0.45)
    assert m.signed_change_from_initial == pytest.approx(0)
    assert m.critical_timestep == 0
    assert m.critical_boundary == "both"
    assert m.evidence_status == "valid"


@pytest.mark.parametrize(
    ("parameters", "command", "expected", "boundary"),
    [
        ({"demand": 0.4}, 0.0, 0.41, "lower"),
        ({"demand": 0.0}, 1.0, 0.37, "upper"),
    ],
)
def test_approach_boundary_without_violation(parameters, command, expected, boundary):
    m = evaluate_margin(fixed_episode(command=command, **parameters))
    assert m.post_initial_minimum_margin == pytest.approx(expected)
    assert m.full_episode_minimum_margin == pytest.approx(expected)
    assert m.critical_boundary == boundary
    assert m.critical_timestep == 1
    assert not m.constraint_violated


@pytest.mark.parametrize(
    ("parameters", "command", "magnitude", "boundary"),
    [
        ({"dt": 1.0, "initial_level": 0.1, "demand": 0.2}, 0.0, 0.15, "lower"),
        ({"dt": 1.0, "initial_level": 0.9, "pump_capacity": 0.3}, 1.0, 0.25, "upper"),
    ],
)
def test_physical_boundary_crossing_uses_preclip_magnitude(
    parameters, command, magnitude, boundary
):
    m = evaluate_margin(fixed_episode(command=command, **parameters))
    assert m.constraint_violated
    assert m.maximum_violation_magnitude == pytest.approx(magnitude)
    assert m.full_episode_minimum_margin == pytest.approx(-magnitude)
    assert m.critical_boundary == boundary


def test_clipped_level_cannot_hide_lower_violation_at_zero_boundary():
    result = fixed_episode(
        dt=1.0,
        initial_level=0.1,
        demand=1.0,
        pump_capacity=0.0,
        min_level=0.0,
        max_level=1.0,
    )
    assert result.trace[0]["info"]["level"] == 0.0
    assert result.trace[0]["info"]["violation_magnitude"] == pytest.approx(0.9)
    m = evaluate_margin(result)
    assert m.full_episode_minimum_margin == pytest.approx(-0.9)
    assert m.critical_boundary == "lower"


def test_clipped_level_cannot_hide_upper_violation_at_one_boundary():
    result = fixed_episode(
        command=1.0,
        dt=1.0,
        initial_level=0.9,
        demand=0.0,
        pump_capacity=2.0,
        min_level=0.0,
        max_level=1.0,
    )
    assert result.trace[0]["info"]["level"] == 1.0
    m = evaluate_margin(result)
    assert m.full_episode_minimum_margin == pytest.approx(-1.9)
    assert m.critical_boundary == "upper"


def test_initial_boundary_is_primary_minimum_even_when_post_margin_improves():
    m = evaluate_margin(fixed_episode(command=1.0, initial_level=0.05))
    assert m.initial_margin == pytest.approx(0)
    assert m.post_initial_minimum_margin > 0
    assert m.full_episode_minimum_margin == pytest.approx(0)
    assert m.signed_change_from_initial > 0
    assert m.critical_timestep == 0


@pytest.mark.parametrize(
    "corruption",
    ["empty", "missing_initial", "missing_level", "nan", "conflicting_indicator", "wrong_step"],
)
def test_missing_or_corrupt_trace_is_rejected(corruption):
    value = fixed_episode().to_dict()
    if corruption == "empty":
        value["trace"] = []
    elif corruption == "missing_initial":
        value["trace"][0]["observation"] = []
    elif corruption == "missing_level":
        del value["trace"][0]["info"]["level"]
    elif corruption == "nan":
        value["trace"][0]["info"]["level"] = float("nan")
    elif corruption == "conflicting_indicator":
        value["trace"][0]["info"]["constraint_violation"] = True
    else:
        value["trace"][0]["step"] = 4
    with pytest.raises(MarginEvidenceError):
        evaluate_margin(value)


def test_runner_step_cap_is_explicitly_incomplete():
    result = fixed_episode(horizon=20)
    value = result.to_dict()
    value["trace"] = value["trace"][:1]
    value["episode_length"] = 1
    value["truncated"] = True
    m = evaluate_margin(value)
    assert m.evidence_status == "incomplete"


def test_repeated_margin_calculation_is_identical():
    result = fixed_episode(command=1.0, initial_level=0.2)
    assert evaluate_margin(result).to_dict() == evaluate_margin(result).to_dict()


def test_genuine_phase36_development_trajectory_if_locally_available():
    experiment_id = "20261027T205018_ac561c1384"
    root = Path("artifacts/experiments")
    if not (root / experiment_id / "trajectory.jsonl").exists():
        pytest.skip("Recorded Phase 3.6 development trajectory is not installed")
    stored = ExperimentStore(root).load(experiment_id)
    m = evaluate_margin({**stored.result, "trace": stored.trajectory})
    assert m.evidence_status == "valid"
    assert m.evaluated_transitions == 200
    assert m.full_episode_minimum_margin == pytest.approx(0.25398803949356075)
