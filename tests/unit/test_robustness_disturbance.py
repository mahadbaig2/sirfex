"""Disturbance mathematics and grid outcome semantics on tiny synthetic cases."""

from dataclasses import replace

import pytest

from sirfex.controllers.analytical_pump import FixedPumpController
from sirfex.core import EvaluationRunner, Scenario
from sirfex.environments.pump import PumpEnvironment
from sirfex.robustness.disturbance import (
    DIRECTIONS,
    FeasibilityDiscrepancy,
    UndefinedAxis,
    _refine_ceiling,
    evaluate_direction,
    prepare_grid,
    transform,
)
from sirfex.robustness.identity import semantic_hash


def anchor(**overrides):
    return Scenario(
        "pump",
        parameters={
            "dt": 0.1,
            "horizon": 3,
            "initial_level": 0.21,
            "target_level": 0.5,
            "demand": 0.5,
            "pump_capacity": 1.0,
            "efficiency": 1.0,
            "leak": 0.02,
            "min_level": 0.05,
            "max_level": 0.95,
            **overrides,
        },
        seed=91,
    )


def evaluate(scenario, command=1.0):
    env = PumpEnvironment()
    try:
        return EvaluationRunner(scenario.parameters["horizon"]).run_episode(
            env, FixedPumpController(command), scenario
        )
    finally:
        env.close()


def test_all_sixteen_directions_are_declared():
    assert len(DIRECTIONS) == 16
    assert ("demand", "capacity", "efficiency", "leak") in DIRECTIONS
    assert ("surplus_demand",) in DIRECTIONS


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        (("demand",), {"demand": 0.6}),
        (("capacity",), {"pump_capacity": 0.8}),
        (("efficiency",), {"efficiency": 0.8}),
        (("leak",), {"leak": 0.22}),
        (("surplus_demand",), {"demand": 0.4}),
        (
            ("demand", "capacity", "efficiency", "leak"),
            {"demand": 0.6, "pump_capacity": 0.8, "efficiency": 0.8, "leak": 0.22},
        ),
    ],
)
def test_prescribed_individual_combined_and_surplus_transforms(direction, expected):
    result = transform(anchor(), direction, 0.2)
    for name, value in expected.items():
        assert result.parameters[name] == pytest.approx(value)
    assert result.seed == 91


def test_zero_target_and_zero_demand_are_valid_without_arbitrary_scale():
    a = anchor(min_level=0.0, target_level=0.0, demand=0.0, leak=0.1)
    assert transform(a, ("demand",), 0.1).parameters["demand"] == 0
    assert transform(a, ("leak",), 0.1).parameters["leak"] == pytest.approx(0.1)


def test_zero_demand_and_zero_leak_make_leak_axis_undefined():
    a = anchor(demand=0.0, leak=0.0)
    with pytest.raises(UndefinedAxis):
        transform(a, ("leak",), 0.1)
    cell = evaluate_direction(a, ("leak",), [0, 0.1], "pump-fixed", lambda s: evaluate(s, 1.0))
    assert cell.status == "undefined_axis"
    assert cell.delta_first is None
    assert cell.evaluated_grid_points == 0


def test_parameter_bound_censors_shared_domain_without_clamping():
    a = anchor(demand=0.95, pump_capacity=2.0, horizon=2)
    points = prepare_grid(a, ("demand",), [0, 0.05, 0.1])
    assert [p.domain_status for p in points] == ["valid", "valid", "censored_bound"]
    cell = evaluate_direction(
        a, ("demand",), [0, 0.05, 0.1], "pump-fixed", lambda s: evaluate(s, 1.0), prepared=points
    )
    assert cell.status == "censored_bound"
    assert cell.delta_first is None
    assert cell.evaluated_grid_points == 2


def test_no_failure_through_valid_grid_is_censored_max():
    a = anchor(horizon=2, initial_level=0.5)
    cell = evaluate_direction(
        a, ("demand",), [0, 0.05, 0.1], "pump-fixed", lambda s: evaluate(s, 1.0)
    )
    assert cell.status == "censored_max"
    assert [row["failure"] for row in cell.grid_outcomes] == [False, False, False]


def test_nominal_failure_is_not_reported_as_positive_threshold():
    a = anchor(horizon=3, initial_level=0.11)
    cell = evaluate_direction(
        a, ("demand",), [0, 0.05, 0.1], "pump-fixed", lambda s: evaluate(s, 0.0)
    )
    assert cell.status == "fails_at_nominal"
    assert cell.delta_first == 0.0


def test_first_observed_failure_and_descriptive_relative_threshold(monkeypatch):
    a = anchor(leak=0.0)

    def threshold(s):
        viable = s.parameters["demand"] < 0.6
        return {
            "classification": "viable" if viable else "infeasible",
            "finite_horizon_viable": viable,
        }

    monkeypatch.setattr("sirfex.robustness.disturbance.analyze_feasibility", threshold)
    points = prepare_grid(a, ("demand",), [0, 0.1, 0.2])
    cell = evaluate_direction(
        a, ("demand",), [0, 0.1, 0.2], "pump-fixed", lambda s: evaluate(s, 0.0), prepared=points
    )
    assert cell.status == "failed"
    assert cell.delta_first == pytest.approx(0.1)
    assert cell.delta_infeasible_grid == pytest.approx(0.2)
    assert cell.relative_failure_threshold == pytest.approx(0.5)
    assert len(cell.grid_outcomes) == 3


def test_nonmonotone_controller_and_physics_vectors_are_flagged(monkeypatch):
    a = anchor(horizon=2, initial_level=0.11, leak=0.0)

    def nonmonotone(s):
        viable = s.parameters["demand"] != pytest.approx(0.55)
        return {
            "classification": "viable" if viable else "infeasible",
            "finite_horizon_viable": viable,
        }

    monkeypatch.setattr("sirfex.robustness.disturbance.analyze_feasibility", nonmonotone)
    points = prepare_grid(a, ("demand",), [0, 0.1, 0.2])

    def policy(s):
        command = 0.0 if s.parameters["demand"] < 0.6 else 1.0
        return evaluate(s, command)

    cell = evaluate_direction(a, ("demand",), [0, 0.1, 0.2], "pump-fixed", policy, prepared=points)
    assert cell.nonmonotone
    assert cell.physics_nonmonotone
    assert [r["failure"] for r in cell.grid_outcomes] == [True, True, False]


def test_proven_infeasibility_and_safe_controller_halts_with_evidence(monkeypatch):
    a = anchor(horizon=2, initial_level=0.5)

    def impossible(s):
        viable = s.parameters["demand"] > 0.5
        return {
            "classification": "viable" if viable else "infeasible",
            "finite_horizon_viable": viable,
        }

    monkeypatch.setattr("sirfex.robustness.disturbance.analyze_feasibility", impossible)
    points = prepare_grid(a, ("demand",), [0, 0.1])
    with pytest.raises(FeasibilityDiscrepancy) as captured:
        evaluate_direction(
            a, ("demand",), [0, 0.1], "pump-fixed", lambda s: evaluate(s, 1.0), prepared=points
        )
    assert captured.value.evidence["delta"] == 0


def test_real_restricted_refinement_and_out_of_domain_refusal():
    a = anchor(horizon=10, initial_level=0.2, demand=0.9, pump_capacity=0.8, leak=0.0)
    points = prepare_grid(a, ("demand",), [0, 0.1])
    assert [p.viable for p in points] == [True, False]
    refined = _refine_ceiling(a, ("demand",), points, physics_nonmonotone=False)
    assert refined["status"] == "restricted_monotone_bracket"
    assert 0 <= refined["viable_lower"] < refined["infeasible_upper"] <= 0.1

    b = anchor(dt=1.0, leak=1.2)
    fake = prepare_grid(b, ("demand",), [0, 0.1])
    fake = (replace(fake[0], viable=True), replace(fake[1], viable=False))
    refusal = _refine_ceiling(b, ("demand",), fake, physics_nonmonotone=False)
    assert refusal["status"] == "ceiling_unresolved_domain"


def test_repeat_is_deterministic():
    a = anchor()
    first = evaluate_direction(
        a, ("capacity",), [0, 0.1], "pump-fixed", lambda s: evaluate(s, 1.0)
    ).to_dict()
    second = evaluate_direction(
        a, ("capacity",), [0, 0.1], "pump-fixed", lambda s: evaluate(s, 1.0)
    ).to_dict()
    assert first == second


@pytest.mark.parametrize(
    "tamper",
    [
        "anchor",
        "direction",
        "delta",
        "order",
        "parameter",
        "scenario_id",
        "feasibility",
        "viable",
        "truncate",
        "environment_version",
        "hash",
        "protocol",
    ],
)
def test_prepared_grid_rejects_tampering_before_evaluation(tamper):
    a = anchor()
    grid = [0, 0.1]
    prepared = prepare_grid(a, ("demand",), grid, protocol_sha256="protocol-a")
    called = False
    direction = ("demand",)
    if tamper == "anchor":
        a = anchor(demand=0.4)
    elif tamper == "direction":
        direction = ("capacity",)
    if tamper == "delta":
        prepared = replace(prepared, points=(replace(prepared[0], delta=0.03), prepared[1]))
    elif tamper == "order":
        prepared = replace(prepared, points=tuple(reversed(prepared.points)))
    elif tamper == "parameter":
        wrong = replace(
            prepared[1].scenario, parameters={**prepared[1].scenario.parameters, "demand": 0.7}
        )
        prepared = replace(prepared, points=(prepared[0], replace(prepared[1], scenario=wrong)))
    elif tamper == "scenario_id":
        prepared = replace(prepared, points=(prepared[0], replace(prepared[1], scenario_id="fake")))
    elif tamper == "feasibility":
        prepared = replace(
            prepared, points=(prepared[0], replace(prepared[1], feasibility="infeasible"))
        )
    elif tamper == "viable":
        prepared = replace(prepared, points=(prepared[0], replace(prepared[1], viable=False)))
    elif tamper == "truncate":
        prepared = replace(prepared, points=prepared.points[:1])
    elif tamper == "environment_version":
        prepared = replace(prepared, environment_version="other")
    elif tamper == "hash":
        prepared = replace(prepared, semantic_sha256="0" * 64)
    elif tamper == "protocol":
        prepared = replace(prepared, protocol_sha256="other")
    if tamper in {"parameter", "feasibility"}:
        # Rehashing user-supplied metadata is not a source of authority.
        from dataclasses import asdict

        value = asdict(prepared)
        value.pop("semantic_sha256")
        prepared = replace(prepared, semantic_sha256=semantic_hash(value))

    def evaluator(s):
        nonlocal called
        called = True
        return evaluate(s, 1.0)

    with pytest.raises(ValueError, match="Prepared grid"):
        evaluate_direction(
            a,
            direction,
            grid,
            "pump-fixed",
            evaluator,
            prepared=prepared,
            protocol_sha256="protocol-a",
        )
    assert not called


def test_prepared_grid_reuses_cached_analyzer_across_controllers(monkeypatch):
    calls = 0

    def counted(s):
        nonlocal calls
        calls += 1
        return {"classification": "viable", "finite_horizon_viable": True}

    monkeypatch.setattr("sirfex.robustness.disturbance.analyze_feasibility", counted)
    a = anchor()
    prepared = prepare_grid(a, ("demand",), [0, 0.1], protocol_sha256="p")
    for controller in ("pump-fixed", "pump-fixed"):
        evaluate_direction(
            a,
            ("demand",),
            [0, 0.1],
            controller,
            lambda s: evaluate(s, 1.0),
            prepared=prepared,
            protocol_sha256="p",
        )
    assert calls == 2
