"""Distinct differential findings and deterministic same-scenario sampling."""

from dataclasses import replace

import pytest

from sirfex.controllers.analytical_pump import FixedPumpController
from sirfex.core import EvaluationRunner
from sirfex.environments.pump import PumpEnvironment
from sirfex.robustness.differential import (
    classify_pair,
    rank_findings,
    sample_differentials,
)
from sirfex.robustness.margins import PumpMargin
from sirfex.search.space import SearchSpace


def margin(controller, post, *, violated=False, scenario_id="scenario_1"):
    return PumpMargin(
        "pump-signed-margin:1",
        "pump",
        "sirfex-pump:1",
        scenario_id,
        controller,
        0.3,
        post,
        min(0.3, post),
        post - 0.3,
        1 if post < 0.3 else 0,
        1,
        "lower",
        violated,
        -post if violated else 0.0,
        2,
        "valid",
        (),
    )


def test_identical_controllers_have_no_artificial_difference():
    finding = classify_pair(margin("x", 0.2), margin("y", 0.2), finite_horizon_viable=True)
    assert finding.classification == "none"
    assert finding.signed_margin_gap == 0


def test_safe_near_miss_is_class_a_not_failure():
    finding = classify_pair(margin("x", 0.1), margin("y", 0.2), finite_horizon_viable=True)
    assert finding.classification == "class_a"
    assert finding.signed_margin_gap == pytest.approx(0.1)
    assert not finding.x_violated


def test_avoidable_failure_requires_viability_and_repeated_evidence():
    x, y = margin("x", -0.01, violated=True), margin("y", 0.1)
    assert classify_pair(x, y, finite_horizon_viable=True).classification == ("class_b_unverified")
    assert (
        classify_pair(
            x, y, finite_horizon_viable=True, replay_statuses=("exact_match", "exact_match")
        ).classification
        == "class_b"
    )
    assert (
        classify_pair(
            x, y, finite_horizon_viable=False, replay_statuses=("exact_match", "exact_match")
        ).classification
        == "none"
    )
    assert classify_pair(x, y, finite_horizon_viable=None).classification == "none"


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (margin("x", -0.1, violated=True), margin("y", -0.01, violated=True)),
        (margin("x", 0.1), margin("y", -0.01, violated=True)),
    ],
)
def test_both_fail_or_reversed_failure_is_not_x_avoidable_discovery(x, y):
    assert classify_pair(x, y, finite_horizon_viable=True).classification == "none"


def test_invalid_or_unmatched_trace_is_rejected():
    with pytest.raises(ValueError):
        classify_pair(
            margin("x", 0.1), margin("y", 0.2, scenario_id="other"), finite_horizon_viable=True
        )
    with pytest.raises(ValueError):
        classify_pair(
            replace(margin("x", 0.1), evidence_status="incomplete"),
            margin("y", 0.2),
            finite_horizon_viable=True,
        )


def test_class_b_ranks_above_larger_class_a_gap_and_ties_are_deterministic():
    class_a = classify_pair(
        margin("x", 0.01, scenario_id="a"),
        margin("y", 0.4, scenario_id="a"),
        finite_horizon_viable=True,
    )
    class_b = classify_pair(
        margin("x", -0.01, violated=True, scenario_id="b"),
        margin("y", 0.05, scenario_id="b"),
        finite_horizon_viable=True,
        replay_statuses=("exact_match", "exact_match"),
    )
    assert class_b.signed_margin_gap < class_a.signed_margin_gap
    assert [r.classification for r in rank_findings([class_a, class_b])] == ["class_b", "class_a"]
    duplicate_a = replace(class_a, scenario_id="z")
    assert [r.scenario_id for r in rank_findings([duplicate_a, class_a])] == ["a", "z"]


def _space():
    return SearchSpace.from_dict(
        {
            "environment_id": "pump",
            "environment_version": "sirfex-pump:1",
            "schema_version": "1",
            "parameters": {},
            "fixed_parameters": {
                "dt": 0.1,
                "horizon": 2,
                "initial_level": 0.11,
                "target_level": 0.5,
                "demand": 0.5,
                "pump_capacity": 1.0,
                "efficiency": 1.0,
                "leak": 0.0,
                "min_level": 0.05,
                "max_level": 0.95,
            },
        }
    )


def _evaluator(controller_id, scenario):
    env = PumpEnvironment()
    try:
        result = EvaluationRunner(2).run_episode(
            env, FixedPumpController(0.0 if controller_id == "x" else 1.0), scenario
        )
        result.controller_id = controller_id
        return result
    finally:
        env.close()


def test_tiny_seeded_search_confirms_replayed_class_b_and_deduplicates():
    first = sample_differentials(
        _space(),
        "x",
        "y",
        _evaluator,
        proposals=2,
        search_seed=17,
        scenario_seed_start=370001,
    )
    second = sample_differentials(
        _space(),
        "x",
        "y",
        _evaluator,
        proposals=2,
        search_seed=17,
        scenario_seed_start=370001,
    )
    assert first == second
    assert first["coverage"] == {
        "proposed": 2,
        "evaluated": 1,
        "excluded_overlap": 1,
        "excluded_feasibility": 0,
    }
    assert len(first["findings"]["class_b"]) == 1
    assert first["findings"]["class_b"][0]["replay_statuses"] == ["exact_match", "exact_match"]


def test_exclusion_uses_physical_configuration_not_only_seed():
    space = _space()
    prior = space.scenario(space.fixed_parameters, 111)
    result = sample_differentials(
        _space(),
        "x",
        "y",
        _evaluator,
        proposals=1,
        search_seed=17,
        scenario_seed_start=222,
        excluded_scenarios=(prior,),
    )
    assert result["coverage"]["excluded_overlap"] == 1
    assert result["findings"]["class_b"] == []
