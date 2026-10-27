"""Actual curriculum transitions, completed episodes and partial exposure."""

import numpy as np
import pytest

from sirfex.core import Scenario
from sirfex.training.exposure import exposure_adequacy
from sirfex.training.protocol import scenario_record
from sirfex.training.pump_gym import PumpGymEnv


def _scenario(seed, *, horizon=3, initial=0.5, demand=0.5):
    return Scenario(
        "pump", {"horizon": horizon, "initial_level": initial, "demand": demand}, seed=seed
    )


def _run_episode(env, action=0.5):
    while True:
        _, _, terminated, truncated, _ = env.step(np.array([action]))
        if terminated or truncated:
            return


def test_nominal_only_counts_physical_interactions_and_completed_episode():
    env = PumpGymEnv([_scenario(1)], seed=7)
    env.reset()
    _run_episode(env)
    exposure = env.exposure_summary(scheduled_interactions=3, model_interactions=3)
    assert exposure["reconciliation_valid"]
    assert exposure["nominal_episode_starts"] == 1
    assert exposure["nominal_completed_episodes"] == 1
    assert exposure["nominal_interactions"] == 3
    assert exposure["stress_interactions"] == 0
    assert exposure["stress_interaction_fraction"] == 0
    assert exposure["partial_nominal_interactions"] == 0
    env.close()


def test_stress_only_records_partial_and_completed_exposure_by_discovery():
    stress = _scenario(2, horizon=3)
    sid = scenario_record(stress)["scenario_id"]
    env = PumpGymEnv(
        [_scenario(1)],
        stress_scenarios=[stress],
        stress_ratio=1,
        discovery_ids={sid: "discovery-two"},
        seed=11,
    )
    env.reset(seed=11)
    env.step(np.array([0.5]))
    partial = env.exposure_summary(scheduled_interactions=3, model_interactions=1)
    assert partial["reconciliation_valid"]
    assert partial["stress_interactions"] == 1
    assert partial["partial_stress_interactions"] == 1
    assert partial["stress_completed_episodes"] == 0
    assert partial["per_discovery_interactions"] == {"discovery-two": 1}
    _run_episode(env)
    done = env.exposure_summary(scheduled_interactions=3, model_interactions=3)
    assert done["reconciliation_valid"]
    assert done["stress_completed_episodes"] == 1
    assert done["completed_stress_episode_lengths"] == [3]
    assert done["mean_completed_stress_episode_length"] == 3
    assert done["distinct_discoveries"] == 1
    assert done["explicit_reset_seeds"] == [11]
    env.close()


def test_early_constraint_termination_is_a_completed_short_stress_episode():
    stress = _scenario(2, horizon=10, initial=0.06, demand=1.0)
    env = PumpGymEnv([_scenario(1)], stress_scenarios=[stress], stress_ratio=1, seed=4)
    env.reset()
    _, _, terminated, truncated, _ = env.step(np.array([0.0]))
    assert terminated and not truncated
    exposure = env.exposure_summary(scheduled_interactions=1, model_interactions=1)
    assert exposure["stress_completed_episodes"] == 1
    assert exposure["completed_stress_episode_lengths"] == [1]
    assert exposure["partial_stress_interactions"] == 0
    assert exposure["reconciliation_valid"]
    env.close()


def test_mixed_sampling_is_deterministic_and_reconciles_partial_last_episode():
    normal, stress = _scenario(1), _scenario(2, horizon=2)
    outputs = []
    for _ in range(2):
        env = PumpGymEnv([normal], stress_scenarios=[stress], stress_ratio=0.5, seed=19)
        for _episode in range(8):
            env.reset()
            _run_episode(env)
        env.reset()
        env.step(np.array([0.5]))
        exposure = env.exposure_summary(
            model_interactions=env._exposure.snapshot()["actual_environment_interactions"]
        )
        assert exposure["reconciliation_valid"]
        assert exposure["episode_starts"] == 9
        assert (
            exposure["partial_nominal_interactions"] + exposure["partial_stress_interactions"] == 1
        )
        assert (
            exposure["stress_interactions"] + exposure["nominal_interactions"]
            == exposure["actual_environment_interactions"]
        )
        outputs.append(exposure)
        env.close()
    assert outputs[0] == outputs[1]


def test_accounting_flags_unassigned_or_inconsistent_model_steps():
    env = PumpGymEnv([_scenario(1)], seed=2)
    env.reset()
    env.step(np.array([0.5]))
    assert not env.exposure_summary(model_interactions=2)["reconciliation_valid"]
    env.reset(options={"scenario": _scenario(9, demand=0.6)})
    env.step(np.array([0.5]))
    bad = env.exposure_summary(model_interactions=2)
    assert bad["invalid_assignments"] == 1
    assert bad["invalid_resets"] == 1
    assert not bad["reconciliation_valid"]
    env.close()


@pytest.mark.parametrize("target", [0.05, 0.10])
def test_interaction_target_uses_episode_boundaries_and_counts_actual_steps(target):
    normal, stress = _scenario(1, horizon=20), _scenario(2, horizon=20, demand=0.6)
    env = PumpGymEnv([normal], stress_scenarios=[stress], seed=9, stress_target_fraction=target)
    for _ in range(100):
        env.reset()
        _run_episode(env)
    result = env.exposure_summary(model_interactions=2000, scheduled_interactions=2000)
    assert result["reconciliation_valid"]
    assert result["sampling_mode"] == "interaction-debt:1"
    assert result["configured_stress_probability"] is None
    assert abs(result["stress_interaction_fraction"] - target) <= 0.01
    assert result["stress_completed_episodes"] >= 5
    env.close()


def test_exposure_gate_reports_adequate_insufficient_and_invalid():
    env = PumpGymEnv(
        [_scenario(1)], stress_scenarios=[_scenario(2, demand=0.6)], stress_ratio=1, seed=1
    )
    for _ in range(10):
        env.reset()
        _run_episode(env)
    summary = env.exposure_summary(model_interactions=30)
    limits = {"stress_completed_episodes": 10, "stress_interactions": 30, "distinct_discoveries": 1}
    assert exposure_adequacy(summary, 1, thresholds=limits)["status"] == "adequate_exposure"
    assert exposure_adequacy(summary, 3)["status"] == "insufficient_exposure"
    assert exposure_adequacy(summary, 1, trial_complete=False)["status"] == "invalid_exposure"
    assert exposure_adequacy(None, 1)["status"] == "invalid_exposure"
    env.close()


def test_scheduled_stress_only_changes_on_episode_boundaries():
    normal, stress = _scenario(1, horizon=10), _scenario(2, horizon=10, demand=0.6)
    env = PumpGymEnv(
        [normal],
        stress_scenarios=[stress],
        seed=7,
        stress_schedule=[
            {"until_step": 100, "stress_fraction": 0.0},
            {"until_step": 200, "stress_fraction": 0.1},
        ],
    )
    for _ in range(20):
        env.reset()
        _run_episode(env)
    result = env.exposure_summary(scheduled_interactions=200, model_interactions=200)
    assert result["reconciliation_valid"]
    assert result["sampling_mode"] == "interaction-schedule:1"
    assert 0 < result["stress_interactions"] <= 20
    assert result["stress_completed_episodes"] >= 1
    env.close()


def test_weighted_pool_requires_valid_exact_weights():
    normal = _scenario(1)
    first, second = _scenario(2, demand=0.6), _scenario(3, demand=0.7)
    ids = [scenario_record(s)["scenario_id"] for s in (first, second)]
    with pytest.raises(ValueError):
        PumpGymEnv(
            [normal], stress_scenarios=[first, second], stress_ratio=1, stress_weights={ids[0]: 1}
        )
    env = PumpGymEnv(
        [normal],
        stress_scenarios=[first, second],
        stress_ratio=1,
        stress_weights={ids[0]: 9, ids[1]: 1},
        seed=11,
    )
    for _ in range(100):
        env.reset()
        _run_episode(env)
    counts = env.exposure_summary()["per_scenario_episode_starts"]
    assert counts[ids[0]] > counts[ids[1]]
    env.close()
