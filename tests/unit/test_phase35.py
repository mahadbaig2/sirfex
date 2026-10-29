"""Phase 3.5 study gates and real PPO transition reconciliation."""

from copy import deepcopy

import pytest

from sirfex.training.exposure import exposure_adequacy
from sirfex.training.phase35 import CANDIDATES, SCREEN_SEEDS, select


def _evaluation(mae, violations=0):
    return {
        "mean_tracking_error": mae,
        "constraint_violation_frequency": violations,
        "execution_failures": 0,
    }


def _rows():
    records = []
    for seed in SCREEN_SEEDS:
        for candidate in CANDIDATES:
            normal = candidate["id"] == "normal"
            records.append(
                {
                    "seed": int(seed),
                    "candidate_id": candidate["id"],
                    "requested_additional_steps": 51200,
                    "actual_additional_steps": 51200,
                    "adequacy": None if normal else {"status": "adequate_exposure"},
                    "development_nominal": _evaluation(0.10),
                    "development_stress": _evaluation(0.10 if normal else 0.09),
                }
            )
    return records


def test_selection_needs_both_seeds_and_nominal_guard():
    rows = _rows()
    assert (
        select(rows)["selected_candidate_id"] == "expanded_10"
        or select(rows)["selected_candidate_id"] == "expanded_5"
    )
    changed = deepcopy(rows)
    for row in changed:
        if row["seed"] == 17 and row["candidate_id"] == "expanded_5":
            row["development_nominal"]["mean_tracking_error"] = 0.12
        if row["seed"] == 17 and row["candidate_id"] == "expanded_10":
            row["development_nominal"]["mean_tracking_error"] = 0.12
    assert select(changed)["status"] == "no_eligible_curriculum"
    changed[2]["adequacy"] = {"status": "insufficient_exposure"}
    assert select(changed)["status"] == "insufficient_exposure"
    assert select(changed[:-1])["status"] == "invalid_study"


def test_underexposed_trial_cannot_support_effect_interpretation():
    summary = {
        "actual_environment_interactions": 5120,
        "model_interactions": 5120,
        "nominal_interactions": 4950,
        "stress_interactions": 170,
        "stress_completed_episodes": 1,
        "distinct_discoveries": 1,
        "reconciliation_valid": True,
        "invalid_assignments": 0,
        "invalid_resets": 0,
        "unassigned_interactions": 0,
    }
    assert exposure_adequacy(summary, 7)["status"] == "insufficient_exposure"


def test_real_ppo_rollout_exposure_reconciles(tmp_path):
    pytest.importorskip("stable_baselines3")
    from sirfex.core import Scenario
    from sirfex.training.config import PPOConfig
    from sirfex.training.ppo import train_ppo
    from sirfex.training.pump_gym import PumpGymEnv

    normal = Scenario("pump", {"horizon": 8, "demand": 0.3}, seed=1)
    stress = Scenario("pump", {"horizon": 8, "demand": 0.6}, seed=2)
    env = PumpGymEnv([normal], stress_scenarios=[stress], stress_target_fraction=0.1, seed=3)
    config = PPOConfig(n_steps=128, batch_size=32, n_epochs=1, torch_threads=2)
    model = train_ppo(env, config, seed=3, timesteps=256, wall_seconds=60)
    exposure = env.exposure_summary(
        scheduled_interactions=256, model_interactions=model.num_timesteps
    )
    assert model.num_timesteps == 256
    assert exposure["reconciliation_valid"]
    assert exposure["nominal_interactions"] + exposure["stress_interactions"] == 256
    assert exposure["episode_starts"] >= exposure["nominal_completed_episodes"]
    assert exposure["stress_episode_starts"] >= 1
    model.save(str(tmp_path / "smoke.zip"))
    assert (tmp_path / "smoke.zip").is_file()
    env.close()
