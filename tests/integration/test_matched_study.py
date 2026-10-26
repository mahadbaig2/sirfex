from pathlib import Path

import pytest

from sirfex.core import Scenario
from sirfex.training.regression import classify_matched_robustness


def _result(frequency=0.5, magnitude=0.2, tracking=0.1):
    return {
        "episodes": 4,
        "execution_failures": 0,
        "constraint_violation_frequency": frequency,
        "maximum_violation_magnitude": magnitude,
        "mean_tracking_error": tracking,
    }


def test_matched_regression_needs_held_out_signal_and_respects_guardrails():
    from sirfex.training.matched_study import GUARDRAILS

    normal = {
        "nominal_holdout": _result(0, 0, 0.1),
        "known_stress": _result(),
        "held_out_stress": _result(),
    }
    robust = {name: dict(value) for name, value in normal.items()}
    robust["known_stress"]["constraint_violation_frequency"] = 0
    assert classify_matched_robustness(normal, robust, GUARDRAILS)["classification"] == (
        "inconclusive"
    )
    robust["held_out_stress"]["constraint_violation_frequency"] = 0.25
    assert classify_matched_robustness(normal, robust, GUARDRAILS)["classification"] == ("improved")
    robust["nominal_holdout"]["mean_tracking_error"] = 0.12
    assert classify_matched_robustness(normal, robust, GUARDRAILS)["classification"] == (
        "regressed"
    )
    robust["nominal_holdout"]["mean_tracking_error"] = 0.1
    robust["held_out_stress"]["execution_failures"] = 1
    assert classify_matched_robustness(normal, robust, GUARDRAILS)["classification"] == (
        "inconclusive"
    )


def test_frozen_study_settings_reject_posthoc_changes():
    from sirfex.training.matched_study import study_config

    assert study_config({})["continuation_steps"] == 25600
    with pytest.raises(ValueError, match="25,600"):
        study_config({"continuation_steps": 51200})
    with pytest.raises(ValueError, match="30%"):
        study_config({"stress_ratio": 0.5})
    with pytest.raises(ValueError, match="generator"):
        study_config({"held_out_seed": 1})


def test_short_paired_continuation_restores_optimizer_and_persists(tmp_path):
    pytest.importorskip("stable_baselines3")
    from scripts.smoke_matched_continuation import smoke
    from sirfex.training.config import PPOConfig
    from sirfex.training.matched_study import model_fingerprint
    from sirfex.training.persistence import TrainingStore
    from sirfex.training.ppo import train_ppo
    from sirfex.training.pump_gym import PumpGymEnv
    from sirfex.training.validation import _checkpoint

    store = TrainingStore(tmp_path / "training")
    scenario = Scenario("pump", {"horizon": 20, "demand": 0.6})
    stress = Scenario("pump", {"horizon": 20, "initial_level": 0.1, "demand": 0.8})
    config = PPOConfig(n_steps=128, batch_size=32, torch_threads=2)
    env = PumpGymEnv(
        [scenario],
        observation_schema="pump-calibrated:1",
        policy_action_schema="pump-normalized:1",
        reward_schema="pump-survival-tracking:2",
    )
    model = train_ppo(env, config, seed=3, timesteps=128)
    base_id = "smokebase001"
    base_dir = store.directory(base_id)
    base_dir.mkdir()
    meta = _checkpoint(
        model,
        base_dir,
        base_id,
        "baseline",
        {
            "ppo": config.to_dict(),
            "policy_action_schema": "pump-normalized:1",
            "reward_schema": "pump-survival-tracking:2",
        },
        3,
        "pump-calibrated:1",
    )
    env.close()
    prepared_id = "smokeprep001"
    store.save(
        prepared_id,
        {
            "manifest.json": {
                "schema_version": "2",
                "training_run_id": prepared_id,
                "status": "prepared",
            },
            "baseline_verification.json": {
                "3": {"checkpoint": meta.to_dict(), "fingerprint": model_fingerprint(model)}
            },
            "partitions.json": {
                "groups": {
                    "training": [{"scenario": scenario.to_dict()}],
                    "known_stress": [{"scenario": stress.to_dict()}],
                }
            },
        },
    )
    result = smoke(prepared_id, store)
    assert result["status"] == "completed"
    assert result["starting_state_equal"]
    assert len(result["checkpoint_hashes"]) == 2
    saved = store.load(result["training_run_id"])
    assert saved["manifest"]["status"] == "completed"
    assert all(Path(meta["path"]).is_file() for meta in saved["manifest"]["checkpoints"].values())
    assert all(row["additional_timesteps"] == 128 for row in saved["evaluation"]["records"])
