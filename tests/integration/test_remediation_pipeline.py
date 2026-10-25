from dataclasses import replace

import pytest

from sirfex.core import Scenario
from sirfex.experiments.replay import replay_experiment
from sirfex.experiments.service import run_and_save_experiment
from sirfex.experiments.storage import ExperimentStore
from sirfex.training.competence import run_competence
from sirfex.training.persistence import TrainingStore
from sirfex.training.remediation_diagnostics import metadata_from_dict


def test_analytical_reference_uses_same_physical_scenario(tmp_path):
    store = ExperimentStore(tmp_path)
    scenario = Scenario("pump", {"dt": 0.2, "horizon": 40, "demand": 0.6, "leak": 0.01})
    manifest = run_and_save_experiment(store, scenario, "pump-analytical", max_steps=40)
    result = store.load(manifest.experiment_id).result
    assert result["episode_length"] == 40
    assert result["failures"] == []
    assert replay_experiment(store, manifest.experiment_id).status == "exact_match"


def test_normalized_ppo_save_load_stochastic_replay_and_contract(tmp_path):
    pytest.importorskip("stable_baselines3")
    from sirfex.training.config import PPOConfig
    from sirfex.training.controller import PPOController
    from sirfex.training.ppo import train_ppo
    from sirfex.training.pump_gym import PumpGymEnv

    store = TrainingStore(tmp_path / "training")
    result = run_competence(
        {
            "profile": "smoke",
            "dataset_count": 1,
            "budgets": [128],
            "policy_action_schema": "pump-normalized:1",
            "reward_schema": "pump-survival-tracking:2",
        },
        store,
    )
    assert result["status"] == "completed"
    saved = store.load(result["training_run_id"])
    m = next(iter(saved["manifest"]["checkpoints"].values()))
    metadata = metadata_from_dict(m)
    PPOController.load(metadata.path, metadata)
    with pytest.raises(ValueError, match="schema"):
        PPOController.load(metadata.path, replace(metadata, policy_action_schema="pump-direct:1"))
    experiments = ExperimentStore(tmp_path / "experiments")
    manifest = run_and_save_experiment(
        experiments,
        Scenario("pump", {"horizon": 12}),
        "ppo",
        controller_config={"checkpoint": m, "inference_seed": 9},
        deterministic=False,
        max_steps=12,
    )
    assert replay_experiment(experiments, manifest.experiment_id).status == "exact_match"
    with pytest.raises(ValueError, match="contract"):
        train_ppo(
            PumpGymEnv(
                observation_schema="pump-calibrated:1", policy_action_schema="pump-normalized:1"
            ),
            PPOConfig.from_dict(m["ppo_config"]),
            seed=3,
            timesteps=128,
            checkpoint=m["path"],
            expected_sha256=m["sha256"],
        )
