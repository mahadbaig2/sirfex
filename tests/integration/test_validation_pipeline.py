from dataclasses import replace

import pytest

from sirfex.core import Scenario
from sirfex.experiments.replay import replay_experiment
from sirfex.experiments.service import run_and_save_experiment
from sirfex.experiments.storage import ExperimentStore
from sirfex.search.persistence import SearchStore
from sirfex.training.checkpoints import CheckpointMetadata, sha256_file
from sirfex.training.config import PPOConfig
from sirfex.training.controller import PPOController
from sirfex.training.persistence import TrainingStore
from sirfex.training.ppo import train_ppo
from sirfex.training.pump_gym import PumpGymEnv
from sirfex.training.selection import discovery_report
from sirfex.training.validation import run_validation


def test_learning_curve_checkpoints_replay_and_continuation(tmp_path):
    pytest.importorskip("stable_baselines3")
    store = TrainingStore(tmp_path / "training")
    experiments = ExperimentStore(tmp_path / "experiments")
    result = run_validation(
        {"profile": "smoke", "dataset_count": 2},
        store,
        SearchStore(tmp_path / "searches"),
        experiments,
        continue_training=True,
    )
    saved = store.load(result["training_run_id"])
    assert result["status"] == "completed"
    assert not result["continuation_performed"]
    assert len(saved["learning_curve"]["points"]) == 4
    assert saved["dataset_partitions"]["leakage_check"]["passed"]
    point = saved["learning_curve"]["points"][-1]
    m = point["checkpoint"]
    metadata = CheckpointMetadata(
        **{
            **m,
            "observation_shape": tuple(m["observation_shape"]),
            "action_shape": tuple(m["action_shape"]),
        }
    )
    assert sha256_file(m["path"]) == m["sha256"]
    PPOController.load(m["path"], metadata)
    with pytest.raises(ValueError, match="schema"):
        PPOController(None, replace(metadata, observation_schema="pump-minimal:1"))
    scenario = Scenario("pump", {"horizon": 12}, environment_version="sirfex-pump:1")
    experiment = run_and_save_experiment(
        experiments, scenario, "ppo", controller_config={"checkpoint": m}, max_steps=12
    )
    assert replay_experiment(experiments, experiment.experiment_id).status == "exact_match"
    config = PPOConfig.from_dict(m["ppo_config"])
    env = PumpGymEnv([scenario], observation_schema=m["observation_schema"])
    model = train_ppo(
        env, config, seed=33, timesteps=128, checkpoint=m["path"], expected_sha256=m["sha256"]
    )
    assert model.num_timesteps == m["training_timesteps"] + 128
    with pytest.raises(ValueError, match="hash"):
        train_ppo(
            env, config, seed=33, timesteps=128, checkpoint=m["path"], expected_sha256="0" * 64
        )
    with pytest.raises(ValueError, match="configuration"):
        train_ppo(
            env,
            replace(config, gamma=0.9),
            seed=33,
            timesteps=128,
            checkpoint=m["path"],
            expected_sha256=m["sha256"],
        )


def test_study_timeout_keeps_readable_partial_artifacts(tmp_path):
    pytest.importorskip("stable_baselines3")
    store = TrainingStore(tmp_path / "training")
    result = run_validation(
        {"profile": "smoke", "wall_seconds": 0.00001},
        store,
        SearchStore(tmp_path / "search"),
        ExperimentStore(tmp_path / "exp"),
    )
    assert result["status"] == "interrupted"
    saved = store.load(result["training_run_id"], allow_partial=True)
    assert saved["manifest"]["error"]["type"] == "TimeoutError"
    assert store.list()[0]["status"] == "interrupted"


def test_discovery_selection_requires_actual_feasible_replay_evidence(tmp_path):
    store = ExperimentStore(tmp_path / "experiments")
    scenario = Scenario(
        "pump",
        {"min_level": 0.45, "initial_level": 0.5, "demand": 0.4},
        environment_version="sirfex-pump:1",
    )
    manifest = run_and_save_experiment(store, scenario, "pump-rule", max_steps=200)
    discovery = {
        "discovery_id": "test",
        "scenario": scenario.to_dict(),
        "observed_failure": True,
        "evidence_status": "observed",
        "objective": {"objective_id": "pump.constraint_violation"},
        "experiment_ids": [manifest.experiment_id],
    }

    class SearchFixture:
        def load(self, _):
            return {"manifest": {"schema_version": "1"}, "discoveries": [discovery]}

    report = discovery_report(SearchFixture(), store, ["fixture"])
    assert len(report["selected"]) == 1
    assert report["records"][0]["replay_verified"]
    discovery["experiment_ids"] = []
    assert not discovery_report(SearchFixture(), store, ["fixture"])["selected"]


def test_extended_gym_checker_and_deadline_checkpoint(tmp_path):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env

    env = PumpGymEnv(observation_schema="pump-calibrated:1")
    check_env(env)
    captured = []
    model = train_ppo(
        env, PPOConfig(), seed=3, timesteps=10000, wall_seconds=0.001, on_model=captured.append
    )
    assert captured == [model]
    assert 0 < model.num_timesteps < 10000
    model.save(str(tmp_path / "partial"))
    assert (tmp_path / "partial.zip").exists()


def test_controlled_continuation_wiring_with_explicit_gate_stub(tmp_path, monkeypatch):
    """Exercise orchestration; the stub is NOT evidence of learned competence."""
    pytest.importorskip("stable_baselines3")
    experiments = ExperimentStore(tmp_path / "experiments")
    scenario = Scenario(
        "pump",
        {"pump_capacity": 8.0, "demand": 0.3, "horizon": 200},
        seed=99,
        environment_version="sirfex-pump:1",
    )
    manifest = run_and_save_experiment(experiments, scenario, "pump-rule", max_steps=200)
    discovery = {
        "discovery_id": "test",
        "scenario": scenario.to_dict(),
        "observed_failure": True,
        "evidence_status": "observed",
        "objective": {"objective_id": "pump.constraint_violation"},
        "experiment_ids": [manifest.experiment_id],
    }

    class Searches:
        def load(self, _):
            return {"manifest": {"schema_version": "1"}, "discoveries": [discovery]}

    monkeypatch.setattr("sirfex.training.validation.competence", lambda _: {"passed": True})
    store = TrainingStore(tmp_path / "training")
    result = run_validation(
        {
            "profile": "smoke",
            "budgets": [128],
            "continuation_steps": 128,
            "schemas": ["pump-minimal:1"],
            "dataset_count": 1,
        },
        store,
        Searches(),
        experiments,
        search_ids=["fixture"],
        continue_training=True,
    )
    assert result["continuation_performed"]
    assert result["classification"] == "inconclusive"
    saved = store.load(result["training_run_id"])
    pair = saved["continuation_results"]["pairs"][0]
    assert pair["budget_verification"]["matched"]
    assert set(pair["baseline"]) == {"nominal_holdout", "known_stress", "held_out_stress"}
    assert pair["training_records"]["normal"]["additional_timesteps"] == 128
    assert pair["training_records"]["robust"]["additional_timesteps"] == 128


def test_keyboard_interrupt_preserves_active_model(tmp_path, monkeypatch):
    class Model:
        num_timesteps = 7

        def save(self, path):
            from pathlib import Path

            Path(path).write_bytes(b"partial checkpoint fixture")

    def interrupted(*args, **kwargs):
        kwargs["on_model"](Model())
        raise KeyboardInterrupt()

    monkeypatch.setattr("sirfex.training.validation.train_ppo", interrupted)
    store = TrainingStore(tmp_path / "training")
    result = run_validation(
        {"profile": "smoke", "dataset_count": 1},
        store,
        SearchStore(tmp_path / "searches"),
        ExperimentStore(tmp_path / "experiments"),
    )
    saved = store.load(result["training_run_id"], allow_partial=True)
    assert saved["manifest"]["checkpoints"]["partial"]["training_timesteps"] == 7
    assert result["status"] == "interrupted"
