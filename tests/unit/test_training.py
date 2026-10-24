from pathlib import Path

import numpy as np
import pytest

from sirfex.core.interfaces import Scenario
from sirfex.environments.pump import PumpEnvironment
from sirfex.training.checkpoints import sha256_file, verify_checkpoint
from sirfex.training.config import PPOConfig, training_profile
from sirfex.training.curricula import ScenarioCurriculum
from sirfex.training.pump_gym import PumpGymEnv
from sirfex.training.regression import classify_regression


def test_pump_training_adapter_matches_evaluation_transition() -> None:
    scenario = Scenario("pump", {"demand": 0.6, "efficiency": 0.8, "horizon": 10}, seed=4)
    evaluation = PumpEnvironment()
    training = PumpGymEnv([scenario], seed=4)
    try:
        evaluation.reset(scenario)
        training.reset(options={"scenario": scenario})
        action = np.array([0.7], dtype=np.float32)
        expected = evaluation.step(action)
        actual = training.step(action)
        np.testing.assert_allclose(actual[0], expected[0])
        assert actual[2:4] == expected[2:4]
        assert actual[4]["level"] == expected[4]["level"]
        assert actual[4]["base_reward"] == expected[1]
    finally:
        evaluation.close()
        training.close()


def test_training_adapter_spaces_and_action_validation() -> None:
    environment = PumpGymEnv([Scenario("pump")])
    try:
        observation, _ = environment.reset(options={"scenario": Scenario("pump")})
        assert environment.observation_space.contains(observation)
        with pytest.raises(ValueError):
            environment.step(np.array([2.0], dtype=np.float32))
    finally:
        environment.close()


def test_profiles_are_cpu_validated() -> None:
    assert training_profile("smoke").ppo.device == "cpu"
    with pytest.raises(ValueError):
        PPOConfig(n_steps=3, batch_size=2).validate()


def test_curriculum_sampling_is_reproducible() -> None:
    normal = [Scenario("pump", {"demand": 0.4}, seed=1)]
    stress = [Scenario("pump", {"demand": 0.9}, seed=2)]
    first = ScenarioCurriculum(normal, stress, 0.5, 8).scenarios(20)
    second = ScenarioCurriculum(normal, stress, 0.5, 8).scenarios(20)
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]


def test_checkpoint_hash_verification(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.zip"
    path.write_bytes(b"checkpoint")
    digest = sha256_file(path)
    verify_checkpoint(path, digest)
    with pytest.raises(ValueError):
        verify_checkpoint(path, "0" * 64)


def test_regression_classification_is_explicit() -> None:
    baseline = {
        "mean_tracking_error": 1.0,
        "constraint_violation_frequency": 0.5,
        "maximum_violation_magnitude": 0.5,
    }
    improved = {
        "mean_tracking_error": 0.9,
        "constraint_violation_frequency": 0.1,
        "maximum_violation_magnitude": 0.1,
    }
    result = classify_regression(
        baseline,
        improved,
        baseline,
        improved,
        baseline,
        improved,
        {"nominal_tracking_error_degradation": 0.1},
    )
    assert result["classification"] == "improved"
