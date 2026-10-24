import numpy as np
import pytest
from gymnasium import spaces

from sirfex.core import EvaluationError, EvaluationRunner, Scenario, managed_environment
from sirfex.core.failures import FailureCategory


class FakeEnvironment:
    id = "fake"
    version = "fake:1"

    def __init__(self, *, reset_error=None, step_error=None, invalid_observation=False):
        self.observation_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)
        self.reset_error = reset_error
        self.step_error = step_error
        self.invalid_observation = invalid_observation
        self.closed = False

    def reset(self, scenario):
        del scenario
        if self.reset_error:
            raise self.reset_error
        observation = np.array([2.0 if self.invalid_observation else 0.0], dtype=np.float32)
        return observation, {}

    def step(self, action):
        del action
        if self.step_error:
            raise self.step_error
        return np.array([0.0], dtype=np.float32), 1.0, True, False, {}

    def evaluate_failure(self, total_reward, terminated, info):
        del total_reward, terminated, info
        return {"outcome": "nominal"}

    def metadata(self):
        return {}

    def close(self):
        self.closed = True


class FakeController:
    id = "fake-controller"

    def __init__(self, *, reset_error=None, action=0):
        self.reset_error = reset_error
        self.action = action

    def reset(self):
        if self.reset_error:
            raise self.reset_error

    def act(self, observation, deterministic=True):
        del observation, deterministic
        return self.action

    def metadata(self):
        return {}


@pytest.mark.parametrize(
    ("environment", "controller", "category"),
    [
        (
            FakeEnvironment(reset_error=RuntimeError("reset")),
            FakeController(),
            FailureCategory.ENVIRONMENT_EXCEPTION,
        ),
        (
            FakeEnvironment(),
            FakeController(reset_error=RuntimeError("controller")),
            FailureCategory.CONTROLLER_EXCEPTION,
        ),
        (
            FakeEnvironment(invalid_observation=True),
            FakeController(),
            FailureCategory.INVALID_OBSERVATION,
        ),
        (FakeEnvironment(), FakeController(action=1), FailureCategory.INVALID_ACTION),
        (
            FakeEnvironment(step_error=RuntimeError("step")),
            FakeController(),
            FailureCategory.ENVIRONMENT_EXCEPTION,
        ),
    ],
)
def test_runner_reports_lifecycle_errors(environment, controller, category) -> None:
    with pytest.raises(EvaluationError) as error:
        EvaluationRunner(max_steps=2).run_episode(environment, controller, Scenario("fake"))
    assert error.value.record.category == category


def test_managed_environment_closes_owned_environment() -> None:
    environment = FakeEnvironment()
    with managed_environment(lambda: environment) as owned:
        assert owned is environment
    assert environment.closed is True


def test_runner_reuses_environment_for_multiple_episodes() -> None:
    environment = FakeEnvironment()
    runner = EvaluationRunner(max_steps=2)
    first = runner.run_episode(environment, FakeController(), Scenario("fake", seed=1))
    second = runner.run_episode(environment, FakeController(), Scenario("fake", seed=2))
    assert first.episode_length == second.episode_length == 1
    assert environment.closed is False
