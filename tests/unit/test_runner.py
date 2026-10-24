import json

from sirfex.controllers import PumpRuleController
from sirfex.core import EvaluationRunner, Scenario
from sirfex.environments import PumpEnvironment


def test_runner_is_environment_agnostic_and_serializable() -> None:
    environment = PumpEnvironment()
    try:
        result = EvaluationRunner(max_steps=3).run_episode(
            environment, PumpRuleController(), Scenario("pump", seed=3)
        )
        json.dumps(result.to_dict())
        assert result.environment_id == "pump"
        assert result.controller_id == "pump-rule"
        assert result.episode_length == 3
        assert result.truncated is True
    finally:
        environment.close()


def test_runner_rejects_mismatched_scenario() -> None:
    environment = PumpEnvironment()
    try:
        try:
            EvaluationRunner().run_episode(environment, PumpRuleController(), Scenario("other"))
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched scenario was accepted")
    finally:
        environment.close()
