from sirfex.controllers import RandomController
from sirfex.core import EvaluationRunner, Scenario
from sirfex.environments import LunarLanderEnvironment


def test_lunarlander_headless_episode() -> None:
    environment = LunarLanderEnvironment()
    try:
        result = EvaluationRunner(max_steps=20).run_episode(
            environment,
            RandomController(environment.action_space),
            Scenario("lunarlander", seed=0),
            record_trace=False,
        )
        assert result.environment_id == "lunarlander"
        assert result.episode_length == 20
        assert result.truncated is True
    finally:
        environment.close()
