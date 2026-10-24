import numpy as np
import pytest

from sirfex.core import ScenarioValidationError
from sirfex.search.objectives import PumpTrackingObjective
from sirfex.search.space import ParameterDefinition, SearchSpace
from sirfex.search.strategies import RandomSearchStrategy, TPESearchStrategy


def test_search_space_sampling_is_deterministic() -> None:
    space = SearchSpace(
        "pump",
        "sirfex-pump:1",
        parameters=(ParameterDefinition("demand", "float", 0.1, 0.9),),
    )
    first = RandomSearchStrategy(4)
    second = RandomSearchStrategy(4)
    assert [first.propose(space) for _ in range(4)] == [second.propose(space) for _ in range(4)]


@pytest.mark.parametrize(
    "parameter",
    [
        ParameterDefinition("bad", "float", 1.0, 0.0),
        ParameterDefinition("bad", "integer", 0.1, 2),
        ParameterDefinition("bad", "categorical", choices=()),
        ParameterDefinition("bad", "unknown"),
    ],
)
def test_search_space_rejects_invalid_parameter_definitions(parameter) -> None:
    with pytest.raises(ScenarioValidationError):
        SearchSpace("pump", "sirfex-pump:1", parameters=(parameter,)).validate()


def test_tracking_objective_is_proxy_not_physical_failure() -> None:
    from sirfex.core.results import EpisodeResult

    result = EpisodeResult(
        environment_id="pump",
        controller_id="pump-rule",
        scenario={"environment_id": "pump", "parameters": {}, "seed": 0},
        seed=0,
        episode_reward=-1.0,
        episode_length=2,
        terminated=False,
        truncated=True,
        outcome="nominal",
        trace=[
            {"observation": [0.1, 0.5, 0.5]},
            {"observation": [0.2, 0.5, 0.5]},
        ],
    )
    score = PumpTrackingObjective().evaluate(result)
    assert score.value == pytest.approx(0.35)
    assert score.evidence_status == "inferred"
    assert score.is_failure is True


def test_tpe_strategy_can_propose_and_observe() -> None:
    strategy = TPESearchStrategy(1, "maximize")
    space = SearchSpace(
        "pump",
        "sirfex-pump:1",
        parameters=(ParameterDefinition("demand", "float", 0.1, 0.9),),
    )
    for _ in range(3):
        candidate = strategy.propose(space)
        assert 0.1 <= candidate["demand"] <= 0.9
        strategy.observe(float(np.float64(candidate["demand"])))
