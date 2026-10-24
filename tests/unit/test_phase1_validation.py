import math

import numpy as np
import pytest

from sirfex.core import Scenario, ScenarioValidationError
from sirfex.environments import LunarLanderEnvironment, PumpEnvironment


@pytest.mark.parametrize(
    "parameters",
    [
        {"unknown": 1.0},
        {"horizon": True},
        {"dt": math.nan},
        {"efficiency": math.inf},
        {"min_level": 0.8, "max_level": 0.2},
        {"initial_level": 0.1, "min_level": 0.2},
    ],
)
def test_pump_rejects_invalid_scenarios(parameters: dict[str, object]) -> None:
    with pytest.raises(ScenarioValidationError):
        PumpEnvironment().validate_scenario(Scenario("pump", parameters=parameters))


def test_pump_rejects_invalid_actions_without_implicit_clipping() -> None:
    environment = PumpEnvironment()
    environment.reset(Scenario("pump"))
    with pytest.raises(ValueError):
        environment.step(np.array([1.1], dtype=np.float32))
    with pytest.raises(ValueError):
        environment.step(np.array([np.nan], dtype=np.float32))
    with pytest.raises(ValueError):
        environment.step(np.array([], dtype=np.float32))


def test_pump_can_explicitly_clip_actions() -> None:
    environment = PumpEnvironment(allow_action_clipping=True)
    environment.reset(Scenario("pump"))
    _, _, _, _, info = environment.step(np.array([1.1], dtype=np.float32))
    assert info["inflow"] == environment._active_parameters.pump_capacity


def test_stock_lunarlander_terminal_classification_is_unknown_without_evidence() -> None:
    environment = LunarLanderEnvironment()
    try:
        result = environment.evaluate_failure(10_000.0, True, {})
        assert result["outcome"] == "unknown"
        assert result["observed_outcome"] == "terminal_outcome_unavailable"
    finally:
        environment.close()
