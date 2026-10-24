import numpy as np

from sirfex.core import Scenario
from sirfex.environments import PumpEnvironment, PumpParameters


def test_pump_seeded_reset_is_deterministic() -> None:
    scenario = Scenario("pump", seed=7)
    first = PumpEnvironment()
    second = PumpEnvironment()
    try:
        assert np.array_equal(first.reset(scenario)[0], second.reset(scenario)[0])
    finally:
        first.close()
        second.close()


def test_pump_mass_balance_without_leak() -> None:
    environment = PumpEnvironment(PumpParameters(demand=0.2, pump_capacity=0.8, dt=0.1))
    try:
        initial = float(environment.reset(Scenario("pump"))[0][0])
        observation, _, _, _, info = environment.step(np.array([0.5], dtype=np.float32))
        expected = initial + 0.1 * (0.8 * 0.5 - 0.2)
        assert np.isclose(observation[0], expected)
        assert np.isclose(info["inflow"] - info["outflow"], 0.2)
    finally:
        environment.close()


def test_pump_rejects_invalid_initial_level() -> None:
    environment = PumpEnvironment()
    try:
        try:
            environment.reset(Scenario("pump", parameters={"initial_level": 2.0}))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid level was accepted")
    finally:
        environment.close()
