import numpy as np
import pytest

from sirfex.controllers.analytical_pump import AnalyticalPumpController
from sirfex.core import Scenario
from sirfex.training.actions import DIRECT, NORMALIZED, to_physical
from sirfex.training.pump_gym import PumpGymEnv
from sirfex.training.rewards import LEGACY, SURVIVAL, components


@pytest.mark.parametrize("value,physical", [(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0), (0.4, 0.7)])
def test_normalized_action_mapping(value, physical):
    assert to_physical([value], NORMALIZED)[0] == pytest.approx(physical)


@pytest.mark.parametrize("value", [[2.0], [float("nan")], [], [0.1, 0.2]])
def test_invalid_actions_are_not_silently_clipped(value):
    with pytest.raises(ValueError):
        to_physical(value, NORMALIZED)


def test_reward_and_action_changes_preserve_physical_trajectory():
    scenario = Scenario("pump", {"horizon": 20})
    a = PumpGymEnv([scenario], policy_action_schema=DIRECT, reward_schema=LEGACY)
    b = PumpGymEnv([scenario], policy_action_schema=NORMALIZED, reward_schema=SURVIVAL)
    a.reset()
    b.reset()
    for _ in range(20):
        first, second = (
            a.step(np.array([0.5], dtype=np.float32)),
            b.step(np.array([0.0], dtype=np.float32)),
        )
        np.testing.assert_array_equal(first[0], second[0])
        assert first[2:4] == second[2:4]
        assert first[4]["constraint_violation"] == second[4]["constraint_violation"]
        assert first[1] != second[1]
        if first[2] or first[3]:
            break


def test_revised_reward_terminal_cost_and_horizon_semantics():
    terms = components(SURVIVAL, 0.05, 0.5, 0.0, 0.01, True)
    assert terms["total"] == pytest.approx(1 - 4 * 0.45**2 - 0.1 - 1)
    env = PumpGymEnv([Scenario("pump", {"horizon": 1})], reward_schema=SURVIVAL)
    env.reset()
    _, _, terminated, truncated, info = env.step(np.array([0.625], dtype=np.float32))
    assert not terminated and truncated
    assert info["reward_components"]["terminal_penalty"] == 0
    with pytest.raises(ValueError):
        PumpGymEnv(termination_schema="continue-and-erase-failure")


@pytest.mark.parametrize("demand,leak", [(0.3, 0.0), (0.5, 0.1), (0.7, 0.02)])
def test_analytical_equilibrium_and_feedback(demand, leak):
    controller = AnalyticalPumpController()
    observation = np.array([0.5, 0.5, demand, 1.0, 0.8, leak])
    assert controller.act(observation)[0] == pytest.approx((demand + leak * 0.5) / 0.8)
    observation[0] = 0.49
    action = float(controller.act(observation)[0])
    next_level = 0.49 + 0.1 * (0.8 * action - demand - leak * 0.49)
    assert 0.5 - next_level == pytest.approx(0.75 * (0.5 - 0.49), abs=1e-8)


def test_analytical_capacity_and_saturation():
    controller = AnalyticalPumpController()
    with pytest.raises(ValueError, match="capacity"):
        controller.act([0.5, 0.5, 0.5, 0.0, 0.8, 0.0])
    assert controller.act([0.1, 0.5, 1.0, 1.0, 0.1, 0.0])[0] == 1
    assert controller.act([0.9, 0.5, 0.0, 1.0, 0.1, 0.0])[0] == 0


def test_survival_reward_separates_short_failure_from_equilibrium():
    safe = sum(
        0.99**i * components(SURVIVAL, 0.5, 0.5, 0.625, 0, False)["total"] for i in range(200)
    )
    failing = sum(
        0.99**i * components(SURVIVAL, 0.5 - 0.05 * (i + 1), 0.5, 0, 0, i == 9)["total"]
        for i in range(10)
    )
    assert safe > failing
