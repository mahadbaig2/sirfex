"""Versioned pump sensor contracts; physical transitions remain in pump.py."""

import gymnasium as gym
import numpy as np

from sirfex.environments.pump import PumpEnvironment

MINIMAL = "pump-minimal:1"
EXTENDED = "pump-calibrated:1"


def observation_space(schema: str):
    if schema == MINIMAL:
        return gym.spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
    if schema == EXTENDED:
        # Explicit calibrated-plant envelope; reject larger parameters, never clip them.
        return gym.spaces.Box(
            np.zeros(6, dtype=np.float32), np.array([1, 1, 1, 1, 10, 10], dtype=np.float32)
        )
    raise ValueError(f"Unsupported observation schema: {schema}")


class PumpObservationEnvironment(PumpEnvironment):
    def __init__(self, schema: str = MINIMAL):
        super().__init__()
        self.set_observation_schema(schema)

    def set_observation_schema(self, schema: str):
        self.observation_space = observation_space(schema)
        self.observation_schema = schema

    def reset(self, scenario):
        values = self.validate_scenario(scenario)
        if self.observation_schema == EXTENDED and (
            values["pump_capacity"] > 10 or values["leak"] > 10
        ):
            raise ValueError("Calibrated observation schema requires capacity and leak <= 10")
        return super().reset(scenario)

    def _observation(self, values=None):
        base = super()._observation(values)
        if self.observation_schema == MINIMAL:
            return base
        p = self._active_parameters
        return np.concatenate(
            (base, np.array([p.efficiency, p.pump_capacity, p.leak], dtype=np.float32))
        )
