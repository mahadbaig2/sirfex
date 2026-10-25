"""Calibrated affine feedback and fixed-command pump diagnostic references."""

import math

import numpy as np


class AnalyticalPumpController:
    id = "pump-analytical"

    def __init__(self, dt=0.1, response=0.25):
        if not math.isfinite(dt) or dt <= 0 or not 0 < response <= 1:
            raise ValueError("Positive dt and response in (0,1] are required")
        self.dt, self.response = dt, response

    def reset(self):
        pass

    def act(self, observation, deterministic=True):
        values = np.asarray(observation, dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("Analytical control requires six finite calibrated observations")
        h, target, demand, efficiency, capacity, leak = values
        effective = capacity * efficiency
        if effective <= 0:
            raise ValueError("Analytical control requires positive effective capacity")
        equilibrium = (demand + leak * target) / effective
        gain = (self.response / self.dt - leak) / effective
        return np.array([np.clip(equilibrium + gain * (target - h), 0.0, 1.0)], dtype=np.float32)

    def metadata(self):
        return {
            "kind": self.id,
            "version": "calibrated-feedback:1",
            "dt": self.dt,
            "response": self.response,
            "unclipped_error_pole": 1 - self.response,
        }


class FixedPumpController:
    id = "pump-fixed"

    def __init__(self, command=0.0):
        if not math.isfinite(command) or not 0 <= command <= 1:
            raise ValueError("Fixed command must be finite in [0,1]")
        self.command = command

    def reset(self):
        pass

    def act(self, observation, deterministic=True):
        return np.array([self.command], dtype=np.float32)

    def metadata(self):
        return {"kind": self.id, "command": self.command}
