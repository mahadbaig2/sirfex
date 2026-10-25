"""Versioned policy coordinates; the physical pump command always remains [0,1]."""

import gymnasium as gym
import numpy as np

DIRECT = "pump-direct:1"
NORMALIZED = "pump-normalized:1"


def policy_action_space(schema=DIRECT):
    if schema not in (DIRECT, NORMALIZED):
        raise ValueError(f"Unknown policy action schema: {schema}")
    return gym.spaces.Box(-1.0 if schema == NORMALIZED else 0.0, 1.0, shape=(1,), dtype=np.float32)


def to_physical(action, schema=DIRECT):
    value = np.asarray(action, dtype=np.float32)
    if not policy_action_space(schema).contains(value):
        raise ValueError("Policy action violates its declared schema")
    return (value + 1.0) / 2.0 if schema == NORMALIZED else value
