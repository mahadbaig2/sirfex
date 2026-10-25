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
    original = np.asarray(action)
    space = policy_action_space(schema)
    if (
        original.shape != (1,)
        or not np.issubdtype(original.dtype, np.number)
        or np.iscomplexobj(original)
        or not np.isfinite(original).all()
        or (original < space.low).any()
        or (original > space.high).any()
    ):
        raise ValueError("Policy action violates its declared schema")
    value = original.astype(np.float32)
    return (value + 1.0) / 2.0 if schema == NORMALIZED else value
