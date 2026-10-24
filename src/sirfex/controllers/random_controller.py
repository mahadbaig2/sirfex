from __future__ import annotations

from typing import Any

import numpy as np


class RandomController:
    def __init__(self, action_space: Any, seed: int | None = None) -> None:
        self.action_space = action_space
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @property
    def id(self) -> str:
        return "random"

    def reset(self) -> None:
        return None

    def act(self, observation: Any, deterministic: bool = True) -> Any:
        del observation, deterministic
        if hasattr(self.action_space, "n"):
            return int(self.rng.integers(self.action_space.n))
        if hasattr(self.action_space, "low") and hasattr(self.action_space, "high"):
            return self.rng.uniform(self.action_space.low, self.action_space.high).astype(
                self.action_space.dtype
            )
        return self.action_space.sample()

    def metadata(self) -> dict[str, Any]:
        return {"kind": "random", "seed": self.seed, "rng": "numpy.default_rng"}
