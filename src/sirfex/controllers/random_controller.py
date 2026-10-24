from __future__ import annotations

from typing import Any

import numpy as np


class RandomController:
    def __init__(self, action_space: Any, seed: int | None = None) -> None:
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    @property
    def id(self) -> str:
        return "random"

    def reset(self) -> None:
        return None

    def act(self, observation: Any, deterministic: bool = True) -> Any:
        del observation, deterministic
        return self.action_space.sample()

    def metadata(self) -> dict[str, Any]:
        return {"kind": "random"}
