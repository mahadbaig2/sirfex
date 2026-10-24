from __future__ import annotations

from typing import Any

import numpy as np


class PumpRuleController:
    """A deterministic hysteresis controller for the simplified pump model."""

    def __init__(self, low_level: float = 0.42, high_level: float = 0.58) -> None:
        self.low_level = low_level
        self.high_level = high_level
        self._command = 0.0

    @property
    def id(self) -> str:
        return "pump-rule"

    def reset(self) -> None:
        self._command = 0.0

    def act(self, observation: Any, deterministic: bool = True) -> np.ndarray:
        del deterministic
        level = float(observation[0])
        if level <= self.low_level:
            self._command = 1.0
        elif level >= self.high_level:
            self._command = 0.0
        return np.array([self._command], dtype=np.float32)

    def metadata(self) -> dict[str, Any]:
        return {"kind": "rule-based", "low_level": self.low_level, "high_level": self.high_level}
