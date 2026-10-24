"""Explicit Phase 1 environment/controller reconstruction factories."""

from __future__ import annotations

from typing import Any

from sirfex.controllers import PumpRuleController, RandomController
from sirfex.environments import LunarLanderEnvironment, PumpEnvironment


def make_environment(environment_id: str) -> Any:
    if environment_id == "pump":
        return PumpEnvironment()
    if environment_id == "lunarlander":
        return LunarLanderEnvironment()
    raise ValueError(f"Unsupported environment: {environment_id}")


def make_controller(
    controller_id: str, environment: Any, config: dict[str, Any] | None = None
) -> Any:
    config = config or {}
    if controller_id == "random":
        return RandomController(environment.action_space, seed=config.get("seed"))
    if controller_id == "pump-rule" and environment.id == "pump":
        return PumpRuleController(
            low_level=float(config.get("low_level", 0.42)),
            high_level=float(config.get("high_level", 0.58)),
        )
    raise ValueError(f"Controller {controller_id!r} cannot be reconstructed for {environment.id!r}")
