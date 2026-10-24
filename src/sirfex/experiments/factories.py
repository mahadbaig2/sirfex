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
        return RandomController(environment.action_space, seed=config.get("seed", 0))
    if controller_id == "pump-rule" and environment.id == "pump":
        return PumpRuleController(
            low_level=float(config.get("low_level", 0.42)),
            high_level=float(config.get("high_level", 0.58)),
        )
    if controller_id == "ppo" and environment.id == "pump":
        from sirfex.training.checkpoints import CheckpointMetadata
        from sirfex.training.controller import PPOController

        checkpoint = config.get("checkpoint", config)
        metadata = CheckpointMetadata(
            **checkpoint
            | {
                "observation_shape": tuple(checkpoint["observation_shape"]),
                "action_shape": tuple(checkpoint["action_shape"]),
            }
        )
        return PPOController.load(metadata.path, metadata)
    raise ValueError(f"Controller {controller_id!r} cannot be reconstructed for {environment.id!r}")
