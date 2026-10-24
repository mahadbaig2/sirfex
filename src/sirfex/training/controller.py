"""SIRFEX controller adapter for trusted local PPO checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sirfex.training.checkpoints import CheckpointMetadata, verify_checkpoint


class PPOController:
    id = "ppo"

    def __init__(self, model: Any, metadata: CheckpointMetadata) -> None:
        self._model = model
        self._metadata = metadata

    @classmethod
    def load(cls, path: Path | str, metadata: CheckpointMetadata) -> PPOController:
        checkpoint = Path(path).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint}")
        verify_checkpoint(checkpoint, metadata.sha256)
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError("Install the benchmark extra to load PPO checkpoints") from exc
        model = PPO.load(str(checkpoint), device="cpu")
        if tuple(model.observation_space.shape or ()) != tuple(metadata.observation_shape):
            raise ValueError("Checkpoint observation shape does not match metadata")
        if tuple(model.action_space.shape or ()) != tuple(metadata.action_shape):
            raise ValueError("Checkpoint action shape does not match metadata")
        return cls(model, metadata)

    def reset(self) -> None:
        return None

    def act(self, observation: Any, deterministic: bool = True) -> np.ndarray:
        value, _ = self._model.predict(
            np.asarray(observation, dtype=np.float32), deterministic=deterministic
        )
        action = np.asarray(value, dtype=np.float32)
        if action.shape != self._metadata.action_shape:
            raise ValueError(
                f"PPO returned action shape {action.shape}, expected {self._metadata.action_shape}"
            )
        return action

    def metadata(self) -> dict[str, Any]:
        return {"kind": "stable-baselines3-ppo", "checkpoint": self._metadata.to_dict()}
