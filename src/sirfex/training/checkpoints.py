"""Trusted local PPO checkpoint metadata and integrity checks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointMetadata:
    checkpoint_id: str
    training_run_id: str
    path: str
    sha256: str
    environment_id: str
    observation_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    ppo_config: dict[str, Any]
    training_seed: int
    training_timesteps: int
    dependencies: dict[str, str]
    observation_schema: str = "pump-minimal:1"
    reward_schema: str = "pump-violation-magnitude:1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "observation_shape": list(self.observation_shape),
            "action_shape": list(self.action_shape),
        }


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path | str, expected_sha256: str) -> None:
    actual = sha256_file(Path(path))
    if actual != expected_sha256:
        raise ValueError(f"Checkpoint hash mismatch: expected {expected_sha256}, got {actual}")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
