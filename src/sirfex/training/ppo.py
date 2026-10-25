"""Thin Stable-Baselines3 PPO orchestration layer."""

from __future__ import annotations

import importlib.metadata
import random
import time
from typing import Any

import numpy as np

from sirfex.training.config import PPOConfig


def dependency_versions() -> dict[str, str]:
    versions = {}
    for name in ("sirfex", "numpy", "gymnasium", "stable-baselines3", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def train_ppo(
    env: Any,
    config: PPOConfig,
    *,
    seed: int,
    timesteps: int,
    checkpoint: str | None = None,
    model: Any = None,
    wall_seconds: float | None = None,
    expected_sha256: str | None = None,
    on_model: Any = None,
):
    """Train or continue PPO; the optimizer remains entirely SB3-owned."""
    config.validate()
    if timesteps < 1:
        raise ValueError("timesteps must be positive")
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:
        raise ImportError("Install the benchmark extra to train PPO") from exc
    if model is None:
        random.seed(seed)
        np.random.seed(seed)
    torch.set_num_threads(config.torch_threads)
    policy_kwargs = {"net_arch": list(config.net_arch)}
    kwargs = {
        "learning_rate": config.learning_rate,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "ent_coef": config.ent_coef,
        "vf_coef": config.vf_coef,
        "policy_kwargs": policy_kwargs,
        "seed": seed,
        "device": "cpu",
        "verbose": 0,
    }
    if checkpoint and model is not None:
        raise ValueError("Supply an in-memory model or a checkpoint, not both")
    fresh = checkpoint is None and model is None
    contract = {
        "policy_action_schema": getattr(env, "policy_action_schema", "pump-direct:1"),
        "reward_schema": getattr(env, "reward_schema", "pump-violation-magnitude:1"),
        "termination_schema": getattr(env, "termination_schema", "pump-terminate-on-violation:1"),
    }
    if checkpoint:
        if expected_sha256 is not None:
            from sirfex.training.checkpoints import verify_checkpoint

            verify_checkpoint(checkpoint, expected_sha256)
        model = PPO.load(checkpoint, env=env, device="cpu")
        for key in (
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "ent_coef",
            "vf_coef",
        ):
            if getattr(model, key) != getattr(config, key):
                raise ValueError(f"Continuation PPO configuration mismatch: {key}")
        model.set_random_seed(seed)
        if model.policy_kwargs.get("net_arch") != list(config.net_arch):
            raise ValueError("Continuation PPO configuration mismatch: net_arch")
    elif model is None:
        model = PPO("MlpPolicy", env, **kwargs)
    if not fresh:
        legacy = {
            "policy_action_schema": "pump-direct:1",
            "reward_schema": "pump-violation-magnitude:1",
            "termination_schema": "pump-terminate-on-violation:1",
        }
        if getattr(model, "sirfex_training_contract", legacy) != contract:
            raise ValueError("Continuation reward/action/termination contract mismatch")
    model.sirfex_training_contract = contract
    if wall_seconds is not None and wall_seconds <= 0:
        raise ValueError("Wall-clock allowance must be positive")
    started = time.perf_counter()
    if on_model is not None:
        on_model(model)

    class Deadline(BaseCallback):
        def _on_step(self):
            return wall_seconds is None or time.perf_counter() - started < wall_seconds

    model.learn(total_timesteps=timesteps, reset_num_timesteps=fresh, callback=Deadline())
    return model
