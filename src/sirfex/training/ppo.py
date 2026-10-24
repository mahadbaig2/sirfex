"""Thin Stable-Baselines3 PPO orchestration layer."""

from __future__ import annotations

import importlib.metadata
import random
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
    env: Any, config: PPOConfig, *, seed: int, timesteps: int, checkpoint: str | None = None
):
    """Train or continue PPO; the optimizer remains entirely SB3-owned."""
    config.validate()
    if timesteps < 1:
        raise ValueError("timesteps must be positive")
    try:
        import torch
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError("Install the benchmark extra to train PPO") from exc
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
    if checkpoint:
        model = PPO.load(checkpoint, env=env, device="cpu")
        model.set_random_seed(seed)
    else:
        model = PPO("MlpPolicy", env, **kwargs)
    model.learn(total_timesteps=timesteps, reset_num_timesteps=checkpoint is None)
    return model
