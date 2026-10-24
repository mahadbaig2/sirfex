"""Validated, serializable Phase 3 training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 32
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    net_arch: tuple[int, ...] = (64, 64)
    device: str = "cpu"
    torch_threads: int = 4

    def validate(self) -> None:
        if self.learning_rate <= 0 or self.n_steps < 2 or self.batch_size < 2:
            raise ValueError("PPO learning rate, rollout steps, and batch size must be positive")
        if self.n_steps % self.batch_size != 0:
            raise ValueError("n_steps must be divisible by batch_size for deterministic rollouts")
        if self.n_epochs < 1 or not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid PPO discount or epoch configuration")
        if self.ent_coef < 0 or self.vf_coef < 0 or not self.net_arch:
            raise ValueError("Invalid PPO coefficient or network configuration")
        if self.device != "cpu":
            raise ValueError("Phase 3 currently supports CPU PPO only")
        if self.torch_threads < 1:
            raise ValueError("torch_threads must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "net_arch": list(self.net_arch)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PPOConfig:
        config = cls(
            learning_rate=float(value.get("learning_rate", 3e-4)),
            n_steps=int(value.get("n_steps", 128)),
            batch_size=int(value.get("batch_size", 32)),
            n_epochs=int(value.get("n_epochs", 4)),
            gamma=float(value.get("gamma", 0.99)),
            gae_lambda=float(value.get("gae_lambda", 0.95)),
            ent_coef=float(value.get("ent_coef", 0.0)),
            vf_coef=float(value.get("vf_coef", 0.5)),
            net_arch=tuple(int(item) for item in value.get("net_arch", [64, 64])),
            device=str(value.get("device", "cpu")),
            torch_threads=int(value.get("torch_threads", 4)),
        )
        config.validate()
        return config


@dataclass(frozen=True)
class TrainingConfig:
    profile: str = "smoke"
    training_timesteps: int = 256
    training_seed: int = 0
    max_episode_steps: int = 200
    normal_scenarios: tuple[dict[str, Any], ...] = ()
    curriculum_ratio: float = 0.3
    discovery_limit: int = 8
    regression_tolerances: dict[str, float] = field(
        default_factory=lambda: {
            "nominal_tracking_error_degradation": 0.10,
            "held_out_violation_frequency_degradation": 0.0,
            "held_out_violation_magnitude_degradation": 0.0,
        }
    )
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def validate(self) -> None:
        if self.profile not in {"smoke", "quick", "standard", "extended"}:
            raise ValueError("Unsupported training profile")
        if self.training_timesteps < 1 or self.max_episode_steps < 1:
            raise ValueError("Training timesteps and episode steps must be positive")
        if not 0 <= self.curriculum_ratio <= 1:
            raise ValueError("curriculum_ratio must be between 0 and 1")
        if self.discovery_limit < 0:
            raise ValueError("discovery_limit cannot be negative")
        self.ppo.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "ppo": self.ppo.to_dict(),
            "normal_scenarios": list(self.normal_scenarios),
        }


def training_profile(
    name: str, *, seed: int = 0, overrides: dict[str, Any] | None = None
) -> TrainingConfig:
    timesteps = {"smoke": 256, "quick": 2_048, "standard": 10_000, "extended": 50_000}
    rollout = {"smoke": (64, 32), "quick": (128, 32), "standard": (256, 64), "extended": (512, 128)}
    if name not in timesteps:
        raise ValueError(f"Unsupported training profile: {name}")
    n_steps, batch_size = rollout[name]
    config = TrainingConfig(
        profile=name,
        training_timesteps=timesteps[name],
        training_seed=seed,
        ppo=PPOConfig(n_steps=n_steps, batch_size=batch_size),
    )
    if overrides:
        values = config.to_dict()
        values.update(overrides)
        values["ppo"] = {**config.ppo.to_dict(), **overrides.get("ppo", {})}
        config = TrainingConfig(
            profile=str(values["profile"]),
            training_timesteps=int(values["training_timesteps"]),
            training_seed=int(values["training_seed"]),
            max_episode_steps=int(values["max_episode_steps"]),
            normal_scenarios=tuple(values.get("normal_scenarios", [])),
            curriculum_ratio=float(values.get("curriculum_ratio", 0.3)),
            discovery_limit=int(values.get("discovery_limit", 8)),
            regression_tolerances=dict(
                values.get("regression_tolerances", config.regression_tolerances)
            ),
            ppo=PPOConfig.from_dict(values["ppo"]),
        )
    config.validate()
    return config
