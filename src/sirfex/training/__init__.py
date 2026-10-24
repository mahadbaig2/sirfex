"""Optional CPU PPO training and robustness-evaluation infrastructure.

Stable-Baselines3 is imported lazily by the training functions so the core
evaluation, search, and visualization packages remain dependency-isolated.
"""

from sirfex.training.config import PPOConfig, TrainingConfig, training_profile

__all__ = ["PPOConfig", "TrainingConfig", "training_profile"]
