"""Short, CPU-only PPO dependency and save/load smoke test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ppo"))
    args = parser.parse_args()
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        print(json.dumps({"status": "unavailable", "reason": str(exc)}, indent=2))
        return 0

    import gymnasium as gym

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make("LunarLander-v3", render_mode=None)
    path = args.output_dir / "ppo_smoke"
    started = time.perf_counter()
    try:
        model = PPO("MlpPolicy", env, device="cpu", n_steps=64, batch_size=64, verbose=0, seed=0)
        model.learn(total_timesteps=args.steps)
        model.save(path)
        PPO.load(path, env=env, device="cpu")
    finally:
        env.close()
    elapsed = time.perf_counter() - started
    print(
        json.dumps({"status": "passed", "steps": args.steps, "elapsed_seconds": elapsed}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
