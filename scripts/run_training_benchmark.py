"""Run an explicit, bounded Phase 3 CPU training benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sirfex.training.config import training_profile
from sirfex.training.persistence import TrainingStore
from sirfex.training.runner import run_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure a local CPU PPO training profile")
    parser.add_argument(
        "--profile", choices=["smoke", "quick", "standard", "extended"], default="smoke"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--store", default="artifacts/training-benchmarks")
    args = parser.parse_args()
    started = time.perf_counter()
    result = run_baseline(
        training_profile(args.profile, seed=args.seed), TrainingStore(Path(args.store))
    )
    result["benchmark"] = {
        "profile": args.profile,
        "seed": args.seed,
        "wall_clock_seconds": time.perf_counter() - started,
        "checkpoint_size_bytes": Path(result["manifest"]["checkpoint"]["path"]).stat().st_size,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
