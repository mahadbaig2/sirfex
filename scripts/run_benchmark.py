from __future__ import annotations

import argparse
import json
from pathlib import Path

from sirfex.benchmarks.runner import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["quick", "standard"], default="quick")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmarks"))
    args = parser.parse_args()
    report = run_benchmark(args.profile, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
