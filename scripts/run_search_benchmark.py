"""Run a small fair random-vs-TPE search comparison."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sirfex.experiments.storage import ExperimentStore
from sirfex.search.comparison import compare_searches
from sirfex.search.objectives import make_objective
from sirfex.search.persistence import SearchStore
from sirfex.search.runner import SearchRunner
from sirfex.search.space import SearchSpace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/search/pump_stress.json"))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/search-benchmark"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = json.load(stream)
    space = SearchSpace.from_dict(config)
    experiment_store = ExperimentStore(args.output / "experiments")
    search_store = SearchStore(args.output / "searches")
    started = time.perf_counter()
    results = {}
    for strategy in ("random", "tpe"):
        results[strategy] = SearchRunner(
            space,
            make_objective(config["objective"]),
            config["controller_id"],
            experiment_store=experiment_store,
            search_store=search_store,
        ).run(strategy, trials=args.trials, strategy_seed=args.seed, scenario_seeds=[0])
    first_id = results["random"]["summary"]["search_id"]
    second_id = results["tpe"]["summary"]["search_id"]
    report = {
        "config": str(args.config),
        "trials_per_strategy": args.trials,
        "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - started,
        "random": results["random"]["summary"],
        "tpe": results["tpe"]["summary"],
        "comparison": compare_searches(search_store, first_id, second_id),
        "caveat": "One bounded run is descriptive evidence, not proof that TPE is superior.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
