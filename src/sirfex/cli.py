from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from sirfex.controllers import PumpRuleController, RandomController
from sirfex.core import EvaluationRunner, Scenario
from sirfex.environments import LunarLanderEnvironment, PumpEnvironment


def _make_environment(name: str) -> Any:
    if name == "lunarlander":
        return LunarLanderEnvironment()
    if name == "pump":
        return PumpEnvironment()
    raise ValueError(f"Unknown environment: {name}")


def _make_controller(name: str, environment: Any) -> Any:
    if name == "random":
        return RandomController(environment.action_space)
    if name == "rule-based" and environment.id == "pump":
        return PumpRuleController()
    raise ValueError(f"Controller {name!r} is not available for {environment.id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sirfex", description="Local-first controller evaluation tools"
    )
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("environments", help="list available Phase 0 environments")
    evaluate = subparsers.add_parser("evaluate", help="run one headless episode")
    evaluate.add_argument("--env", choices=["lunarlander", "pump"], required=True)
    evaluate.add_argument("--controller", choices=["random", "rule-based"], required=True)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--max-steps", type=int, default=1_000)
    benchmark = subparsers.add_parser("benchmark", help="run the Phase 0 benchmark suite")
    benchmark.add_argument("--profile", choices=["quick", "standard"], default="quick")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )

    if args.command == "environments":
        print("lunarlander\npump")
        return 0
    if args.command == "evaluate":
        environment = _make_environment(args.env)
        try:
            controller = _make_controller(args.controller, environment)
            result = EvaluationRunner(max_steps=args.max_steps).run_episode(
                environment,
                controller,
                Scenario(environment_id=environment.id, seed=args.seed),
                record_trace=False,
            )
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        finally:
            environment.close()
        return 0
    if args.command == "benchmark":
        from sirfex.benchmarks.runner import run_benchmark

        print(json.dumps(run_benchmark(args.profile), indent=2, sort_keys=True))
        return 0
    parser.error("unknown command")
    return 2
