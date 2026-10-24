from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from sirfex.controllers import PumpRuleController, RandomController
from sirfex.core import EvaluationRunner, Scenario
from sirfex.experiments.comparison import compare_experiments
from sirfex.experiments.factories import make_environment
from sirfex.experiments.replay import replay_experiment
from sirfex.experiments.service import load_scenario_file, run_and_save_experiment
from sirfex.experiments.storage import ExperimentStore, StorageError


def _make_environment(name: str) -> Any:
    return make_environment(name)


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
    scenarios = subparsers.add_parser("scenarios", help="inspect and validate scenarios")
    scenarios_sub = scenarios.add_subparsers(dest="scenarios_command", required=True)
    scenario_list = scenarios_sub.add_parser("list")
    scenario_list.add_argument("--env", choices=["lunarlander", "pump"], required=True)
    scenario_validate = scenarios_sub.add_parser("validate")
    scenario_validate.add_argument("--file", required=True)
    experiments = subparsers.add_parser("experiments", help="persist and inspect experiments")
    experiments.add_argument("--store", default="artifacts/experiments")
    experiments_sub = experiments.add_subparsers(dest="experiments_command", required=True)
    experiment_run = experiments_sub.add_parser("run")
    experiment_run.add_argument("--scenario", required=True)
    experiment_run.add_argument("--controller", choices=["random", "pump-rule"], required=True)
    experiment_run.add_argument("--max-steps", type=int, default=1_000)
    experiments_sub.add_parser("list").add_argument("--env")
    experiments_sub.choices["list"].add_argument("--controller")
    experiment_show = experiments_sub.add_parser("show")
    experiment_show.add_argument("experiment_id")
    experiment_replay = experiments_sub.add_parser("replay")
    experiment_replay.add_argument("experiment_id")
    experiment_replay.add_argument("--tolerance", type=float, default=1e-9)
    experiment_compare = experiments_sub.add_parser("compare")
    experiment_compare.add_argument("first_id")
    experiment_compare.add_argument("second_id")
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
    if args.command == "scenarios":
        if args.scenarios_command == "list":
            environment = _make_environment(args.env)
            try:
                print(json.dumps(environment.scenario_spec(), indent=2, sort_keys=True))
            finally:
                environment.close()
            return 0
        if args.scenarios_command == "validate":
            scenario = load_scenario_file(args.file)
            environment = _make_environment(scenario.environment_id)
            try:
                resolved = environment.validate_scenario(scenario)
                print(json.dumps({"valid": True, "resolved_parameters": resolved}, indent=2))
            finally:
                environment.close()
            return 0
    if args.command == "experiments":
        store = ExperimentStore(args.store)
        try:
            if args.experiments_command == "run":
                scenario = load_scenario_file(args.scenario)
                manifest = run_and_save_experiment(
                    store, scenario, args.controller, max_steps=args.max_steps
                )
                print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
                return 0
            if args.experiments_command == "list":
                print(
                    json.dumps(
                        [
                            item.to_dict()
                            for item in store.list(
                                environment_id=args.env, controller_id=args.controller
                            )
                        ],
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            if args.experiments_command == "show":
                stored = store.load(args.experiment_id)
                print(
                    json.dumps(
                        {"manifest": stored.manifest.to_dict(), "result": stored.result},
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            if args.experiments_command == "replay":
                print(
                    json.dumps(
                        replay_experiment(store, args.experiment_id, args.tolerance).to_dict(),
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            if args.experiments_command == "compare":
                print(
                    json.dumps(
                        compare_experiments(store, args.first_id, args.second_id).to_dict(),
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
        except (StorageError, ValueError, OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
    parser.error("unknown command")
    return 2
