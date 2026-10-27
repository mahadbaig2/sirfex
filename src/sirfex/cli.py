from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np

from sirfex.controllers import PumpRuleController, RandomController
from sirfex.core import EvaluationRunner, Scenario
from sirfex.experiments.comparison import compare_experiments
from sirfex.experiments.factories import make_environment
from sirfex.experiments.replay import replay_experiment
from sirfex.experiments.service import load_scenario_file, run_and_save_experiment
from sirfex.experiments.storage import ExperimentStore, StorageError
from sirfex.search.comparison import compare_searches
from sirfex.search.objectives import make_objective
from sirfex.search.persistence import SearchStorageError, SearchStore
from sirfex.search.runner import SearchRunner
from sirfex.search.space import SearchSpace


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
    benchmark.add_argument("--profile", choices=["quick", "standard", "stress"], default="quick")
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
    experiment_run.add_argument(
        "--controller",
        choices=["random", "pump-rule", "pump-analytical", "pump-fixed", "ppo"],
        required=True,
    )
    experiment_run.add_argument(
        "--controller-config",
        default=None,
        help="JSON controller configuration, including trusted PPO checkpoint metadata",
    )
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
    search = subparsers.add_parser("search", help="discover risky valid scenarios")
    search.add_argument("--experiment-store", default="artifacts/experiments")
    search.add_argument("--search-store", default="artifacts/searches")
    search.add_argument("--training-store", default="artifacts/training")
    search_sub = search.add_subparsers(dest="search_command", required=True)
    avoidable = search_sub.add_parser("avoidable")
    avoidable.add_argument("--config", required=True)
    avoidable.add_argument("--baseline")
    search_spaces = search_sub.add_parser("spaces")
    search_spaces.add_argument("--env", choices=["lunarlander", "pump"], required=True)
    search_validate = search_sub.add_parser("validate")
    search_validate.add_argument("--file", required=True)
    search_run = search_sub.add_parser("run")
    search_run.add_argument("--config", required=True)
    search_run.add_argument("--strategy", choices=["random", "tpe"], required=True)
    search_run.add_argument("--trials", type=int, required=True)
    search_run.add_argument("--seed", type=int, default=0)
    search_run.add_argument("--scenario-seeds", default="0")
    search_run.add_argument("--held-out-seeds", default="")
    search_sub.add_parser("list")
    search_show = search_sub.add_parser("show")
    search_show.add_argument("search_id")
    search_discoveries = search_sub.add_parser("discoveries")
    search_discoveries.add_argument("search_id")
    search_replay = search_sub.add_parser("replay")
    search_replay.add_argument("discovery_id")
    search_compare = search_sub.add_parser("compare")
    search_compare.add_argument("first_id")
    search_compare.add_argument("second_id")
    train = subparsers.add_parser("train", help="run optional local PPO robustness experiments")
    train.add_argument("--store", default="artifacts/training")
    train.add_argument("--search-store", default="artifacts/searches")
    train.add_argument("--experiment-store", default="artifacts/experiments")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    for command in ("diagnose-policy", "diagnose-reward", "competence"):
        study = train_sub.add_parser(command)
        study.add_argument("--config", required=True)
        if command == "diagnose-policy":
            study.add_argument("--checkpoint")
        if command == "competence":
            study.add_argument("--allow-extended", action="store_true")
    matched = train_sub.add_parser("robustness-study")
    matched_source = matched.add_mutually_exclusive_group(required=True)
    matched_source.add_argument("--config")
    matched_source.add_argument("--study")
    matched.add_argument("--prepare-only", action="store_true")
    phase34 = train_sub.add_parser("phase-3-4")
    phase34_source = phase34.add_mutually_exclusive_group(required=True)
    phase34_source.add_argument("--config")
    phase34_source.add_argument("--study")
    phase34.add_argument("--stage", choices=("prepare", "screen", "confirm"), required=True)
    phase34.add_argument("--allow-extended", action="store_true")
    for command in ("diagnose", "feasibility"):
        train_sub.add_parser(command).add_argument("--search", action="append")
    for command in ("learning-curve", "validate"):
        study = train_sub.add_parser(command)
        study.add_argument("--config", required=True)
        study.add_argument("--search", action="append")
        study.add_argument("--allow-extended", action="store_true")
    train_sub.add_parser("report").add_argument("training_run_id")
    train_baseline = train_sub.add_parser("baseline")
    train_baseline.add_argument("--config", required=True)
    train_baseline.add_argument("--seed", type=int, default=0)
    train_improve = train_sub.add_parser("improve")
    train_improve.add_argument("--baseline", required=True)
    train_improve.add_argument("--search", required=True)
    train_improve.add_argument("--config", required=True)
    train_improve.add_argument("--seed", type=int, default=0)
    train_sub.add_parser("list")
    train_show = train_sub.add_parser("show")
    train_show.add_argument("training_run_id")
    train_evaluate = train_sub.add_parser("evaluate")
    train_evaluate.add_argument("training_run_id")
    train_compare = train_sub.add_parser("compare")
    train_compare.add_argument("training_run_id")
    train_verify = train_sub.add_parser("verify")
    train_verify.add_argument("training_run_id")
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
                controller_config = None
                if args.controller_config:
                    with open(args.controller_config, encoding="utf-8") as stream:
                        controller_config = json.load(stream)
                manifest = run_and_save_experiment(
                    store,
                    scenario,
                    args.controller,
                    max_steps=args.max_steps,
                    controller_config=controller_config,
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
    if args.command == "search":
        experiment_store = ExperimentStore(args.experiment_store)
        search_store = SearchStore(args.search_store)
        try:
            if args.search_command == "avoidable":
                from sirfex.search.avoidable import run_avoidable
                from sirfex.training.persistence import TrainingStore

                with open(args.config, encoding="utf-8") as stream:
                    config = json.load(stream)
                result = run_avoidable(
                    config,
                    experiment_store,
                    search_store,
                    TrainingStore(args.training_store),
                    baseline=args.baseline,
                )
                print(json.dumps(result, indent=2))
                return 0 if result["status"] == "completed" else 1
            if args.search_command == "spaces":
                environment = _make_environment(args.env)
                try:
                    print(json.dumps(environment.scenario_spec(), indent=2, sort_keys=True))
                finally:
                    environment.close()
                return 0
            if args.search_command == "validate":
                with open(args.file, encoding="utf-8") as stream:
                    config = json.load(stream)
                space = SearchSpace.from_dict(config)
                environment = _make_environment(space.environment_id)
                try:
                    sample = space.scenario(space.sample_parameters(np.random.default_rng(0)), 0)
                    environment.validate_scenario(sample)
                finally:
                    environment.close()
                print(
                    json.dumps(
                        {"valid": True, "search_space": space.to_dict()}, indent=2, sort_keys=True
                    )
                )
                return 0
            if args.search_command == "run":
                with open(args.config, encoding="utf-8") as stream:
                    config = json.load(stream)
                space = SearchSpace.from_dict(config)
                runner = SearchRunner(
                    space,
                    make_objective(config["objective"]),
                    config["controller_id"],
                    experiment_store=experiment_store,
                    search_store=search_store,
                )
                scenario_seeds = [
                    int(value) for value in args.scenario_seeds.split(",") if value.strip()
                ]
                held_out_seeds = [
                    int(value) for value in args.held_out_seeds.split(",") if value.strip()
                ]
                print(
                    json.dumps(
                        runner.run(
                            args.strategy,
                            trials=args.trials,
                            strategy_seed=args.seed,
                            scenario_seeds=scenario_seeds,
                            held_out_seeds=held_out_seeds,
                        ),
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            if args.search_command == "list":
                print(json.dumps(search_store.list(), indent=2, sort_keys=True))
                return 0
            if args.search_command == "show":
                print(json.dumps(search_store.load(args.search_id), indent=2, sort_keys=True))
                return 0
            if args.search_command == "discoveries":
                print(
                    json.dumps(
                        search_store.load(args.search_id)["discoveries"], indent=2, sort_keys=True
                    )
                )
                return 0
            if args.search_command == "replay":
                search_id, discovery = search_store.find_discovery(args.discovery_id)
                statuses = [
                    replay_experiment(experiment_store, experiment_id).to_dict()
                    for experiment_id in discovery["experiment_ids"]
                ]
                search_store.save_verification(
                    search_id,
                    args.discovery_id,
                    {
                        "discovery_id": args.discovery_id,
                        "search_id": search_id,
                        "verification": statuses,
                    },
                )
                print(
                    json.dumps(
                        {
                            "search_id": search_id,
                            "discovery_id": args.discovery_id,
                            "verification": statuses,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            if args.search_command == "compare":
                print(
                    json.dumps(
                        compare_searches(search_store, args.first_id, args.second_id),
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
        except (SearchStorageError, StorageError, ValueError, OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
    if args.command == "train":
        from sirfex.training.checkpoints import verify_checkpoint
        from sirfex.training.config import training_profile
        from sirfex.training.persistence import TrainingStorageError, TrainingStore
        from sirfex.training.runner import run_baseline, run_improvement

        training_store = TrainingStore(args.store)
        try:
            if args.train_command == "phase-3-4":
                from sirfex.training.phase34 import confirm, prepare, screen

                if args.stage == "prepare":
                    if not args.config:
                        raise ValueError("Preparation requires --config")
                    with open(args.config, encoding="utf-8") as stream:
                        value = json.load(stream)
                    result = prepare(
                        value,
                        training_store,
                        SearchStore(args.search_store),
                        ExperimentStore(args.experiment_store),
                        allow_extended=args.allow_extended,
                    )
                else:
                    if not args.study:
                        raise ValueError("Screening and confirmation require --study")
                    result = (
                        screen(args.study, training_store)
                        if args.stage == "screen"
                        else confirm(
                            args.study, training_store, ExperimentStore(args.experiment_store)
                        )
                    )
                print(json.dumps(result, indent=2))
                return (
                    0
                    if result["status"]
                    in {"prepared", "selected", "no_eligible_curriculum", "completed"}
                    else 1
                )
            if args.train_command == "robustness-study":
                from sirfex.training.matched_study import prepare_study, run_prepared_study

                if args.study and args.prepare_only:
                    raise ValueError("--prepare-only applies to a new configuration")
                if args.config:
                    with open(args.config, encoding="utf-8") as stream:
                        value = json.load(stream)
                    prepared = prepare_study(
                        value,
                        training_store,
                        SearchStore(args.search_store),
                        ExperimentStore(args.experiment_store),
                    )
                    if args.prepare_only:
                        print(json.dumps(prepared, indent=2))
                        return 0
                    run_id = prepared["training_run_id"]
                else:
                    run_id = args.study
                result = run_prepared_study(
                    run_id, training_store, ExperimentStore(args.experiment_store)
                )
                print(json.dumps(result, indent=2))
                return 0 if result["status"] == "completed" else 1
            if args.train_command in {"diagnose-policy", "diagnose-reward", "competence"}:
                from sirfex.training.competence import run_competence
                from sirfex.training.remediation_diagnostics import policy_study, reward_study

                with open(args.config, encoding="utf-8") as stream:
                    value = json.load(stream)
                if args.train_command == "diagnose-policy":
                    result = policy_study(value, training_store, args.checkpoint)
                elif args.train_command == "diagnose-reward":
                    result = reward_study(
                        value, training_store, ExperimentStore(args.experiment_store)
                    )
                else:
                    result = run_competence(
                        value, training_store, allow_extended=args.allow_extended
                    )
                print(json.dumps(result, indent=2))
                return 0 if result.get("status", "completed") == "completed" else 1
            if args.train_command in {"feasibility", "diagnose"}:
                from sirfex.training.selection import discovery_report

                searches = SearchStore(args.search_store)
                sources = (
                    args.search
                    if args.search is not None
                    else [row["search_id"] for row in searches.list()]
                )
                report = discovery_report(searches, ExperimentStore(args.experiment_store), sources)
                if args.train_command == "diagnose":
                    from sirfex.training.evaluation import evaluate_controller
                    from sirfex.training.protocol import competence, datasets

                    reference = evaluate_controller(
                        PumpRuleController, datasets()["validation"], "validation", max_steps=200
                    ).to_dict()
                    report["rule_validation"] = reference
                    report["rule_competence"] = competence(reference)
                print(json.dumps(report, indent=2, sort_keys=True))
                return 0
            if args.train_command == "report":
                print(
                    json.dumps(
                        training_store.load(args.training_run_id, allow_partial=True),
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            if args.train_command in {"learning-curve", "validate"}:
                from sirfex.training.validation import run_validation

                with open(args.config, encoding="utf-8") as stream:
                    value = json.load(stream)
                result = run_validation(
                    value,
                    training_store,
                    SearchStore(args.search_store),
                    ExperimentStore(args.experiment_store),
                    search_ids=args.search,
                    continue_training=args.train_command == "validate",
                    allow_extended=args.allow_extended,
                )
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0 if result["status"] == "completed" else 1
            if args.train_command == "list":
                print(json.dumps(training_store.list(), indent=2, sort_keys=True))
                return 0
            if args.train_command in {"show", "evaluate", "compare", "verify"}:
                loaded = training_store.load(args.training_run_id)
                if args.train_command == "show":
                    print(json.dumps(loaded, indent=2, sort_keys=True))
                elif args.train_command == "evaluate":
                    print(json.dumps(loaded.get("evaluation", {}), indent=2, sort_keys=True))
                elif args.train_command == "compare":
                    print(json.dumps(loaded.get("regression", {}), indent=2, sort_keys=True))
                else:
                    checkpoints = loaded["manifest"].get("checkpoints", {})
                    if "checkpoint" in loaded["manifest"]:
                        checkpoints["baseline"] = loaded["manifest"]["checkpoint"]
                    verified = []
                    for name, metadata in checkpoints.items():
                        verify_checkpoint(metadata["path"], metadata["sha256"])
                        verified.append(
                            {"name": name, "status": "verified", "sha256": metadata["sha256"]}
                        )
                    print(
                        json.dumps(
                            {"training_run_id": args.training_run_id, "checkpoints": verified},
                            indent=2,
                            sort_keys=True,
                        )
                    )
                return 0
            with open(args.config, encoding="utf-8") as stream:
                config_value = json.load(stream)
            profile = str(config_value.pop("profile", "smoke"))
            seed = int(config_value.pop("training_seed", getattr(args, "seed", 0)))
            config = training_profile(profile, seed=seed, overrides=config_value)
            if args.train_command == "baseline":
                print(json.dumps(run_baseline(config, training_store), indent=2, sort_keys=True))
            elif args.train_command == "improve":
                print(
                    json.dumps(
                        run_improvement(
                            config,
                            training_store,
                            args.baseline,
                            args.search,
                            SearchStore(args.search_store),
                        ),
                        indent=2,
                        sort_keys=True,
                    )
                )
            return 0
        except (
            TrainingStorageError,
            SearchStorageError,
            ValueError,
            OSError,
            json.JSONDecodeError,
            ImportError,
        ) as exc:
            parser.error(str(exc))
    parser.error("unknown command")
    return 2
