"""Bounded Phase 3.1 learning curves and gated, paired robustness studies."""

import json
import math
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sirfex import __version__
from sirfex.controllers import PumpRuleController
from sirfex.core import Scenario
from sirfex.environments.pump_observations import EXTENDED, MINIMAL
from sirfex.training.checkpoints import CheckpointMetadata, sha256_file, write_json
from sirfex.training.config import PPOConfig
from sirfex.training.controller import PPOController
from sirfex.training.diagnostics import REWARD_SCHEMA
from sirfex.training.evaluation import evaluate_controller
from sirfex.training.ppo import dependency_versions, train_ppo
from sirfex.training.protocol import (
    GATE,
    compare_paired,
    competence,
    content_hash,
    datasets,
    leakage_check,
    matched_budget,
    scenario_record,
)
from sirfex.training.pump_gym import PumpGymEnv
from sirfex.training.runner import _git_commit
from sirfex.training.selection import discovery_report

PROFILES = {
    "smoke": {
        "budgets": [128, 256],
        "seeds": [3],
        "wall_seconds": 90,
        "continuation_steps": 256,
        "dataset_count": 4,
    },
    "pilot": {
        "budgets": [10240, 25600],
        "seeds": [3],
        "wall_seconds": 900,
        "continuation_steps": 10240,
        "dataset_count": 12,
    },
    "standard": {
        "budgets": [10240, 25600, 51200],
        "seeds": [3, 17, 41],
        "wall_seconds": 3600,
        "continuation_steps": 25600,
        "dataset_count": 24,
    },
    "extended": {
        "budgets": [25600, 51200, 102400],
        "seeds": [3, 17, 41],
        "wall_seconds": 7200,
        "continuation_steps": 51200,
        "dataset_count": 32,
    },
}


def validation_config(value, *, allow_extended=False):
    allowed = {
        "profile",
        "budgets",
        "seeds",
        "wall_seconds",
        "continuation_steps",
        "dataset_count",
        "dataset_seed",
        "stress_ratio",
        "ppo",
        "schemas",
    }
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Unknown validation configuration field")
    profile = value.get("profile", "pilot")
    if profile not in PROFILES or (profile == "extended" and not allow_extended):
        raise ValueError("Unknown profile, or extended requires --allow-extended")
    config = {
        **PROFILES[profile],
        "profile": profile,
        "dataset_seed": 3101,
        "stress_ratio": 0.3,
        "schemas": [MINIMAL, EXTENDED],
        **value,
    }
    ppo = PPOConfig.from_dict(
        {"n_steps": 128, "batch_size": 32, "torch_threads": 2, **value.get("ppo", {})}
    )
    config["ppo"] = ppo.to_dict()
    for key in ("continuation_steps", "dataset_count", "dataset_seed"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    budgets = config["budgets"]
    if (
        not budgets
        or any(type(x) is not int or x <= 0 or x % ppo.n_steps for x in budgets)
        or sorted(set(budgets)) != budgets
        or config["continuation_steps"] % ppo.n_steps
    ):
        raise ValueError("Budgets must increase and be positive rollout multiples")
    if not config["seeds"] or any(type(x) is not int or x < 0 for x in config["seeds"]):
        raise ValueError("Independent training seeds must be nonnegative integers")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Independent training seeds must be unique")
    if not config["schemas"] or any(s not in (MINIMAL, EXTENDED) for s in config["schemas"]):
        raise ValueError("Unknown observation schema")
    if len(set(config["schemas"])) != len(config["schemas"]):
        raise ValueError("Duplicate observation schema")
    if not math.isfinite(config["wall_seconds"]) or config["wall_seconds"] <= 0:
        raise ValueError("Wall-clock budget must be finite and positive")
    if not 0 < config["stress_ratio"] <= 1:
        raise ValueError("Stress ratio must be in (0,1]")
    caps = PROFILES[profile]
    if (
        budgets[-1] > caps["budgets"][-1]
        or len(config["seeds"]) > len(caps["seeds"])
        or config["wall_seconds"] > caps["wall_seconds"]
        or config["continuation_steps"] > caps["continuation_steps"]
        or config["dataset_count"] > caps["dataset_count"]
    ):
        raise ValueError("Requested work exceeds the selected profile's resource caps")
    return config


def _checkpoint(model, directory, run_id, name, config, seed, schema):
    path = directory / "checkpoints" / (name + ".zip")
    path.parent.mkdir(exist_ok=True)
    model.save(str(path))
    meta = CheckpointMetadata(
        name,
        run_id,
        str(path.resolve()),
        sha256_file(path),
        "pump",
        (3,) if schema == MINIMAL else (6,),
        (1,),
        config["ppo"],
        seed,
        model.num_timesteps,
        dependency_versions(),
        schema,
        REWARD_SCHEMA,
    )
    write_json(path.with_suffix(".metadata.json"), meta.to_dict())
    return meta


def _evaluate(model, metadata, scenarios, name):
    started = time.perf_counter()
    summary = evaluate_controller(
        lambda: PPOController(model, metadata),
        scenarios,
        name,
        max_steps=max(x.parameters["horizon"] for x in scenarios),
        observation_schema=metadata.observation_schema,
    ).to_dict()
    summary["wall_seconds"] = time.perf_counter() - started
    return summary


def run_validation(
    value,
    store,
    search_store,
    experiment_store,
    *,
    search_ids=None,
    continue_training=False,
    allow_extended=False,
):
    config = validation_config(value, allow_extended=allow_extended)
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:10]}"
    directory = store.directory(run_id)
    directory.mkdir()
    marker = directory / ".incomplete"
    marker.touch()
    started = time.perf_counter()
    deadline = started + config["wall_seconds"]
    manifest = {
        "schema_version": "2",
        "training_run_id": run_id,
        "phase": "3.1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "sirfex_version": __version__,
        "git_commit": _git_commit(),
        "environment_version": "sirfex-pump:1",
        "dependencies": dependency_versions(),
        "python": sys.version,
        "platform": platform.platform(),
        "checkpoints": {},
        "resource_limits": {
            "wall_seconds": config["wall_seconds"],
            "maximum_training_steps": len(config["schemas"])
            * len(config["seeds"])
            * config["budgets"][-1]
            + 2 * len(config["seeds"]) * config["continuation_steps"],
            "torch_threads": config["ppo"]["torch_threads"],
            "parallel_workers": 1,
        },
    }
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "config.json", config)
    curves, selections, pairs = [], {}, []
    state = {"schema_version": "learning-curves:1", "points": curves}
    summary = {
        "training_run_id": run_id,
        "classification": "inconclusive",
        "continuation_performed": False,
        "competent_baselines": 0,
    }
    active_model, active_env = None, None

    def remember_model(model):
        nonlocal active_model
        active_model = model

    active_seed, active_schema = config["seeds"][0], config["schemas"][0]
    try:
        sources = (
            search_ids if search_ids is not None else [m["search_id"] for m in search_store.list()]
        )
        report = discovery_report(search_store, experiment_store, sources)
        write_json(directory / "feasibility_report.json", report)
        partitions = datasets(config["dataset_seed"], config["dataset_count"])
        stress = [Scenario.from_dict(row["scenario"]) for row in report["selected"]]
        if stress:
            partitions["curriculum"] = stress
        check = leakage_check(partitions)
        if not check["passed"]:
            raise ValueError(f"Partition leakage: {check['issues']}")
        dataset_json = {
            name: [scenario_record(s) for s in group] for name, group in partitions.items()
        }
        partition_hashes = {name: content_hash(group) for name, group in dataset_json.items()}
        write_json(
            directory / "dataset_partitions.json",
            {
                "schema_version": "partitions:1",
                "groups": dataset_json,
                "hashes": partition_hashes,
                "leakage_check": check,
            },
        )
        protocol = {
            "schema_version": "pump-validation:1",
            "configuration": config,
            "competence_gate": GATE,
            "dataset_hashes": partition_hashes,
            "reward_schema": REWARD_SCHEMA,
            "reward_violation_weight": 10,
            "selection": (
                "first competent checkpoint per schema/seed; schema chosen by validation MAE"
            ),
            "generalization": "unseen physical parameters; pump trajectories are deterministic",
            "nominal_tracking_degradation_tolerance": 0.10,
            "independent_seed_inference_minimum": 3,
        }
        manifest["protocol_sha256"] = content_hash(protocol)
        write_json(directory / "protocol.json", protocol)
        reference = evaluate_controller(
            PumpRuleController, partitions["validation"], "validation", max_steps=200
        ).to_dict()
        write_json(
            directory / "reward_diagnostics.json",
            {
                "schema_version": "reward-diagnostics:1",
                "reward_schema": REWARD_SCHEMA,
                "rule_validation": reference,
            },
        )
        ppo = PPOConfig.from_dict(config["ppo"])
        for schema in config["schemas"]:
            for seed in config["seeds"]:
                active_schema, active_seed = schema, seed
                active_env = PumpGymEnv(
                    partitions["training"], seed=seed, observation_schema=schema
                )
                active_model = None
                for budget in config["budgets"]:
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("Study wall-clock budget exhausted")
                    before = active_model.num_timesteps if active_model is not None else 0
                    train_start = time.perf_counter()
                    active_model = train_ppo(
                        active_env,
                        ppo,
                        seed=seed,
                        timesteps=budget - before,
                        model=active_model,
                        wall_seconds=deadline - time.perf_counter(),
                        on_model=remember_model,
                    )
                    elapsed = time.perf_counter() - train_start
                    metadata = _checkpoint(
                        active_model,
                        directory,
                        run_id,
                        f"{schema.split(':')[0]}_{seed}_{active_model.num_timesteps}",
                        config,
                        seed,
                        schema,
                    )
                    manifest["checkpoints"][metadata.checkpoint_id] = metadata.to_dict()
                    if active_model.num_timesteps != budget:
                        raise TimeoutError(
                            "Training interrupted by study deadline; checkpoint saved"
                        )
                    evaluation = _evaluate(
                        active_model, metadata, partitions["validation"], "validation"
                    )
                    gate = competence(evaluation)
                    point = {
                        "seed": seed,
                        "observation_schema": schema,
                        "timesteps": budget,
                        "checkpoint": metadata.to_dict(),
                        "training_seconds": elapsed,
                        "steps_per_second": (budget - before) / elapsed,
                        "validation": evaluation,
                        "competence": gate,
                        "sampled_episodes": dict(active_env.sample_counts),
                        "checkpoint_bytes": Path(metadata.path).stat().st_size,
                    }
                    curves.append(point)
                    key = f"{schema}/{seed}"
                    if gate["passed"] and key not in selections:
                        selections[key] = point
                    write_json(directory / "learning_curve.json", state)
                    write_json(directory / "manifest.json", manifest)
                    print(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "schema": schema,
                                "seed": seed,
                                "timesteps": budget,
                                "violations": evaluation["constraint_violation_frequency"],
                                "tracking": evaluation["mean_tracking_error"],
                                "competent": gate["passed"],
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                active_env.close()
                active_env = None
        summary["competent_baselines"] = len(selections)
        summary["eligible_discoveries"] = len(stress)
        summary["continuation_gates"] = {
            "feasible_discovery_available": bool(stress),
            "competent_baseline_for_each_seed": any(
                all(f"{schema}/{seed}" in selections for seed in config["seeds"])
                for schema in config["schemas"]
            ),
        }
        write_json(
            directory / "baseline_selection.json",
            {
                "schema_version": "selection:1",
                "criterion": GATE,
                "selected": {k: v["checkpoint"] for k, v in selections.items()},
            },
        )
        candidates = [
            s
            for s in config["schemas"]
            if all(f"{s}/{seed}" in selections for seed in config["seeds"])
        ]
        if not candidates:
            summary["stopping_reason"] = (
                "No observation schema passed competence for all training seeds"
            )
        elif not stress:
            summary["stopping_reason"] = "No replay-verified physically feasible discoveries"
        elif not continue_training:
            summary["stopping_reason"] = "Learning-curve command; continuation not requested"
        else:
            chosen = min(
                candidates,
                key=lambda s: sum(
                    selections[f"{s}/{seed}"]["validation"]["mean_tracking_error"]
                    for seed in config["seeds"]
                ),
            )
            summary["chosen_observation_schema"] = chosen
            evaluation_sets = {
                "nominal_holdout": partitions["nominal_holdout"],
                "held_out_stress": partitions["held_out_stress"],
                "known_stress": stress,
            }
            evaluation_hash = content_hash(
                {k: [s.to_dict() for s in v] for k, v in evaluation_sets.items()}
            )
            reference_final = {
                k: evaluate_controller(PumpRuleController, v, k, max_steps=200).to_dict()
                for k, v in evaluation_sets.items()
            }
            write_json(directory / "evaluation.json", {"rule_reference": reference_final})
            for seed in config["seeds"]:
                base = selections[f"{chosen}/{seed}"]["checkpoint"]
                pair, records = {"seed": seed}, {}
                for group, ratio in (("normal", 0.0), ("robust", config["stress_ratio"])):
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("Continuation deadline exhausted")
                    active_schema, active_seed = chosen, seed
                    active_env = PumpGymEnv(
                        partitions["training"],
                        seed=seed + 5000,
                        observation_schema=chosen,
                        stress_scenarios=stress,
                        stress_ratio=ratio,
                    )
                    active_model = train_ppo(
                        active_env,
                        ppo,
                        seed=seed + 5000,
                        timesteps=config["continuation_steps"],
                        checkpoint=base["path"],
                        expected_sha256=base["sha256"],
                        wall_seconds=deadline - time.perf_counter(),
                        on_model=remember_model,
                    )
                    metadata = _checkpoint(
                        active_model, directory, run_id, f"{group}_{seed}", config, seed, chosen
                    )
                    manifest["checkpoints"][metadata.checkpoint_id] = metadata.to_dict()
                    additional = active_model.num_timesteps - base["training_timesteps"]
                    records[group] = {
                        "starting_sha256": base["sha256"],
                        "observation_schema": chosen,
                        "reward_schema": REWARD_SCHEMA,
                        "ppo_config": config["ppo"],
                        "additional_timesteps": additional,
                        "continuation_seed": seed + 5000,
                        "evaluation_hash": evaluation_hash,
                        "stress_ratio": ratio,
                        "status": "completed"
                        if additional == config["continuation_steps"]
                        else "partial",
                        "sampled_episodes": dict(active_env.sample_counts),
                    }
                    if records[group]["status"] != "completed":
                        raise TimeoutError("Partial continuation cannot enter a matched comparison")
                    pair[group] = {
                        k: _evaluate(active_model, metadata, v, k)
                        for k, v in evaluation_sets.items()
                    }
                    active_env.close()
                    active_env = None
                pair["budget_verification"] = matched_budget(records["normal"], records["robust"])
                pair["training_records"] = records
                if not pair["budget_verification"]["matched"]:
                    raise ValueError("Unmatched continuation budgets")
                pairs.append(pair)
                write_json(directory / "continuation_results.json", {"pairs": pairs})
            summary["continuation_performed"] = True
        regression = compare_paired(pairs)
        summary["classification"] = regression["classification"]
        write_json(directory / "regression.json", regression)
        write_json(
            directory / "uncertainty.json",
            {
                "training_seed_count": len(config["seeds"]),
                "continuation_seed_pairs": len(pairs),
                "result": regression,
                "limitation": (
                    "Scenarios within one learned policy are not independent training seeds"
                ),
            },
        )
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = (
            "interrupted" if isinstance(exc, (KeyboardInterrupt, TimeoutError)) else "failed"
        )
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        summary["stopping_reason"] = str(exc)
        if active_model is not None:
            meta = _checkpoint(
                active_model, directory, run_id, "partial", config, active_seed, active_schema
            )
            manifest["checkpoints"]["partial"] = meta.to_dict()
        if not isinstance(exc, (KeyboardInterrupt, TimeoutError)):
            raise
    finally:
        if active_env is not None:
            active_env.close()
        manifest["wall_seconds"] = time.perf_counter() - started
        manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
        try:
            import psutil

            memory = psutil.Process().memory_info()
            manifest["memory"] = {
                "rss_bytes": memory.rss,
                "peak_working_set_bytes": getattr(memory, "peak_wset", None),
            }
        except ImportError:
            manifest["memory"] = None
        manifest["artifact_bytes"] = sum(
            p.stat().st_size for p in directory.rglob("*") if p.is_file()
        )
        summary["status"] = manifest["status"]
        write_json(directory / "learning_curve.json", state)
        write_json(directory / "summary.json", summary)
        write_json(directory / "manifest.json", manifest)
        if manifest["status"] == "completed":
            marker.unlink()
    return summary
