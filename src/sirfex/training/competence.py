"""Sequential, bounded Phase 3.2 studies with the unchanged physical competence gate."""

import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sirfex.environments.pump_observations import EXTENDED
from sirfex.training.actions import DIRECT, policy_action_space
from sirfex.training.checkpoints import write_json
from sirfex.training.config import PPOConfig
from sirfex.training.controller import PPOController
from sirfex.training.evaluation import evaluate_controller
from sirfex.training.policy_diagnostics import diagnose_policy
from sirfex.training.ppo import dependency_versions, train_ppo
from sirfex.training.protocol import (
    GATE,
    competence,
    content_hash,
    datasets,
    leakage_check,
    scenario_record,
)
from sirfex.training.pump_gym import PumpGymEnv
from sirfex.training.rewards import LEGACY, TERMINATION, reward_spec
from sirfex.training.runner import _git_commit
from sirfex.training.validation import _checkpoint, validation_config


def competence_config(value, *, allow_extended=False):
    extras = {
        k: value[k]
        for k in ("policy_action_schema", "reward_schema", "termination_schema", "comparator")
        if k in value
    }
    base = {k: v for k, v in value.items() if k not in extras}
    config = validation_config({"schemas": [EXTENDED], **base}, allow_extended=allow_extended)
    if len(config["schemas"]) != 1:
        raise ValueError("Run one observation design per controlled competence study")
    config.update(
        {
            "policy_action_schema": DIRECT,
            "reward_schema": LEGACY,
            "termination_schema": TERMINATION,
            **extras,
        }
    )
    policy_action_space(config["policy_action_schema"])
    reward_spec(config["reward_schema"])
    if config["termination_schema"] != TERMINATION:
        raise ValueError("Unsupported termination contract")
    return config


def run_competence(value, store, *, allow_extended=False):
    config = competence_config(value, allow_extended=allow_extended)
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:10]}"
    directory = store.directory(run_id)
    directory.mkdir()
    marker = directory / ".incomplete"
    marker.touch()
    started, cpu_started = time.perf_counter(), time.process_time()
    deadline = started + config["wall_seconds"]
    manifest = {
        "schema_version": "2",
        "phase": "3.2",
        "kind": "competence",
        "training_run_id": run_id,
        "status": "running",
        "git_commit": _git_commit(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "checkpoints": {},
        "environment_version": "sirfex-pump:1",
        "dependencies": dependency_versions(),
        "resource_limits": {
            "wall_seconds": config["wall_seconds"],
            "parallel_workers": 1,
            "torch_threads": config["ppo"]["torch_threads"],
            "device": "cpu",
            "maximum_training_steps": config["budgets"][-1] * len(config["seeds"]),
        },
    }
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "config.json", config)
    model, env, points, selected = None, None, [], {}
    schema, active_seed = config["schemas"][0], config["seeds"][0]
    summary = {
        "training_run_id": run_id,
        "readiness": "blocked",
        "competent_seeds": [],
        "robustness_continuation_performed": False,
    }

    def remember(model_value):
        nonlocal model
        model = model_value

    try:
        partitions = datasets(config["dataset_seed"], config["dataset_count"])
        serialized = {k: [scenario_record(s) for s in v] for k, v in partitions.items()}
        hashes = {k: content_hash(v) for k, v in serialized.items()}
        write_json(
            directory / "dataset_partitions.json",
            {"groups": serialized, "hashes": hashes, "leakage_check": leakage_check(partitions)},
        )
        protocol = {
            "schema": "competence-study:1",
            "configuration": config,
            "gate": GATE,
            "partition_hashes": hashes,
            "selection": "first competent validation checkpoint per seed",
            "inference_mode": "deterministic",
            "reward": reward_spec(config["reward_schema"]),
            "final_holdouts": "reserved; never evaluated by this command",
        }
        write_json(directory / "protocol.json", protocol)
        manifest["protocol_sha256"] = content_hash(protocol)
        ppo = PPOConfig.from_dict(config["ppo"])
        for seed in config["seeds"]:
            active_seed, model = seed, None
            env = PumpGymEnv(
                partitions["training"],
                seed=seed,
                observation_schema=schema,
                policy_action_schema=config["policy_action_schema"],
                reward_schema=config["reward_schema"],
                termination_schema=config["termination_schema"],
            )
            for budget in config["budgets"]:
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Competence study wall deadline reached")
                before = model.num_timesteps if model is not None else 0
                segment = time.perf_counter()
                model = train_ppo(
                    env,
                    ppo,
                    seed=seed,
                    timesteps=budget - before,
                    model=model,
                    wall_seconds=deadline - time.perf_counter(),
                    on_model=remember,
                )
                elapsed = time.perf_counter() - segment
                meta = _checkpoint(
                    model,
                    directory,
                    run_id,
                    f"ppo_{seed}_{model.num_timesteps}",
                    config,
                    seed,
                    schema,
                )
                manifest["checkpoints"][meta.checkpoint_id] = meta.to_dict()
                if model.num_timesteps != budget:
                    raise TimeoutError("Partial rollout saved; requested budget not complete")
                evaluated = time.perf_counter()
                metrics = evaluate_controller(
                    lambda m=model, d=meta: PPOController(m, d),
                    partitions["validation"],
                    "nominal_validation",
                    max_steps=200,
                    observation_schema=schema,
                    reward_schema=config["reward_schema"],
                ).to_dict()
                gate = competence(metrics)
                point = {
                    "seed": seed,
                    "timesteps": budget,
                    "checkpoint": meta.to_dict(),
                    "validation": metrics,
                    "competence": gate,
                    "training_seconds": elapsed,
                    "steps_per_second": (budget - before) / elapsed,
                    "validation_seconds": time.perf_counter() - evaluated,
                    "sampled_episodes": dict(env.sample_counts),
                    "checkpoint_bytes": Path(meta.path).stat().st_size,
                }
                points.append(point)
                if gate["passed"] and str(seed) not in selected:
                    selected[str(seed)] = meta.to_dict()
                write_json(directory / "learning_curve.json", {"points": points})
                write_json(
                    directory / "baseline_selection.json", {"selected": selected, "gate": GATE}
                )
                write_json(directory / "manifest.json", manifest)
                print(
                    f"{run_id}: seed={seed} steps={budget} violations="
                    f"{metrics['constraint_violation_frequency']:.3f} "
                    f"MAE={metrics['mean_tracking_error']} competent={gate['passed']}",
                    file=sys.stderr,
                    flush=True,
                )
            # Read-only diagnostic loads its own model and forks the torch RNG.
            diagnostic = diagnose_policy(meta, partitions["validation"], samples=16)
            write_json(directory / f"policy_{seed}.json", diagnostic)
            env.close()
            env = None
        summary["competent_seeds"] = [int(s) for s in selected]
        summary["competence_replicated"] = len(selected) >= 3 and len(selected) == len(
            config["seeds"]
        )
        summary["readiness"] = "partially_ready" if selected else "blocked"
        summary["reason"] = (
            "Avoidable-failure evidence and readiness review required"
            if selected
            else "No competent nominal PPO checkpoint"
        )
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = (
            "interrupted" if isinstance(exc, (KeyboardInterrupt, TimeoutError)) else "failed"
        )
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if model is not None:
            meta = _checkpoint(model, directory, run_id, "partial", config, active_seed, schema)
            manifest["checkpoints"]["partial"] = meta.to_dict()
        if not isinstance(exc, (KeyboardInterrupt, TimeoutError)):
            raise
    finally:
        if env is not None:
            env.close()
        manifest["wall_seconds"] = time.perf_counter() - started
        manifest["cpu_seconds"] = time.process_time() - cpu_started
        try:
            import psutil

            memory = psutil.Process().memory_info()
            manifest["memory"] = {
                "rss_bytes": memory.rss,
                "peak_working_set_bytes": getattr(memory, "peak_wset", None),
            }
        except ImportError:
            manifest["memory"] = None
        summary["status"] = manifest["status"]
        write_json(directory / "summary.json", summary)
        write_json(directory / "manifest.json", manifest)
        if manifest["status"] == "completed":
            marker.unlink()
    return summary
