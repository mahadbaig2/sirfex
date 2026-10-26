"""Phase 3.3: frozen, paired continuation of the competent Phase 3.2 PPO policies."""

from __future__ import annotations

import hashlib
import math
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from sirfex import __version__
from sirfex.controllers import PumpRuleController
from sirfex.controllers.analytical_pump import AnalyticalPumpController
from sirfex.core import Scenario
from sirfex.experiments.replay import replay_experiment
from sirfex.experiments.service import run_and_save_experiment
from sirfex.search.avoidable import validate_avoidable_evidence
from sirfex.training.checkpoints import verify_checkpoint, write_json
from sirfex.training.config import PPOConfig
from sirfex.training.controller import PPOController
from sirfex.training.evaluation import evaluate_controller
from sirfex.training.feasibility import analyze_feasibility
from sirfex.training.ppo import dependency_versions, train_ppo
from sirfex.training.protocol import (
    content_hash,
    curriculum_compatibility,
    leakage_check,
    paired_interval,
    scenario_record,
)
from sirfex.training.pump_gym import PumpGymEnv
from sirfex.training.remediation_diagnostics import metadata_from_dict
from sirfex.training.runner import _git_commit
from sirfex.training.validation import _checkpoint

BASELINE_HASHES = {
    "3": "859c9244e83fce5cf31600d9f7fae0038e053fc8b3dc1663218ed616257e70f9",
    "17": "05b5fe45c1f46b7eb28699bd4a1dc693e9c11d56fba3376211c5985168a9d333",
    "41": "48fe432e373aef9f64e55e0092f9b09ba95ef407bcc6d81f76a1f61432fd83f0",
}
DISCOVERY_IDS = {
    "disc_05c97a1dec_0000",
    "disc_05c97a1dec_0001",
    "disc_05c97a1dec_0002",
}
ACTION = "pump-normalized:1"
REWARD = "pump-survival-tracking:2"
TERMINATION = "pump-terminate-on-violation:1"
OBSERVATION = "pump-calibrated:1"
GUARDRAILS = {
    "nominal_tracking_error_degradation": 0.10,
    "nominal_violation_frequency_degradation": 0.0,
    "held_out_violation_frequency_degradation": 0.0,
    "held_out_violation_magnitude_degradation": 0.0,
    "execution_failure_increase": 0,
}


def study_config(value):
    allowed = {
        "baseline_run",
        "search_id",
        "continuation_steps",
        "wall_seconds",
        "stress_ratio",
        "held_out_seed",
        "new_held_out_count",
        "holdout_attempt_cap",
    }
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Unknown matched study setting")
    config = {
        "baseline_run": "20261025T205339_64b622c146",
        "search_id": "20261026T090912_05c97a1dec",
        "continuation_steps": 25600,
        "wall_seconds": 2400,
        "stress_ratio": 0.3,
        "held_out_seed": 33103,
        "new_held_out_count": 48,
        "holdout_attempt_cap": 5000,
        **value,
    }
    if config["baseline_run"] != "20261025T205339_64b622c146":
        raise ValueError("The principal study must use the frozen Phase 3.2 study")
    if config["search_id"] != "20261026T090912_05c97a1dec":
        raise ValueError("The principal study must use the frozen Phase 3.2 search")
    if type(config["continuation_steps"]) is not int or config["continuation_steps"] != 25600:
        raise ValueError("The principal continuation budget is frozen at 25,600 steps per arm")
    if (
        type(config["wall_seconds"]) not in (int, float)
        or not math.isfinite(config["wall_seconds"])
        or not 0 < config["wall_seconds"] <= 3600
    ):
        raise ValueError("Wall deadline must be finite, positive and at most one hour")
    if config["stress_ratio"] != 0.3:
        raise ValueError("The principal robust curriculum is fixed at 30% stress")
    if (
        type(config["held_out_seed"]) is not int
        or config["held_out_seed"] != 33103
        or type(config["new_held_out_count"]) is not int
        or config["new_held_out_count"] != 48
        or type(config["holdout_attempt_cap"]) is not int
        or config["holdout_attempt_cap"] != 5000
    ):
        raise ValueError("The held-out generator specification is frozen")
    return config


def model_fingerprint(model):
    """Hash policy parameters and optimizer moments without relying on ZIP bytes."""
    import torch

    def normalized(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
            }
        if isinstance(value, dict):
            return {
                str(k): normalized(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))
            }
        if isinstance(value, (tuple, list)):
            return [normalized(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        raise TypeError(f"Unsupported optimizer state item: {type(value).__name__}")

    state = {
        "policy": normalized(model.policy.state_dict()),
        "optimizer": normalized(model.policy.optimizer.state_dict()),
        "num_timesteps": model.num_timesteps,
    }
    if not state["optimizer"]["state"]:
        raise ValueError("Baseline checkpoint does not contain restored PPO optimizer state")
    return {
        "sha256": content_hash(state),
        "num_timesteps": model.num_timesteps,
        "optimizer_state_entries": len(state["optimizer"]["state"]),
        "current_progress_remaining": model._current_progress_remaining,
    }


def _selected_baselines(source):
    if source["manifest"].get("phase") != "3.2" or source["manifest"].get("status") != "completed":
        raise ValueError("Phase 3.2 baseline study is not complete")
    selected = source["baseline_selection"]["selected"]
    if set(selected) != set(BASELINE_HASHES):
        raise ValueError("Three frozen seed selections are required")
    checks = {}
    for seed, value in selected.items():
        meta = metadata_from_dict(value)
        if (
            meta.sha256 != BASELINE_HASHES[seed]
            or meta.training_seed != int(seed)
            or meta.environment_id != "pump"
            or meta.observation_schema != OBSERVATION
            or meta.policy_action_schema != ACTION
            or meta.reward_schema != REWARD
            or meta.termination_schema != TERMINATION
            or tuple(meta.observation_shape) != (6,)
            or tuple(meta.action_shape) != (1,)
        ):
            raise ValueError(f"Frozen baseline contract mismatch for seed {seed}")
        verify_checkpoint(meta.path, meta.sha256)
        versions = dependency_versions()
        if any(versions.get(k) != v for k, v in meta.dependencies.items()):
            raise ValueError(f"Dependency versions differ from baseline seed {seed}")
        model = PPOController.load(meta.path, meta)._model
        if model.num_timesteps != meta.training_timesteps:
            raise ValueError("Checkpoint training-step metadata mismatch")
        checks[seed] = {
            "checkpoint": meta.to_dict(),
            "fingerprint": model_fingerprint(model),
            "optimizer_restored": True,
            "deterministic_inference": True,
        }
    return checks


def _known_failures(search, experiments):
    if (
        search["manifest"].get("status") != "completed"
        or search["manifest"].get("objective_id") != "pump.constraint_violation"
    ):
        raise ValueError("Frozen search is incomplete or uses the wrong objective")
    report = search.get("avoidable")
    if not report or report.get("status") != "completed":
        raise ValueError("Avoidability evidence is incomplete")
    discoveries = {d["discovery_id"]: d for d in search["discoveries"]}
    if set(discoveries) != DISCOVERY_IDS:
        raise ValueError("The frozen known-discovery set changed")
    result = {}
    for discovery_id in sorted(DISCOVERY_IDS):
        discovery = discoveries[discovery_id]
        rows = [r for r in report["records"] if r["discovery_id"] == discovery_id]
        if len(rows) != 2 or {r["reference_id"] for r in rows} != {"pump-rule", "pump-analytical"}:
            raise ValueError("Both reference witnesses are required")
        scenario = Scenario.from_dict(discovery["scenario"])
        if (
            scenario.environment_version != "sirfex-pump:1"
            or discovery.get("objective", {}).get("objective_id") != "pump.constraint_violation"
            or discovery.get("replay_status") not in ("exact_match", "tolerance_match")
            or not curriculum_compatibility(scenario)["compatible"]
            or analyze_feasibility(scenario)["finite_horizon_viable"] is not True
        ):
            raise ValueError("Known scenario is not a compatible feasible pump failure")
        witnesses = []
        for row in rows:
            if (
                not row.get("avoidable")
                or row.get("controller_config", {}).get("checkpoint", {}).get("sha256")
                != BASELINE_HASHES["3"]
                or row["failure_experiment_id"] not in discovery["experiment_ids"]
            ):
                raise ValueError("Discovery does not prove the frozen seed-3 failure")
            check = validate_avoidable_evidence(
                experiments, row["failure_experiment_id"], row["reference_experiment_id"]
            )
            if not check["avoidable"]:
                raise ValueError(f"Reference witness no longer verifies: {check.get('reason')}")
            witnesses.append(
                {
                    "controller": row["reference_id"],
                    "failure_experiment_id": row["failure_experiment_id"],
                    "reference_experiment_id": row["reference_experiment_id"],
                    "replay": check["replay"],
                    "horizon": check["witness"]["horizon"],
                }
            )
        result[discovery_id] = {**scenario_record(scenario), "witnesses": witnesses}
    return result


def _new_held_out(config, prior):
    rng = np.random.default_rng(config["held_out_seed"])
    accepted, attempts = [], 0
    while len(accepted) < config["new_held_out_count"] and attempts < config["holdout_attempt_cap"]:
        attempts += 1
        scenario = Scenario(
            "pump",
            {
                "dt": 0.1,
                "horizon": 200,
                "min_level": 0.05,
                "max_level": 0.95,
                "target_level": 0.5,
                "initial_level": float(rng.uniform(0.15, 0.85)),
                "demand": float(rng.uniform(0.35, 0.95)),
                "pump_capacity": float(rng.uniform(0.5, 1.4)),
                "efficiency": float(rng.uniform(0.6, 1.0)),
                "leak": float(rng.uniform(0, 0.12)),
            },
            seed=210000 + len(accepted),
            environment_version="sirfex-pump:1",
            provenance={"partition": "phase3.3_held_out", "generator": "pump-feasible-stress:1"},
        )
        if analyze_feasibility(scenario)["finite_horizon_viable"] is not True:
            continue
        if not leakage_check({"prior": prior + accepted, "candidate": [scenario]})["passed"]:
            continue
        accepted.append(scenario)
    if len(accepted) != config["new_held_out_count"]:
        raise ValueError("Could not freeze 48 independent feasible stress scenarios")
    return accepted, attempts


def _records(group):
    return [scenario_record(s) for s in group]


def _load_groups(partitions):
    if not partitions["leakage_check"]["passed"]:
        raise ValueError("Frozen partition leakage check failed")
    groups = {
        name: [Scenario.from_dict(row["scenario"]) for row in rows]
        for name, rows in partitions["groups"].items()
    }
    if any(
        content_hash(_records(group)) != partitions["hashes"][name]
        for name, group in groups.items()
    ):
        raise ValueError("Frozen partition hash mismatch")
    if not leakage_check(groups)["passed"]:
        raise ValueError("Current physical/seed partition leakage check failed")
    return groups


def _evaluate(model, meta, scenarios, name):
    summary = evaluate_controller(
        lambda: PPOController(model, meta),
        scenarios,
        name,
        max_steps=200,
        observation_schema=meta.observation_schema,
        reward_schema=meta.reward_schema,
    ).to_dict()
    rows = summary["episode_results"]
    summary["constraint_violation_count"] = sum(r.get("constraint_violation", False) for r in rows)
    summary["full_horizon_count"] = sum(r.get("full_horizon", False) for r in rows)
    summary["time_to_first_violation_seconds"] = [
        r.get("time_to_first_violation_seconds") for r in rows
    ]
    summary["checkpoint_sha256"] = meta.sha256
    return summary


def prepare_study(value, training, searches, experiments):
    """Freeze every study input and development observation before final training."""
    config = study_config(value)
    source = training.load(config["baseline_run"])
    baselines = _selected_baselines(source)
    search = searches.load(config["search_id"])
    known = _known_failures(search, experiments)
    source_partitions = source["dataset_partitions"]
    for name, rows in source_partitions["groups"].items():
        if content_hash(rows) != source_partitions["hashes"][name]:
            raise ValueError(f"Phase 3.2 source partition changed: {name}")
    original = {
        name: [Scenario.from_dict(row["scenario"]) for row in rows]
        for name, rows in source_partitions["groups"].items()
    }
    known_scenarios = [Scenario.from_dict(row["scenario"]) for row in known.values()]
    prior = [s for group in original.values() for s in group] + known_scenarios
    new_stress, attempts = _new_held_out(config, prior)
    groups = {
        **original,
        "known_stress": known_scenarios,
        "held_out_stress": original["held_out_stress"] + new_stress,
    }
    check = leakage_check(groups)
    if not check["passed"]:
        raise ValueError(f"Frozen partition overlap: {check['issues']}")
    serialized = {name: _records(group) for name, group in groups.items()}
    hashes = {name: content_hash(rows) for name, rows in serialized.items()}
    if not all(
        analyze_feasibility(s)["finite_horizon_viable"] is True for s in groups["held_out_stress"]
    ):
        raise ValueError("Held-out stress contains a nonviable scenario")
    protocol = {
        "schema_version": "matched-robustness:1",
        "hypothesis": "30% known stress lowers held-out feasible-stress violation frequency",
        "primary_outcome": "held_out_stress.constraint_violation_frequency",
        "paired_unit": "independent baseline training seed",
        "seeds": [3, 17, 41],
        "continuation_seed_policy": "baseline seed + 5000, identical for both arms",
        "continuation_steps_per_arm": config["continuation_steps"],
        "curriculum": {
            "normal_stress_ratio": 0.0,
            "robust_stress_ratio": 0.3,
            "stress_sampling": "uniform across 3 frozen discoveries",
        },
        "held_out_generator": {
            "seed": config["held_out_seed"],
            "new_scenarios": 48,
            "accepted_proposals": attempts,
            "bounds": {
                "initial_level": [0.15, 0.85],
                "demand": [0.35, 0.95],
                "pump_capacity": [0.5, 1.4],
                "efficiency": [0.6, 1.0],
                "leak": [0, 0.12],
            },
            "filter": "finite_horizon_viable and no physical/seed overlap",
        },
        "partition_hashes": hashes,
        "source_partition_hashes": source_partitions["hashes"],
        "known_discovery_ids": sorted(known),
        "baseline_hashes": BASELINE_HASHES,
        "contracts": {
            "observation": OBSERVATION,
            "action": ACTION,
            "reward": REWARD,
            "termination": TERMINATION,
        },
        "regression_guardrails": GUARDRAILS,
        "inference_mode": "deterministic",
        "final_holdout_access": "after six verified matched continuation checkpoints only",
        "uncertainty": "paired training-seed bootstrap, exploratory with n=3",
    }
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:10]}"
    directory = training.directory(run_id)
    directory.mkdir()
    (directory / ".incomplete").touch()
    manifest = {
        "schema_version": "2",
        "phase": "3.3",
        "kind": "matched-robustness",
        "training_run_id": run_id,
        "status": "preparing",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "sirfex_version": __version__,
        "environment_version": "sirfex-pump:1",
        "dependencies": dependency_versions(),
        "python": sys.version,
        "platform": platform.platform(),
        "checkpoints": {},
        "resource_limits": {
            "wall_seconds": config["wall_seconds"],
            "maximum_additional_steps": 153600,
            "parallel_workers": 1,
            "torch_threads": 2,
            "device": "cpu",
        },
        "protocol_sha256": content_hash(protocol),
    }
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "config.json", config)
    write_json(directory / "protocol.json", protocol)
    write_json(
        directory / "partitions.json",
        {
            "schema_version": "matched-partitions:1",
            "groups": serialized,
            "hashes": hashes,
            "source_hashes": source_partitions["hashes"],
            "leakage_check": check,
            "new_held_out_attempts": attempts,
        },
    )
    write_json(directory / "baseline_verification.json", baselines)
    write_json(
        directory / "curriculum.json",
        {
            "schema_version": "known-avoidable-curriculum:1",
            "known": known,
            "normal_ratio": 0.0,
            "robust_ratio": 0.3,
            "selection": "three fixed seed-3 discoveries, uniformly sampled",
            "known_scenarios_are_not_unseen_stress": True,
        },
    )
    precheck = {}
    for seed, row in baselines.items():
        meta = metadata_from_dict(row["checkpoint"])
        model = PPOController.load(meta.path, meta)._model
        precheck[seed] = _evaluate(model, meta, known_scenarios, "known_stress_precheck")
    write_json(directory / "known_stress_precheck.json", precheck)
    manifest["status"] = "prepared"
    write_json(directory / "manifest.json", manifest)
    return {
        "training_run_id": run_id,
        "status": "prepared",
        "protocol_sha256": manifest["protocol_sha256"],
        "partition_hashes": hashes,
        "known_failure_counts": {s: v["constraint_violation_count"] for s, v in precheck.items()},
    }


def _reference_summary(controller, scenarios, name, *, observation_schema):
    result = evaluate_controller(
        controller,
        scenarios,
        name,
        max_steps=200,
        observation_schema=observation_schema,
        reward_schema=REWARD,
    ).to_dict()
    result["constraint_violation_count"] = sum(
        r.get("constraint_violation", False) for r in result["episode_results"]
    )
    result["full_horizon_count"] = sum(
        r.get("full_horizon", False) for r in result["episode_results"]
    )
    return result


def _evaluate_final(
    run_id, directory, manifest, groups, baselines, training, experiments, deadline
):
    from sirfex.training.regression import classify_matched_robustness

    sets = {name: groups[name] for name in ("nominal_holdout", "known_stress", "held_out_stress")}
    evaluations = {
        "sets": {k: content_hash(_records(v)) for k, v in sets.items()},
        "seeds": {},
        "references": {},
    }
    replay_rows = []
    for seed in ("3", "17", "41"):
        evaluations["seeds"][seed] = {}
        for arm in ("baseline", "normal", "robust"):
            if time.perf_counter() >= deadline:
                raise TimeoutError("Study deadline reached during final evaluation")
            data = (
                baselines[seed]["checkpoint"]
                if arm == "baseline"
                else manifest["checkpoints"][f"seed_{seed}_{arm}"]
            )
            meta = metadata_from_dict(data)
            model = PPOController.load(meta.path, meta)._model
            evaluations["seeds"][seed][arm] = {}
            for name, scenarios in sets.items():
                started = time.perf_counter()
                summary = _evaluate(model, meta, scenarios, name)
                summary["evaluation_seconds"] = time.perf_counter() - started
                evaluations["seeds"][seed][arm][name] = summary
                write_json(directory / "evaluation.json", evaluations)
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Study deadline reached during final evaluation")
                record = run_and_save_experiment(
                    experiments,
                    scenarios[0],
                    "ppo",
                    max_steps=200,
                    controller_config={"checkpoint": meta.to_dict()},
                )
                replay = replay_experiment(experiments, record.experiment_id).to_dict()
                replay_rows.append(
                    {
                        "seed": int(seed),
                        "arm": arm,
                        "set": name,
                        "experiment_id": record.experiment_id,
                        "checkpoint_sha256": meta.sha256,
                        "replay": replay,
                    }
                )
                write_json(directory / "replay_verification.json", {"records": replay_rows})
    for controller_id, factory, schema in (
        ("pump-rule", PumpRuleController, "pump-minimal:1"),
        ("pump-analytical", AnalyticalPumpController, OBSERVATION),
    ):
        evaluations["references"][controller_id] = {}
        for name, scenarios in sets.items():
            if time.perf_counter() >= deadline:
                raise TimeoutError("Study deadline reached during reference evaluation")
            started = time.perf_counter()
            evaluations["references"][controller_id][name] = _reference_summary(
                factory, scenarios, name, observation_schema=schema
            )
            evaluations["references"][controller_id][name]["evaluation_seconds"] = (
                time.perf_counter() - started
            )
            write_json(directory / "evaluation.json", evaluations)
            record = run_and_save_experiment(
                experiments, scenarios[0], controller_id, max_steps=200
            )
            replay_rows.append(
                {
                    "controller": controller_id,
                    "set": name,
                    "experiment_id": record.experiment_id,
                    "replay": replay_experiment(experiments, record.experiment_id).to_dict(),
                }
            )
            write_json(directory / "replay_verification.json", {"records": replay_rows})
    if any(r["replay"]["status"] not in ("exact_match", "tolerance_match") for r in replay_rows):
        raise ValueError("Representative experiment replay failed")
    pairs = []
    for seed in ("3", "17", "41"):
        rows = evaluations["seeds"][seed]
        classified = classify_matched_robustness(rows["normal"], rows["robust"], GUARDRAILS)
        differences = {
            name: {
                metric: rows["robust"][name][metric] - rows["normal"][name][metric]
                for metric in (
                    "constraint_violation_frequency",
                    "maximum_violation_magnitude",
                    "mean_violation_magnitude",
                    "mean_tracking_error",
                )
            }
            for name in sets
        }
        pairs.append(
            {
                "seed": int(seed),
                "classification": classified,
                "differences_robust_minus_normal": differences,
            }
        )
    frequency = [
        p["differences_robust_minus_normal"]["held_out_stress"]["constraint_violation_frequency"]
        for p in pairs
    ]
    magnitude = [
        p["differences_robust_minus_normal"]["held_out_stress"]["maximum_violation_magnitude"]
        for p in pairs
    ]
    uncertainty = {
        "unit": "three independent baseline training-seed pairs",
        "held_out_frequency": paired_interval(frequency),
        "held_out_max_magnitude": paired_interval(magnitude),
        "frequency_median": statistics.median(frequency),
        "frequency_range": [min(frequency), max(frequency)],
        "frequency_directions": {
            "improved": sum(x < 0 for x in frequency),
            "unchanged": sum(x == 0 for x in frequency),
            "worsened": sum(x > 0 for x in frequency),
        },
        "warning": "Bootstrap with n=3 is exploratory; episodes are not independent policies",
    }
    incomplete = any(p["classification"].get("reason") for p in pairs)
    guardrails_pass = all(p["classification"]["guardrails_pass"] for p in pairs)
    if incomplete:
        overall = "inconclusive"
        reason = "At least one paired evaluation has incomplete execution evidence"
    elif not guardrails_pass:
        overall = "regressed"
        reason = "At least one matched seed violates a frozen guardrail"
    elif all(x < 0 for x in frequency) and uncertainty["held_out_frequency"]["interval_95"][1] < 0:
        overall = "improved"
        reason = "All three pairs reduce the primary held-out violation frequency"
    else:
        overall = "inconclusive"
        reason = "No consistent three-seed improvement on the primary held-out outcome"
    regression = {
        "classification": overall,
        "reason": reason,
        "primary_outcome": "held_out_stress.constraint_violation_frequency",
        "policy": GUARDRAILS,
        "pairs": pairs,
    }
    write_json(directory / "evaluation.json", evaluations)
    write_json(directory / "paired_comparison.json", {"pairs": pairs})
    write_json(directory / "uncertainty.json", uncertainty)
    write_json(directory / "regression.json", regression)
    return regression, evaluations, replay_rows


def run_prepared_study(run_id, training, experiments):
    """Train exactly six branches, then unseal final outcomes once all six verify."""
    saved = training.load(run_id, allow_partial=True)
    directory = training.directory(run_id)
    manifest, config, protocol = saved["manifest"], saved["config"], saved["protocol"]
    if manifest.get("phase") != "3.3" or manifest.get("status") != "prepared":
        raise ValueError("A prepared Phase 3.3 study is required")
    if content_hash(protocol) != manifest["protocol_sha256"]:
        raise ValueError("Frozen protocol hash mismatch")
    if saved["partitions"]["hashes"] != protocol["partition_hashes"]:
        raise ValueError("Partitions differ from the frozen protocol")
    if study_config(config) != config or protocol["regression_guardrails"] != GUARDRAILS:
        raise ValueError("Frozen study configuration or guardrails changed")
    groups = _load_groups(saved["partitions"])
    baselines = saved["baseline_verification"]
    if set(baselines) != set(BASELINE_HASHES):
        raise ValueError("Frozen baseline verification is incomplete")
    known = saved["curriculum"]["known"]
    if set(known) != DISCOVERY_IDS or [
        scenario_record(s)["scenario_id"] for s in groups["known_stress"]
    ] != [known[k]["scenario_id"] for k in sorted(known)]:
        raise ValueError("Known curriculum changed after protocol freeze")
    ppo = PPOConfig.from_dict(baselines["3"]["checkpoint"]["ppo_config"])
    if any(
        PPOConfig.from_dict(row["checkpoint"]["ppo_config"]) != ppo for row in baselines.values()
    ):
        raise ValueError("Baseline PPO hyperparameters differ")
    if config["continuation_steps"] % ppo.n_steps:
        raise ValueError("Continuation budget is not an exact rollout multiple")
    started = time.perf_counter()
    deadline = started + config["wall_seconds"]
    manifest["status"] = "training"
    write_json(directory / "manifest.json", manifest)
    records = []
    active_model, active_env = None, None
    active_seed, active_arm = None, None
    try:
        for seed in ("3", "17", "41"):
            base = metadata_from_dict(baselines[seed]["checkpoint"])
            if base.sha256 != BASELINE_HASHES[seed]:
                raise ValueError("Baseline identity differs from frozen selection")
            verify_checkpoint(base.path, base.sha256)
            for arm, ratio in (("normal", 0.0), ("robust", config["stress_ratio"])):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Matched study wall deadline reached")
                active_seed, active_arm, active_model = seed, arm, None
                active_env = PumpGymEnv(
                    groups["training"],
                    seed=int(seed) + 5000,
                    observation_schema=base.observation_schema,
                    stress_scenarios=groups["known_stress"],
                    stress_ratio=ratio,
                    policy_action_schema=base.policy_action_schema,
                    reward_schema=base.reward_schema,
                    termination_schema=base.termination_schema,
                )
                starting = {}

                def remember(
                    model, *, state=starting, expected=baselines[seed]["fingerprint"]["sha256"]
                ):
                    nonlocal active_model
                    active_model = model
                    state.update(model_fingerprint(model))
                    if state["sha256"] != expected:
                        raise ValueError("Continuation branch did not restore identical PPO state")

                segment = time.perf_counter()
                active_model = train_ppo(
                    active_env,
                    ppo,
                    seed=int(seed) + 5000,
                    timesteps=config["continuation_steps"],
                    checkpoint=base.path,
                    expected_sha256=base.sha256,
                    wall_seconds=deadline - time.perf_counter(),
                    on_model=remember,
                )
                elapsed = time.perf_counter() - segment
                metadata = _checkpoint(
                    active_model,
                    directory,
                    run_id,
                    f"seed_{seed}_{arm}",
                    {
                        "ppo": ppo.to_dict(),
                        "policy_action_schema": ACTION,
                        "reward_schema": REWARD,
                        "termination_schema": TERMINATION,
                    },
                    int(seed) + 5000,
                    OBSERVATION,
                )
                verify_checkpoint(metadata.path, metadata.sha256)
                manifest["checkpoints"][metadata.checkpoint_id] = metadata.to_dict()
                additional = active_model.num_timesteps - base.training_timesteps
                record = {
                    "seed": int(seed),
                    "arm": arm,
                    "starting_checkpoint": base.checkpoint_id,
                    "starting_sha256": base.sha256,
                    "starting_state": starting,
                    "continuation_seed": int(seed) + 5000,
                    "starting_timesteps": base.training_timesteps,
                    "final_timesteps": active_model.num_timesteps,
                    "additional_timesteps": additional,
                    "requested_additional_timesteps": config["continuation_steps"],
                    "ppo_config": ppo.to_dict(),
                    "contracts": protocol["contracts"],
                    "curriculum_ratio": ratio,
                    "sampled_episodes": dict(active_env.sample_counts),
                    "sampled_scenarios": dict(active_env.sample_counts_by_scenario_id),
                    "stress_exposures_by_discovery": {
                        discovery_id: active_env.sample_counts_by_scenario_id.get(
                            known[discovery_id]["scenario_id"], 0
                        )
                        for discovery_id in sorted(known)
                    },
                    "checkpoint": metadata.to_dict(),
                    "checkpoint_bytes": Path(metadata.path).stat().st_size,
                    "training_seconds": elapsed,
                    "steps_per_second": additional / elapsed,
                    "status": "completed"
                    if additional == config["continuation_steps"]
                    else "partial",
                }
                records.append(record)
                write_json(directory / "continuation_runs.json", {"records": records})
                write_json(directory / "manifest.json", manifest)
                print(
                    f"{run_id}: seed={seed} arm={arm} additional={additional} "
                    f"seconds={elapsed:.1f}",
                    file=sys.stderr,
                    flush=True,
                )
                active_env.close()
                active_env, active_model = None, None
                if record["status"] != "completed":
                    raise TimeoutError("Partial continuation cannot enter final evaluation")
            pair = records[-2:]
            if (
                pair[0]["starting_state"]["sha256"] != pair[1]["starting_state"]["sha256"]
                or pair[0]["additional_timesteps"] != pair[1]["additional_timesteps"]
                or pair[0]["continuation_seed"] != pair[1]["continuation_seed"]
                or pair[0]["ppo_config"] != pair[1]["ppo_config"]
                or pair[0]["contracts"] != pair[1]["contracts"]
            ):
                raise ValueError("Paired continuation variables or budgets differ")
        if len(records) != 6 or any(r["status"] != "completed" for r in records):
            raise ValueError("Six complete continuations are required")
        manifest["status"] = "evaluating"
        write_json(directory / "manifest.json", manifest)
        regression, evaluations, replays = _evaluate_final(
            run_id, directory, manifest, groups, baselines, training, experiments, deadline
        )
        summary = {
            "training_run_id": run_id,
            "status": "completed",
            "classification": regression["classification"],
            "reason": regression["reason"],
            "independent_training_seed_pairs": 3,
            "completed_continuations": 6,
            "actual_additional_training_steps": sum(r["additional_timesteps"] for r in records),
            "evaluation_scenarios_per_controller": {
                name: len(groups[name])
                for name in ("nominal_holdout", "known_stress", "held_out_stress")
            },
            "representative_replays": len(replays),
            "failed_replays": sum(
                r["replay"]["status"] not in ("exact_match", "tolerance_match") for r in replays
            ),
            "final_holdout_evaluated": bool(evaluations),
        }
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = (
            "interrupted" if isinstance(exc, (KeyboardInterrupt, TimeoutError)) else "failed"
        )
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        summary = {
            "training_run_id": run_id,
            "status": manifest["status"],
            "classification": "inconclusive",
            "reason": str(exc),
            "completed_continuations": sum(r["status"] == "completed" for r in records),
            "final_holdout_evaluated": (directory / "evaluation.json").exists(),
        }
        if active_model is not None and active_seed is not None:
            meta = _checkpoint(
                active_model,
                directory,
                run_id,
                f"partial_{active_seed}_{active_arm}",
                {
                    "ppo": ppo.to_dict(),
                    "policy_action_schema": ACTION,
                    "reward_schema": REWARD,
                    "termination_schema": TERMINATION,
                },
                int(active_seed) + 5000,
                OBSERVATION,
            )
            manifest["checkpoints"][meta.checkpoint_id] = meta.to_dict()
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
        write_json(directory / "summary.json", summary)
        write_json(directory / "manifest.json", manifest)
        if manifest["status"] == "completed":
            (directory / ".incomplete").unlink()
    return summary
