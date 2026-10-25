"""Measured action/reward studies using existing scenarios and experiment evidence."""

from datetime import UTC, datetime
from uuid import uuid4

import numpy as np

from sirfex.core import Scenario
from sirfex.experiments.replay import replay_experiment
from sirfex.experiments.service import run_and_save_experiment
from sirfex.training.checkpoints import CheckpointMetadata
from sirfex.training.policy_diagnostics import diagnose_policy
from sirfex.training.protocol import content_hash, scenario_record
from sirfex.training.rewards import LEGACY, SURVIVAL, components, reward_spec
from sirfex.training.runner import _git_commit


def metadata_from_dict(value):
    return CheckpointMetadata(
        **{
            **value,
            "observation_shape": tuple(value["observation_shape"]),
            "action_shape": tuple(value["action_shape"]),
        }
    )


def diagnostic_cases(value, store):
    source = store.load(value["source_run"])
    cases = [
        Scenario.from_dict(r["scenario"])
        for r in source["dataset_partitions"]["groups"]["validation"]
    ]
    if not 1 <= len(cases) <= 100:
        raise ValueError("Diagnostic validation batch must contain 1..100 scenarios")
    return source, cases


def save_diagnostic(store, kind, config, result):
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:10]}"
    store.save(
        run_id,
        {
            "manifest.json": {
                "schema_version": "2",
                "training_run_id": run_id,
                "phase": "3.2",
                "kind": kind,
                "status": "completed",
                "git_commit": _git_commit(),
                "created_at_utc": datetime.now(UTC).isoformat(),
            },
            "config.json": config,
            "diagnostics.json": result,
            "summary.json": {"training_run_id": run_id, "kind": kind},
        },
    )
    return {"training_run_id": run_id, "kind": kind}


def policy_study(value, store, checkpoint_id=None):
    source, cases = diagnostic_cases(value, store)
    checkpoints = source["manifest"].get("checkpoints", {})
    if checkpoint_id is not None:
        checkpoints = {checkpoint_id: checkpoints[checkpoint_id]}
    records = [
        diagnose_policy(
            metadata_from_dict(m),
            cases,
            seed=value.get("seed", 32),
            samples=value.get("samples", 32),
        )
        for m in checkpoints.values()
    ]
    if not records:
        raise ValueError("No source checkpoints")
    return save_diagnostic(store, "policy-diagnostics", value, {"records": records})


def episode_diagnostics(stored, gamma=0.99, *, include_steps=True):
    if not 0 < gamma <= 1:
        raise ValueError("Discount must be in (0,1]")
    p = stored.scenario.parameters
    target = float(p.get("target_level", 0.5))
    trajectory = stored.trajectory
    rewards = {}
    for schema in (LEGACY, SURVIVAL):
        rows = [
            components(
                schema,
                t["info"]["level"],
                target,
                t["action"][0],
                t["info"]["violation_magnitude"],
                t["terminated"],
            )
            for t in trajectory
        ]
        rewards[schema] = {
            "specification": reward_spec(schema),
            "episode_return": sum(r["total"] for r in rows),
            "discounted_return": sum(gamma**i * r["total"] for i, r in enumerate(rows)),
            "component_sums": {k: sum(r[k] for r in rows) for k in rows[0] if k != "reward_schema"},
        }
        if include_steps:
            rewards[schema]["per_step"] = rows
    actions = [t["action"][0] for t in trajectory]
    return {
        "experiment_id": stored.manifest.experiment_id,
        **scenario_record(stored.scenario),
        "controller_id": stored.manifest.controller_id,
        "inference_mode": "deterministic"
        if stored.manifest.evaluation_config["deterministic"]
        else "stochastic",
        "constraint_violation": any(t["info"]["constraint_violation"] for t in trajectory),
        "maximum_violation_magnitude": max(t["info"]["violation_magnitude"] for t in trajectory),
        "mean_tracking_error": float(
            np.mean([abs(t["info"]["level"] - target) for t in trajectory])
        ),
        "mean_applied_action": float(np.mean(actions)),
        "action_std": float(np.std(actions)),
        "lower_bound_fraction": float(np.mean(np.array(actions) == 0)),
        "upper_bound_fraction": float(np.mean(np.array(actions) == 1)),
        "episode_length": len(trajectory),
        "termination_reason": trajectory[-1]["info"]["termination_reason"],
        "rewards": rewards,
        "gamma": gamma,
    }


def reward_study(value, store, experiment_store):
    source, cases = diagnostic_cases(value, store)
    groups = [
        ("rule", "pump-rule", {}, True),
        ("zero", "pump-fixed", {"command": 0}, True),
        ("full", "pump-fixed", {"command": 1}, True),
        ("analytical", "pump-analytical", {}, True),
    ]
    for name, checkpoint in source["manifest"].get("checkpoints", {}).items():
        if checkpoint["training_timesteps"] == max(
            m["training_timesteps"] for m in source["manifest"]["checkpoints"].values()
        ):
            for deterministic in (True, False):
                groups.append(
                    (
                        f"{name}/{'deterministic' if deterministic else 'stochastic'}",
                        "ppo",
                        {"checkpoint": checkpoint, "inference_seed": value.get("seed", 32)},
                        deterministic,
                    )
                )
    results = {}
    for label, controller_id, config, deterministic in groups:
        episodes = []
        for scenario in cases:
            manifest = run_and_save_experiment(
                experiment_store,
                scenario,
                controller_id,
                controller_config=config,
                max_steps=scenario.parameters["horizon"],
                deterministic=deterministic,
            )
            episodes.append(
                episode_diagnostics(
                    experiment_store.load(manifest.experiment_id), gamma=value.get("gamma", 0.99)
                )
            )
        results[label] = {
            "episodes": episodes,
            "representative_replay": replay_experiment(
                experiment_store, episodes[0]["experiment_id"]
            ).to_dict(),
            "summary": {
                "violations": sum(e["constraint_violation"] for e in episodes),
                "scenarios": len(cases),
                "mean_tracking_error": float(np.mean([e["mean_tracking_error"] for e in episodes])),
                "mean_length": float(np.mean([e["episode_length"] for e in episodes])),
                "rewards": {
                    schema: {
                        key: float(np.mean([e["rewards"][schema][key] for e in episodes]))
                        for key in ("episode_return", "discounted_return")
                    }
                    for schema in (LEGACY, SURVIVAL)
                },
            },
        }
    return save_diagnostic(
        store,
        "reward-diagnostics",
        value,
        {
            "validation_sha256": content_hash([scenario_record(s) for s in cases]),
            "groups": results,
            "comparison": "Both reward formulas scored on identical recorded physical trajectories",
        },
    )
