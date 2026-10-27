"""Measured, development-only comparisons of compatible pump controllers."""

from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np

from sirfex import __version__
from sirfex.controllers import PumpRuleController
from sirfex.controllers.analytical_pump import AnalyticalPumpController
from sirfex.core import Scenario
from sirfex.environments.pump_observations import PumpObservationEnvironment
from sirfex.research.store import ResearchStore
from sirfex.training.controller import PPOController
from sirfex.training.matched_study import ACTION, OBSERVATION, REWARD, TERMINATION
from sirfex.training.persistence import TrainingStore
from sirfex.training.protocol import content_hash, scenario_record
from sirfex.training.remediation_diagnostics import metadata_from_dict
from sirfex.training.runner import _git_commit

SOURCE_32 = "20261025T205339_64b622c146"
SOURCE_33 = "20261026T222151_846c8296a2"
SOURCE_34 = "20261027T135821_f307f4551c"
SOURCE_35 = "20261027T143749_d88d2dd37f"
ARMS = ("normal", "expanded_5", "expanded_10")


def _check_contract(meta):
    if (
        meta.environment_id != "pump"
        or meta.observation_schema != OBSERVATION
        or meta.policy_action_schema != ACTION
        or meta.reward_schema != REWARD
        or meta.termination_schema != TERMINATION
        or tuple(meta.observation_shape) != (6,)
        or tuple(meta.action_shape) != (1,)
    ):
        raise ValueError("Incompatible diagnostic checkpoint contract")


def _scenarios(training):
    old32, old34, old35 = (
        training.load(SOURCE_32),
        training.load(SOURCE_34),
        training.load(SOURCE_35),
    )
    nominal_rows = old32["dataset_partitions"]["groups"]["validation"]
    stress_rows = old34["stress_benchmark"]["groups"]["screening_stress"]
    if (
        content_hash(nominal_rows) != old32["dataset_partitions"]["hashes"]["validation"]
        or content_hash(stress_rows) != old34["stress_benchmark"]["hashes"]["screening_stress"]
        or old35["manifest"]["status"] != "no_eligible_curriculum"
        or old35["protocol"]["development_stress_hash"]
        != old34["stress_benchmark"]["hashes"]["screening_stress"]
    ):
        raise ValueError("Development evidence identity changed")
    # Access only the two development partitions. No final scenario rows leave this function.
    return (
        [Scenario.from_dict(row["scenario"]) for row in nominal_rows],
        [Scenario.from_dict(row["scenario"]) for row in stress_rows],
        old32,
        old35,
    )


def _policy_distance(a, b):
    left = a.policy.state_dict()
    right = b.policy.state_dict()
    if left.keys() != right.keys():
        raise ValueError("PPO architectures differ")
    sq = count = 0
    for key in left:
        x, y = left[key].detach().cpu().numpy(), right[key].detach().cpu().numpy()
        if x.shape != y.shape:
            raise ValueError("PPO parameter shapes differ")
        sq += float(np.square(x.astype(np.float64) - y.astype(np.float64)).sum())
        count += x.size
    return {
        "parameter_l2": math.sqrt(sq),
        "parameter_count": count,
        "meaning": "descriptive weight distance; not a physical outcome",
    }


def _baseline_observations(scenarios, baseline, stride):
    records = []
    for scenario in scenarios:
        environment = PumpObservationEnvironment(OBSERVATION)
        baseline.reset()
        observation, _ = environment.reset(scenario)
        params = environment.validate_scenario(scenario)
        sid = scenario_record(scenario)["scenario_id"]
        for step in range(params["horizon"]):
            if step % stride == 0:
                level, target, demand, efficiency, capacity, leak = map(float, observation)
                records.append(
                    {
                        "scenario_id": sid,
                        "partition": scenario.provenance["partition"],
                        "step": step,
                        "observation": observation.tolist(),
                        "level": level,
                        "target_error": target - level,
                        "demand": demand,
                        "capacity": capacity,
                        "efficiency": efficiency,
                        "leak": leak,
                        "initial_level": params["initial_level"],
                        "distance_to_min": level - params["min_level"],
                        "distance_to_max": params["max_level"] - level,
                        "corrective_authority": capacity * efficiency - demand - leak * level,
                    }
                )
            action = baseline.act(observation)
            observation, _, done, truncated, _ = environment.step(action)
            if done or truncated:
                break
        environment.close()
    return records


def _bin(record):
    error = record["target_error"]
    boundary = min(record["distance_to_min"], record["distance_to_max"])
    return {
        "partition": record["partition"],
        "tracking_error": "below_target"
        if error > 0.05
        else "above_target"
        if error < -0.05
        else "near_target",
        "boundary_margin": "within_0.15" if boundary <= 0.15 else "beyond_0.15",
        "authority": "limited" if record["corrective_authority"] <= 0.05 else "available",
        "demand_ratio": "high"
        if record["demand"] / (record["capacity"] * record["efficiency"]) >= 0.85
        else "lower",
        "initial_level": "lower" if record["initial_level"] < 0.5 else "upper",
    }


def _action_comparison(records, controllers):
    actions = defaultdict(list)
    strata = defaultdict(lambda: defaultdict(list))
    for record in records:
        obs = np.asarray(record["observation"], dtype=np.float32)
        reference = AnalyticalPumpController()
        actions["analytical"].append(float(reference.act(obs)[0]))
        rule = PumpRuleController()
        rule.reset()
        actions["hysteresis_reset"].append(float(rule.act(obs)[0]))
        for name, controller in controllers.items():
            value = float(controller.act(obs)[0])
            actions[name].append(value)
            for dimension, label in _bin(record).items():
                strata[(name, dimension)][label].append(value)
    values = {name: np.asarray(seq, dtype=float) for name, seq in actions.items()}
    summaries = {
        name: {
            "count": len(seq),
            "mean_physical_command": float(seq.mean()),
            "std_physical_command": float(seq.std()),
            "lower_saturation": float(np.mean(seq <= 1e-6)),
            "upper_saturation": float(np.mean(seq >= 1 - 1e-6)),
        }
        for name, seq in values.items()
    }
    shifts = {}
    for reference_name, other_name in (
        ("baseline", "normal"),
        ("baseline", "expanded_5"),
        ("baseline", "expanded_10"),
        ("normal", "expanded_5"),
        ("normal", "expanded_10"),
    ):
        delta = values[other_name] - values[reference_name]
        shifts[f"{reference_name}_to_{other_name}"] = {
            "mean_absolute_physical_action_shift": float(np.mean(np.abs(delta))),
            "mean_signed_physical_action_shift": float(np.mean(delta)),
            "median_signed_physical_action_shift": float(np.median(delta)),
            "increased_command_fraction": float(np.mean(delta > 1e-6)),
            "decreased_command_fraction": float(np.mean(delta < -1e-6)),
        }
    conditioned = {}
    for (name, dimension), groups in strata.items():
        if name == "baseline":
            continue
        key = f"{name}:{dimension}"
        conditioned[key] = {
            label: {
                "count": len(seq),
                "mean_physical_command": statistics.mean(seq),
                "baseline_mean_physical_command": statistics.mean(
                    strata[("baseline", dimension)][label]
                ),
                "mean_signed_shift": statistics.mean(seq)
                - statistics.mean(strata[("baseline", dimension)][label]),
            }
            for label, seq in groups.items()
        }
    return {
        "schema_version": "sirfex-fixed-observation-actions:1",
        "observation_source": (
            "deterministic baseline rollouts on development scenarios, at configured stride"
        ),
        "reference_note": (
            "Hysteresis action is from a fresh reset at each fixed state; "
            "its memory is not matched to an independent trajectory."
        ),
        "summary": summaries,
        "shifts_from_baseline": shifts,
        "state_conditioned": conditioned,
    }


def diagnose(config: dict, training: TrainingStore, research: ResearchStore) -> dict:
    if not isinstance(config, dict) or set(config) - {"seeds", "observation_stride"}:
        raise ValueError("Unknown diagnostic configuration")
    seeds = config.get("seeds", [3, 17])
    stride = config.get("observation_stride", 5)
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(s) is not int or s not in (3, 17) for s in seeds)
        or len(set(seeds)) != len(seeds)
        or type(stride) is not int
        or not 1 <= stride <= 20
    ):
        raise ValueError("Diagnostic seeds must be 3/17; stride must be 1..20")
    started = time.perf_counter()
    nominal, stress, source32, source35 = _scenarios(training)
    source33 = training.load(SOURCE_33)
    if source33["manifest"].get("status") != "completed":
        raise ValueError("Phase 3.3 checkpoint source incomplete")
    rows = {(r["seed"], r["candidate_id"]): r for r in source35["screening_runs"]["records"]}
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:10]}"
    protocol = {
        "schema_version": "sirfex-policy-diagnostics:1",
        "seeds": seeds,
        "observation_stride": stride,
        "source_studies": [SOURCE_32, SOURCE_33, SOURCE_35],
        "development_nominal_hash": source35["protocol"]["development_nominal_hash"],
        "development_stress_hash": source35["protocol"]["development_stress_hash"],
        "final_sets_accessed": False,
    }
    manifest = {
        "schema_version": "sirfex-research:1",
        "run_id": run_id,
        "kind": "policy-regression-diagnostics",
        "status": "running",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "sirfex_version": __version__,
        "protocol_sha256": content_hash(protocol),
    }
    research.create(run_id, {"manifest.json": manifest, "protocol.json": protocol})
    all_pairs, action_rows, tracking_rows, distances, hypotheses = [], {}, {}, {}, []
    try:
        for seed in seeds:
            baseline_meta = metadata_from_dict(
                source32["baseline_selection"]["selected"][str(seed)]
            )
            metas = {"baseline": baseline_meta}
            for arm in ARMS:
                metas[arm] = metadata_from_dict(rows[seed, arm]["checkpoint"])
            for arm in ("normal", "robust"):
                metas[f"phase33_{arm}"] = metadata_from_dict(
                    source33["manifest"]["checkpoints"][f"seed_{seed}_{arm}"]
                )
            for meta in metas.values():
                _check_contract(meta)
            controllers = {
                name: PPOController.load(meta.path, meta) for name, meta in metas.items()
            }
            observations = _baseline_observations(nominal + stress, controllers["baseline"], stride)
            if not observations:
                raise ValueError("No development observations")
            action_rows[str(seed)] = _action_comparison(observations, controllers)
            distances[str(seed)] = {
                arm: _policy_distance(controllers["baseline"]._model, controllers[arm]._model)
                for arm in ARMS
            }
            metrics = {}
            for arm in ARMS:
                row = rows[seed, arm]
                metrics[arm] = {
                    name: {
                        key: row[name][key]
                        for key in (
                            "mean_tracking_error",
                            "constraint_violation_frequency",
                            "execution_failures",
                            "action_statistics",
                            "checkpoint_sha256",
                        )
                    }
                    for name in ("development_nominal", "development_stress")
                }
            tracking_rows[str(seed)] = metrics
            all_pairs.append(
                {
                    "seed": seed,
                    "observation_count": len(observations),
                    "source_scenario_ids": sorted({r["scenario_id"] for r in observations}),
                    "checkpoint_sha256": {name: meta.sha256 for name, meta in metas.items()},
                    "fixed_actions": action_rows[str(seed)],
                    "development_metrics": metrics,
                    "parameter_distance": distances[str(seed)],
                }
            )
            for arm in ("expanded_5", "expanded_10"):
                nom_n = metrics["normal"]["development_nominal"]["mean_tracking_error"]
                nom_r = metrics[arm]["development_nominal"]["mean_tracking_error"]
                stress_n = metrics["normal"]["development_stress"]["mean_tracking_error"]
                stress_r = metrics[arm]["development_stress"]["mean_tracking_error"]
                hypotheses.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "observed_nominal_mae_delta": nom_r - nom_n,
                        "observed_stress_mae_delta": stress_r - stress_n,
                        "fixed_observation_action_shift": action_rows[str(seed)][
                            "shifts_from_baseline"
                        ][f"baseline_to_{arm}"],
                        "plausible_explanation": (
                            "Policy action changes on shared development states may contribute to "
                            "trajectory differences; curriculum and optimizer randomness "
                            "are not isolated."
                        ),
                        "causal_status": "unconfirmed",
                    }
                )
            research.write(run_id, "controller_pairs.json", {"pairs": all_pairs})
        research.write(run_id, "action_differences.json", action_rows)
        research.write(run_id, "tracking_differences.json", tracking_rows)
        research.write(run_id, "policy_shift.json", distances)
        research.write(run_id, "hypotheses.json", {"records": hypotheses})
        research.write(
            run_id,
            "scenario_set.json",
            {
                "development_nominal": [scenario_record(s)["scenario_id"] for s in nominal],
                "development_stress": [scenario_record(s)["scenario_id"] for s in stress],
            },
        )
        summary = {
            "run_id": run_id,
            "status": "completed",
            "seeds": seeds,
            "observations_per_seed": {str(p["seed"]): p["observation_count"] for p in all_pairs},
            "runtime_seconds": time.perf_counter() - started,
            "final_holdouts_accessed": False,
        }
        research.write(run_id, "summary.json", summary)
        manifest["status"] = "completed"
        return summary
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        research.write(run_id, "manifest.json", manifest)
