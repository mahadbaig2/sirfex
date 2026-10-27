"""Phase 3.4: frozen stress benchmark and staged nominal-preserving study.

The final groups are generated without consulting candidate policies. Screening
uses separate development groups; final outcomes are opened only after selection.
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np

from sirfex import __version__
from sirfex.controllers import PumpRuleController
from sirfex.controllers.analytical_pump import AnalyticalPumpController
from sirfex.core import Scenario
from sirfex.experiments.replay import replay_experiment
from sirfex.experiments.service import run_and_save_experiment
from sirfex.search.avoidable import run_avoidable
from sirfex.training.checkpoints import verify_checkpoint, write_json
from sirfex.training.config import PPOConfig
from sirfex.training.controller import PPOController
from sirfex.training.evaluation import evaluate_controller
from sirfex.training.feasibility import analyze_feasibility
from sirfex.training.matched_study import (
    ACTION,
    OBSERVATION,
    REWARD,
    TERMINATION,
    _evaluate,
    _selected_baselines,
    model_fingerprint,
)
from sirfex.training.ppo import dependency_versions, train_ppo
from sirfex.training.protocol import content_hash, leakage_check, paired_interval, scenario_record
from sirfex.training.pump_gym import PumpGymEnv
from sirfex.training.remediation_diagnostics import metadata_from_dict
from sirfex.training.runner import _git_commit
from sirfex.training.validation import _checkpoint

PROFILES = {
    "smoke": {"screen_steps": 128, "confirm_steps": 128, "search_trials": 2, "wall_seconds": 300},
    "pilot": {
        "screen_steps": 2048,
        "confirm_steps": 5120,
        "search_trials": 16,
        "wall_seconds": 1200,
    },
    "standard": {
        "screen_steps": 5120,
        "confirm_steps": 25600,
        "search_trials": 40,
        "wall_seconds": 3600,
    },
    "extended": {
        "screen_steps": 10240,
        "confirm_steps": 51200,
        "search_trials": 80,
        "wall_seconds": 7200,
    },
}
STRATA = ("nominal_like", "moderate", "high", "near_boundary")
SOURCE_32 = "20261025T205339_64b622c146"
SOURCE_33 = "20261026T222151_846c8296a2"
OLD_SEARCH = "20261026T090912_05c97a1dec"
SCREEN_SEED = "3"
CONFIRM_SEEDS = ("3", "17", "41")


def configuration(value, *, allow_extended=False):
    if not isinstance(value, dict) or set(value) - {"profile", "benchmark_seed"}:
        raise ValueError("Unknown Phase 3.4 configuration field")
    profile = value.get("profile", "standard")
    if profile not in PROFILES or (profile == "extended" and not allow_extended):
        raise ValueError("Extended research requires explicit opt-in")
    seed = value.get("benchmark_seed", 3401)
    if type(seed) is not int or seed < 0:
        raise ValueError("Benchmark seed must be a nonnegative integer")
    return {
        "profile": profile,
        "benchmark_seed": seed,
        "source_phase_3_2": SOURCE_32,
        "source_phase_3_3": SOURCE_33,
        "screen_seed": int(SCREEN_SEED),
        "confirm_seeds": [int(x) for x in CONFIRM_SEEDS],
        "normal_degradation_tolerance": 0.10,
        "stress_ratios": [0.0, 0.05, 0.10],
        "historical_ratio": 0.30,
        "development_stress_per_stratum": 6,
        "final_stress_per_stratum": 16,
        "final_nominal_count": 12,
        "screen_stress_gain_minimum": 0.005,
        "selection_primary": "development stress violation frequency, then mean tracking error",
        "final_primary": "held-out feasible-stress violation frequency",
        "final_co_primary": "held-out feasible-stress mean absolute tracking error",
        "total_training_step_cap": 5 * PROFILES[profile]["screen_steps"]
        + 6 * PROFILES[profile]["confirm_steps"],
        **PROFILES[profile],
    }


def physical_difficulty(scenario):
    p = scenario.parameters
    effective = p["pump_capacity"] * p["efficiency"]
    required = p["demand"] + p["leak"] * p["target_level"]
    return {
        "target_authority_margin": effective - required,
        "target_authority_ratio": required / effective,
        "initial_boundary_margin": min(
            p["initial_level"] - p["min_level"], p["max_level"] - p["initial_level"]
        ),
        "initial_tracking_error": abs(p["initial_level"] - p["target_level"]),
        "corrective_inflow_margin_at_min": effective - p["demand"] - p["leak"] * p["min_level"],
        "leak": p["leak"],
        "demand_to_effective_capacity": p["demand"] / effective,
        "horizon": p["horizon"],
    }


def _candidate(rng, stratum, seed, partition):
    effective_capacity = float(rng.uniform(0.75, 1.30))
    efficiency = float(rng.uniform(0.70, 1.0))
    pump_capacity = effective_capacity / efficiency
    leak = float(rng.uniform(0, 0.10))
    ratio_ranges = {
        "nominal_like": (0.53, 0.70),
        "moderate": (0.72, 0.86),
        "high": (0.88, 0.99),
        "near_boundary": (0.94, 1.06),
    }
    ratio = float(rng.uniform(*ratio_ranges[stratum]))
    initial = (
        float(rng.uniform(0.30, 0.70))
        if stratum != "near_boundary"
        else float(rng.uniform(0.08, 0.20) if rng.random() < 0.5 else rng.uniform(0.80, 0.92))
    )
    return Scenario(
        "pump",
        {
            "dt": 0.1,
            "horizon": 200,
            "min_level": 0.05,
            "max_level": 0.95,
            "target_level": 0.5,
            "initial_level": initial,
            "demand": ratio * effective_capacity - leak * 0.5,
            "pump_capacity": pump_capacity,
            "efficiency": efficiency,
            "leak": leak,
        },
        seed=seed,
        environment_version="sirfex-pump:1",
        provenance={
            "partition": partition,
            "difficulty_stratum": stratum,
            "generator": "pump-phase34:1",
        },
    )


def generate_benchmark(config, prior):
    """Controller-independent stratified draw with exact finite-horizon filtering."""
    rng = np.random.default_rng(config["benchmark_seed"])
    groups, records, counts = {}, {}, {}
    designs = (
        ("screening_stress", config["development_stress_per_stratum"]),
        ("final_stress", config["final_stress_per_stratum"]),
        ("final_nominal", config["final_nominal_count"]),
    )
    seen = list(prior)
    serial = 0
    for name, count in designs:
        groups[name], records[name], counts[name] = [], [], {}
        names = STRATA if name != "final_nominal" else ("nominal_like",)
        for stratum in names:
            accepted = attempted = invalid = infeasible = duplicate = 0
            while accepted < count and attempted < count * 100:
                attempted += 1
                serial += 1
                scenario = _candidate(rng, stratum, 3400000 + serial, name)
                try:
                    analysis = analyze_feasibility(scenario)
                except ValueError:
                    invalid += 1
                    continue
                if analysis["finite_horizon_viable"] is not True:
                    infeasible += 1
                    continue
                if not leakage_check({"prior": seen, "candidate": [scenario]})["passed"]:
                    duplicate += 1
                    continue
                seen.append(scenario)
                groups[name].append(scenario)
                records[name].append(
                    {
                        **scenario_record(scenario),
                        "stratum": stratum,
                        "difficulty": physical_difficulty(scenario),
                        "feasibility": analysis,
                    }
                )
                accepted += 1
            if accepted != count:
                raise ValueError(f"Could not fill feasible {name}/{stratum} stratum")
            counts[name][stratum] = {
                "attempted": attempted,
                "accepted": accepted,
                "invalid": invalid,
                "infeasible": infeasible,
                "duplicate": duplicate,
            }
    return groups, {
        "groups": records,
        "counts": counts,
        "hashes": {k: content_hash(v) for k, v in records.items()},
        "population": (
            "fixed dt=.1,horizon=200,bounds=.05/.95,target=.5; "
            "target authority ratio and initial-boundary strata"
        ),
        "leakage_check": leakage_check({"historical": prior, **groups}),
    }


def historical_diagnosis(phase33):
    """Describe observed behavior and associations; do not infer a causal ratio effect."""
    evaluations = phase33["evaluation"]["seeds"]
    runs = {(r["seed"], r["arm"]): r for r in phase33["continuation_runs"]["records"]}
    result = {
        "source_study": SOURCE_33,
        "pairs": [],
        "interpretation": (
            "observational across three policy seeds; ratio and repeated "
            "discovery exposure are confounded"
        ),
    }
    for seed in CONFIRM_SEEDS:
        pair = {
            "seed": int(seed),
            "stress_episode_exposure": runs[int(seed), "robust"]["sampled_episodes"]["stress"],
            "stress_discovery_exposure": runs[int(seed), "robust"]["stress_exposures_by_discovery"],
            "sets": {},
        }
        for set_name in ("nominal_holdout", "known_stress", "held_out_stress"):
            normal = evaluations[seed]["normal"][set_name]
            robust = evaluations[seed]["robust"][set_name]
            rows = {}
            for arm, summary in (("normal", normal), ("robust", robust)):
                episodes = summary["episode_results"]
                demands = [x["scenario"]["parameters"]["demand"] for x in episodes]
                initials = [x["scenario"]["parameters"]["initial_level"] for x in episodes]
                med_demand, med_initial = statistics.median(demands), statistics.median(initials)
                rows[arm] = {
                    "mean_action": summary["action_statistics"]["mean"],
                    "action_std": summary["action_statistics"]["std"],
                    "lower_saturation": summary["action_statistics"]["lower_bound_fraction"],
                    "upper_saturation": summary["action_statistics"]["upper_bound_fraction"],
                    "tracking_mean": summary["mean_tracking_error"],
                    "tracking_range": [
                        min(x["mean_tracking_error"] for x in episodes),
                        max(x["mean_tracking_error"] for x in episodes),
                    ],
                    "tracking_by_demand": {
                        "lower_half": statistics.mean(
                            x["mean_tracking_error"]
                            for x in episodes
                            if x["scenario"]["parameters"]["demand"] <= med_demand
                        ),
                        "upper_half": statistics.mean(
                            x["mean_tracking_error"]
                            for x in episodes
                            if x["scenario"]["parameters"]["demand"] > med_demand
                        ),
                    },
                    "tracking_by_initial_level": {
                        "lower_half": statistics.mean(
                            x["mean_tracking_error"]
                            for x in episodes
                            if x["scenario"]["parameters"]["initial_level"] <= med_initial
                        ),
                        "upper_half": statistics.mean(
                            x["mean_tracking_error"]
                            for x in episodes
                            if x["scenario"]["parameters"]["initial_level"] > med_initial
                        ),
                    },
                    "violation_frequency": summary["constraint_violation_frequency"],
                    "mean_episode_length": summary["mean_episode_length"],
                }
            pair["sets"][set_name] = {
                "normal": rows["normal"],
                "robust": rows["robust"],
                "tracking_difference": robust["mean_tracking_error"]
                - normal["mean_tracking_error"],
            }
        result["pairs"].append(pair)
    return result


def _load_json(directory, name):
    return json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))


def _groups(benchmark):
    groups = {
        name: [Scenario.from_dict(row["scenario"]) for row in rows]
        for name, rows in benchmark["groups"].items()
    }
    for name, rows in benchmark["groups"].items():
        if content_hash(rows) != benchmark["hashes"][name]:
            raise ValueError("Frozen Phase 3.4 benchmark hash mismatch")
        for scenario, row in zip(groups[name], rows, strict=True):
            if scenario_record(scenario)["scenario_id"] != row["scenario_id"]:
                raise ValueError("Frozen scenario identity mismatch")
            if analyze_feasibility(scenario)["finite_horizon_viable"] is not True:
                raise ValueError("Frozen scenario lost finite-horizon viability")
    return groups


def _memory():
    try:
        import psutil

        info = psutil.Process().memory_info()
        return {"rss_bytes": info.rss, "peak_working_set_bytes": getattr(info, "peak_wset", None)}
    except ImportError:
        return None


def _search_configuration(trials):
    return {
        "environment_id": "pump",
        "environment_version": "sirfex-pump:1",
        "schema_version": "1",
        "objective": "pump.constraint_violation",
        "controller_id": "ppo",
        "baseline_seed": 3,
        "strategy": "random",
        "seed": 34,
        "scenario_seed": 340000,
        "trials": trials,
        "wall_seconds": 180,
        "description": "Phase 3.4 bounded development discovery, final benchmark excluded",
        "fixed_parameters": {
            "dt": 0.1,
            "horizon": 200,
            "target_level": 0.5,
            "min_level": 0.05,
            "max_level": 0.95,
        },
        "parameters": {
            "initial_level": {"type": "float", "low": 0.10, "high": 0.90},
            "demand": {"type": "float", "low": 0.35, "high": 1.05},
            "pump_capacity": {"type": "float", "low": 0.5, "high": 1.4},
            "efficiency": {"type": "float", "low": 0.6, "high": 1.0},
            "leak": {"type": "float", "low": 0.0, "high": 0.12},
        },
    }


def prepare(value, training, searches, experiments, *, allow_extended=False):
    """Freeze benchmark, discovery pool and screening rule before training."""
    started = time.perf_counter()
    config = configuration(value, allow_extended=allow_extended)
    source32, source33 = training.load(SOURCE_32), training.load(SOURCE_33)
    if source33["manifest"].get("status") != "completed":
        raise ValueError("Historical Phase 3.3 study is incomplete")
    baselines = _selected_baselines(source32)
    old_groups = {
        name: [Scenario.from_dict(row["scenario"]) for row in rows]
        for name, rows in source33["partitions"]["groups"].items()
    }
    prior = [scenario for group in old_groups.values() for scenario in group]
    benchmark_groups, benchmark = generate_benchmark(config, prior)
    if not benchmark["leakage_check"]["passed"]:
        raise ValueError("New benchmark overlaps historical physics or seeds")
    # Only designated development policies may be inspected before final freeze.
    dev_meta = metadata_from_dict(baselines[SCREEN_SEED]["checkpoint"])
    dev_model = PPOController.load(dev_meta.path, dev_meta)._model
    development_calibration = {
        "baseline_seed": int(SCREEN_SEED),
        "baseline": _evaluate(
            dev_model, dev_meta, benchmark_groups["screening_stress"], "screening_stress"
        ),
        "rule": evaluate_controller(
            PumpRuleController,
            benchmark_groups["screening_stress"],
            "screening_stress",
            max_steps=200,
        ).to_dict(),
        "analytical": evaluate_controller(
            AnalyticalPumpController,
            benchmark_groups["screening_stress"],
            "screening_stress",
            max_steps=200,
            observation_schema=OBSERVATION,
            reward_schema=REWARD,
        ).to_dict(),
    }
    search_spec = _search_configuration(config["search_trials"])
    discovery_run = run_avoidable(
        search_spec,
        experiments,
        searches,
        training,
        baseline=SOURCE_32,
        excluded_scenarios=prior + [s for group in benchmark_groups.values() for s in group],
    )
    found = searches.load(discovery_run["search_id"])
    old = source33["curriculum"]["known"]
    original = {
        key: {"scenario": row["scenario"], "source": OLD_SEARCH, "witnesses": row["witnesses"]}
        for key, row in sorted(old.items())
    }
    new = {}
    for row in found.get("avoidable", {}).get("records", []):
        if not row.get("avoidable"):
            continue
        discovery_id = row["discovery_id"]
        scenario = Scenario.from_dict(row["scenario"])
        if not leakage_check(
            {
                "old_and_final": prior + [s for g in benchmark_groups.values() for s in g],
                "new": [scenario],
            }
        )["passed"]:
            raise ValueError("Search discovery leaked into final or historical partitions")
        new.setdefault(
            discovery_id,
            {
                "scenario": row["scenario"],
                "source": discovery_run["search_id"],
                "source_controller_sha256": dev_meta.sha256,
                "failure_experiment_id": row["failure_experiment_id"],
                "reference_experiment_id": row["reference_experiment_id"],
                "replay": row["replay"],
                "feasibility": row["feasibility"],
            },
        )
    new_scenarios = [Scenario.from_dict(r["scenario"]) for r in new.values()]
    if not leakage_check(
        {
            "original": [Scenario.from_dict(r["scenario"]) for r in original.values()],
            "new": new_scenarios,
        }
    )["passed"]:
        raise ValueError("Expanded discovery pool duplicates historical failures")
    candidates = [{"id": "normal", "stress_ratio": 0.0, "pool": "none"}]
    for ratio in (0.05, 0.10):
        candidates.append(
            {"id": f"original_{int(ratio * 100)}", "stress_ratio": ratio, "pool": "original"}
        )
    if len(new) >= 2:
        for ratio in (0.05, 0.10):
            candidates.append(
                {"id": f"expanded_{int(ratio * 100)}", "stress_ratio": ratio, "pool": "expanded"}
            )
    screening = {
        "candidates": candidates,
        "policy": {
            "nominal_relative_mae_max": config["normal_degradation_tolerance"],
            "nominal_violation_increase_max": 0,
            "execution_failure_increase_max": 0,
            "development_stress_mae_gain_min": config["screen_stress_gain_minimum"],
            "stress_violation_increase_max": 0,
            "selection": (
                "eligible candidate with lowest stress violation frequency, "
                "then lowest MAE; stable ID tie break"
            ),
            "no_eligible": "stop without final training or holdout evaluation",
        },
        "development_only": ["phase3.2_validation", "screening_stress"],
        "historical_30_percent": "Phase 3.3 measured comparator; no new 30% arm",
    }
    protocol = {
        "schema_version": "phase-3.4-protocol:1",
        "config": config,
        "source_checkpoint_hashes": {
            seed: baselines[seed]["checkpoint"]["sha256"] for seed in CONFIRM_SEEDS
        },
        "benchmark_hashes": benchmark["hashes"],
        "development_validation_hash": source32["dataset_partitions"]["hashes"]["validation"],
        "training_hash": source32["dataset_partitions"]["hashes"]["training"],
        "discovery_search_id": discovery_run["search_id"],
        "discovery_pool_hash": content_hash({"original": original, "new": new}),
        "screening": screening,
        "contracts": {
            "observation": OBSERVATION,
            "action": ACTION,
            "reward": REWARD,
            "termination": TERMINATION,
        },
        "final_analysis": {
            "primary": config["final_primary"],
            "co_primary": config["final_co_primary"],
            "frequency_floor_fallback": (
                "use co-primary tracking if both arms have zero violation frequency"
            ),
            "nominal_tracking_relative_max": 0.10,
            "nominal_violation_increase_max": 0,
            "execution_failure_increase_max": 0,
            "independent_unit": "baseline policy training seed",
            "paired_bootstrap": "5000 draws, seed 3101; exploratory at n=3",
            "improved": (
                "all seeds pass nominal guards and improve at least one "
                "held-out co-primary with none worse"
            ),
            "regressed": "any completed nominal/safety guardrail fails",
            "otherwise": "inconclusive",
        },
    }
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:10]
    manifest = {
        "schema_version": "2",
        "phase": "3.4",
        "kind": "discriminating-stress-study",
        "training_run_id": run_id,
        "status": "prepared",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "sirfex_version": __version__,
        "environment_version": "sirfex-pump:1",
        "dependencies": dependency_versions(),
        "python": sys.version,
        "platform": platform.platform(),
        "checkpoints": {},
        "protocol_sha256": content_hash(protocol),
        "preparation_seconds": time.perf_counter() - started,
    }
    training.save(
        run_id,
        {
            "manifest.json": manifest,
            "config.json": config,
            "protocol.json": protocol,
            "diagnostics.json": historical_diagnosis(source33),
            "stress_benchmark.json": benchmark,
            "difficulty_strata.json": {"strata": STRATA, "counts": benchmark["counts"]},
            "candidate_curricula.json": {
                "screening": screening,
                "original": original,
                "new": new,
                "discovery_search": discovery_run,
            },
            "baseline_verification.json": baselines,
            "development_results.json": {"calibration": development_calibration},
        },
    )
    return {
        "study_id": run_id,
        "status": "prepared",
        "protocol_sha256": content_hash(protocol),
        "benchmark_hashes": benchmark["hashes"],
        "new_avoidable_discoveries": len(new),
        "screening_candidates": [c["id"] for c in candidates],
        "development_baseline_violations": development_calibration["baseline"][
            "constraint_violation_count"
        ],
    }


def _validated_study(run_id, training, statuses):
    saved = training.load(run_id, allow_partial=True)
    manifest = saved["manifest"]
    if manifest.get("phase") != "3.4" or manifest.get("status") not in statuses:
        raise ValueError(f"Phase 3.4 study must be in {sorted(statuses)}")
    if content_hash(saved["protocol"]) != manifest["protocol_sha256"]:
        raise ValueError("Frozen Phase 3.4 protocol hash changed")
    if saved["protocol"]["benchmark_hashes"] != saved["stress_benchmark"]["hashes"]:
        raise ValueError("Frozen benchmark identity changed")
    return saved, _groups(saved["stress_benchmark"])


def _pools(candidate_data):
    original = [Scenario.from_dict(v["scenario"]) for v in candidate_data["original"].values()]
    expanded = original + [
        Scenario.from_dict(v["scenario"]) for v in candidate_data["new"].values()
    ]
    return {"none": [], "original": original, "expanded": expanded}


def select_candidate(rows, policy):
    """Frozen development-only rule; never accepts a final holdout result."""
    if not rows or rows[0]["candidate_id"] != "normal":
        raise ValueError("Normal development control must be first")
    normal = rows[0]
    eligible = []
    decisions = []
    for row in rows[1:]:
        nominal_n = normal["development_nominal"]
        nominal_r = row["development_nominal"]
        stress_n = normal["development_stress"]
        stress_r = row["development_stress"]
        checks = {
            "nominal_tracking": nominal_r["mean_tracking_error"]
            <= nominal_n["mean_tracking_error"] * (1 + policy["nominal_relative_mae_max"]),
            "nominal_violation": nominal_r["constraint_violation_frequency"]
            <= nominal_n["constraint_violation_frequency"]
            + policy["nominal_violation_increase_max"],
            "stress_violation": stress_r["constraint_violation_frequency"]
            <= stress_n["constraint_violation_frequency"] + policy["stress_violation_increase_max"],
            "execution": all(
                row[key]["execution_failures"]
                <= normal[key]["execution_failures"] + policy["execution_failure_increase_max"]
                for key in ("development_nominal", "development_stress")
            ),
            "stress_gain": (
                stress_r["constraint_violation_frequency"]
                < stress_n["constraint_violation_frequency"]
                or stress_r["mean_tracking_error"]
                <= stress_n["mean_tracking_error"] - policy["development_stress_mae_gain_min"]
            ),
        }
        if any(row[key]["episodes"] == 0 for key in ("development_nominal", "development_stress")):
            checks["execution"] = False
        decision = {
            "candidate_id": row["candidate_id"],
            "checks": checks,
            "eligible": all(checks.values()),
        }
        decisions.append(decision)
        if decision["eligible"]:
            eligible.append(row)
    winner = min(
        eligible,
        key=lambda r: (
            r["development_stress"]["constraint_violation_frequency"],
            r["development_stress"]["mean_tracking_error"],
            r["candidate_id"],
        ),
        default=None,
    )
    return {
        "status": "selected" if winner else "no_eligible_curriculum",
        "selected_candidate_id": winner["candidate_id"] if winner else None,
        "decisions": decisions,
        "rule": policy,
    }


def screen(run_id, training):
    """Run bounded CPU screening using development validation only."""
    saved, groups = _validated_study(run_id, training, {"prepared"})
    manifest, config = saved["manifest"], saved["config"]
    directory = training.directory(run_id)
    baseline = metadata_from_dict(saved["baseline_verification"][SCREEN_SEED]["checkpoint"])
    verify_checkpoint(baseline.path, baseline.sha256)
    source32 = training.load(SOURCE_32)
    nominal = [
        Scenario.from_dict(r["scenario"])
        for r in source32["dataset_partitions"]["groups"]["validation"]
    ]
    training_pool = [
        Scenario.from_dict(r["scenario"])
        for r in source32["dataset_partitions"]["groups"]["training"]
    ]
    if (
        content_hash([scenario_record(s) for s in nominal])
        != saved["protocol"]["development_validation_hash"]
    ):
        raise ValueError("Development validation hash changed")
    if (
        content_hash([scenario_record(s) for s in training_pool])
        != saved["protocol"]["training_hash"]
    ):
        raise ValueError("Training partition hash changed")
    candidates = saved["candidate_curricula"]["screening"]["candidates"]
    pools = _pools(saved["candidate_curricula"])
    ppo = PPOConfig.from_dict(baseline.ppo_config)
    started = time.perf_counter()
    deadline = started + max(1, config["wall_seconds"] - manifest["preparation_seconds"])
    manifest["status"] = "screening"
    write_json(directory / "manifest.json", manifest)
    rows = []
    try:
        for candidate in candidates:
            if time.perf_counter() >= deadline:
                raise TimeoutError("Phase 3.4 screening deadline reached")
            cid = candidate["id"]
            env = PumpGymEnv(
                training_pool,
                seed=5003,
                observation_schema=OBSERVATION,
                stress_scenarios=pools[candidate["pool"]],
                stress_ratio=candidate["stress_ratio"],
                policy_action_schema=ACTION,
                reward_schema=REWARD,
                termination_schema=TERMINATION,
            )
            fingerprint = {}

            def check_start(model, target=fingerprint):
                target.update(model_fingerprint(model))
                if (
                    target["sha256"]
                    != saved["baseline_verification"][SCREEN_SEED]["fingerprint"]["sha256"]
                ):
                    raise ValueError("Screening candidate changed starting PPO state")

            segment = time.perf_counter()
            model = train_ppo(
                env,
                ppo,
                seed=5003,
                timesteps=config["screen_steps"],
                checkpoint=baseline.path,
                expected_sha256=baseline.sha256,
                wall_seconds=deadline - time.perf_counter(),
                on_model=check_start,
            )
            elapsed = time.perf_counter() - segment
            meta = _checkpoint(
                model,
                directory,
                run_id,
                f"screen_{cid}",
                {
                    "ppo": ppo.to_dict(),
                    "policy_action_schema": ACTION,
                    "reward_schema": REWARD,
                    "termination_schema": TERMINATION,
                },
                5003,
                OBSERVATION,
            )
            verify_checkpoint(meta.path, meta.sha256)
            manifest["checkpoints"][meta.checkpoint_id] = meta.to_dict()
            actual = model.num_timesteps - baseline.training_timesteps
            row = {
                "candidate_id": cid,
                "pool": candidate["pool"],
                "stress_ratio": candidate["stress_ratio"],
                "starting_sha256": baseline.sha256,
                "starting_fingerprint": fingerprint,
                "checkpoint": meta.to_dict(),
                "requested_additional_steps": config["screen_steps"],
                "actual_additional_steps": actual,
                "training_seconds": elapsed,
                "sampled_episodes": dict(env.sample_counts),
                "sampled_scenarios": dict(env.sample_counts_by_scenario_id),
                "development_nominal": _evaluate(model, meta, nominal, "development_nominal"),
                "development_stress": _evaluate(
                    model, meta, groups["screening_stress"], "development_stress"
                ),
            }
            rows.append(row)
            write_json(
                directory / "development_results.json",
                {
                    "calibration": saved["development_results"]["calibration"],
                    "screening_runs": rows,
                },
            )
            write_json(directory / "manifest.json", manifest)
            env.close()
            if actual != config["screen_steps"]:
                raise TimeoutError("Incomplete screening candidate cannot be selected")
            print(
                f"{run_id}: screened {cid} steps={actual} seconds={elapsed:.1f}",
                file=sys.stderr,
                flush=True,
            )
        selection = select_candidate(rows, saved["candidate_curricula"]["screening"]["policy"])
        manifest["status"] = selection["status"]
        selected = next(
            (c for c in candidates if c["id"] == selection["selected_candidate_id"]), None
        )
        selection["selected_configuration"] = selected
        write_json(directory / "selected_curriculum.json", selection)
        if selected:
            frozen = {
                "schema_version": "phase-3.4-final:1",
                "preregistered_protocol_sha256": manifest["protocol_sha256"],
                "selected_curriculum": selected,
                "selected_pool_scenario_ids": [
                    scenario_record(s)["scenario_id"] for s in pools[selected["pool"]]
                ],
                "final_benchmark_hashes": {
                    k: saved["stress_benchmark"]["hashes"][k]
                    for k in ("final_nominal", "final_stress")
                },
                "decision": selection["decisions"],
                "analysis": saved["protocol"]["final_analysis"],
            }
            manifest["final_protocol_sha256"] = content_hash(frozen)
            write_json(directory / "final_protocol.json", frozen)
    except BaseException as exc:
        manifest["status"] = (
            "interrupted" if isinstance(exc, (KeyboardInterrupt, TimeoutError)) else "failed"
        )
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if not isinstance(exc, (KeyboardInterrupt, TimeoutError)):
            raise
    finally:
        manifest["screening_seconds"] = time.perf_counter() - started
        manifest["memory"] = _memory()
        write_json(directory / "manifest.json", manifest)
    result = {
        "study_id": run_id,
        "status": manifest["status"],
        "selection": _load_json(directory, "selected_curriculum")
        if (directory / "selected_curriculum.json").exists()
        else None,
        "actual_screening_steps": sum(r["actual_additional_steps"] for r in rows),
    }
    write_json(directory / "summary.json", result)
    return result


def classify_final(evaluation, strata, policy):
    """Use the frozen co-primary rule; seeds, never episodes, are replicates."""
    pairs = []
    for seed in CONFIRM_SEEDS:
        n, r = evaluation["seeds"][seed]["normal"], evaluation["seeds"][seed]["selected"]
        checks = {
            "nominal_tracking": r["final_nominal"]["mean_tracking_error"]
            <= n["final_nominal"]["mean_tracking_error"]
            * (1 + policy["nominal_tracking_relative_max"]),
            "nominal_violations": r["final_nominal"]["constraint_violation_frequency"]
            <= n["final_nominal"]["constraint_violation_frequency"]
            + policy["nominal_violation_increase_max"],
            "stress_violations": r["final_stress"]["constraint_violation_frequency"]
            <= n["final_stress"]["constraint_violation_frequency"],
            "execution": all(
                r[name]["execution_failures"]
                <= n[name]["execution_failures"] + policy["execution_failure_increase_max"]
                for name in ("final_nominal", "final_stress")
            ),
        }
        incomplete = any(x["episodes"] != x["scenarios"] for arm in (n, r) for x in arm.values())
        normal_rows, robust_rows = (
            n["final_stress"]["episode_results"],
            r["final_stress"]["episode_results"],
        )
        sensitivity = {}
        for stratum in STRATA:
            indices = [i for i, item in enumerate(strata) if item == stratum]
            sensitivity[stratum] = {
                "normal_violations": sum(normal_rows[i]["constraint_violation"] for i in indices),
                "selected_violations": sum(robust_rows[i]["constraint_violation"] for i in indices),
                "normal_tracking": statistics.mean(
                    normal_rows[i]["mean_tracking_error"] for i in indices
                ),
                "selected_tracking": statistics.mean(
                    robust_rows[i]["mean_tracking_error"] for i in indices
                ),
                "count": len(indices),
            }
        pairs.append(
            {
                "seed": int(seed),
                "checks": checks,
                "incomplete": incomplete,
                "classification": "inconclusive"
                if incomplete
                else "regressed"
                if not all(checks.values())
                else "inconclusive",
                "held_out_frequency_difference": r["final_stress"]["constraint_violation_frequency"]
                - n["final_stress"]["constraint_violation_frequency"],
                "held_out_tracking_difference": r["final_stress"]["mean_tracking_error"]
                - n["final_stress"]["mean_tracking_error"],
                "nominal_tracking_difference": r["final_nominal"]["mean_tracking_error"]
                - n["final_nominal"]["mean_tracking_error"],
                "stratum_sensitivity": sensitivity,
            }
        )
    frequency = [p["held_out_frequency_difference"] for p in pairs]
    tracking = [p["held_out_tracking_difference"] for p in pairs]
    frequency_stat, tracking_stat = paired_interval(frequency), paired_interval(tracking)
    all_pass = all(not p["incomplete"] and all(p["checks"].values()) for p in pairs)
    floor = all(
        evaluation["seeds"][seed][arm]["final_stress"]["constraint_violation_frequency"] == 0
        for seed in CONFIRM_SEEDS
        for arm in ("normal", "selected")
    )
    improved_frequency = all(x < 0 for x in frequency) and frequency_stat["interval_95"][1] < 0
    improved_tracking = (
        floor and all(x < 0 for x in tracking) and tracking_stat["interval_95"][1] < 0
    )
    if any(not p["incomplete"] and not all(p["checks"].values()) for p in pairs):
        classification = "regressed"
        reason = "At least one complete matched pair violated a frozen nominal or safety guardrail"
    elif all_pass and (improved_frequency or improved_tracking):
        classification = "improved"
        reason = (
            "Consistent held-out paired improvement under the frozen floor "
            "fallback and nominal guardrails"
        )
    else:
        classification = "inconclusive"
        reason = "No consistent, complete paired held-out improvement under the frozen rule"
    return {
        "classification": classification,
        "reason": reason,
        "primary_frequency_floor": floor,
        "co_primary_used": floor,
        "pairs": pairs,
        "uncertainty": {
            "unit": "independent baseline training seed",
            "frequency": frequency_stat,
            "tracking": tracking_stat,
            "frequency_median": statistics.median(frequency),
            "tracking_median": statistics.median(tracking),
            "frequency_direction_counts": {
                "improved": sum(x < 0 for x in frequency),
                "unchanged": sum(x == 0 for x in frequency),
                "worsened": sum(x > 0 for x in frequency),
            },
            "warning": (
                "n=3 paired bootstrap is exploratory; deterministic scenarios "
                "are not independent policies"
            ),
        },
        "policy": policy,
    }


def confirm(run_id, training, experiments):
    """Run selected-versus-normal confirmation, then open the sealed final sets."""
    saved, groups = _validated_study(run_id, training, {"selected"})
    manifest, config = saved["manifest"], saved["config"]
    directory = training.directory(run_id)
    selected = saved["selected_curriculum"]["selected_configuration"]
    if selected is None:
        raise ValueError("No eligible curriculum may enter confirmation")
    frozen = saved["final_protocol"]
    if content_hash(frozen) != manifest["final_protocol_sha256"]:
        raise ValueError("Selected final protocol hash changed")
    if (
        frozen["selected_curriculum"] != selected
        or frozen["preregistered_protocol_sha256"] != manifest["protocol_sha256"]
    ):
        raise ValueError("Selected curriculum differs from frozen decision")
    pools = _pools(saved["candidate_curricula"])
    if [scenario_record(s)["scenario_id"] for s in pools[selected["pool"]]] != frozen[
        "selected_pool_scenario_ids"
    ]:
        raise ValueError("Frozen selected discovery pool changed")
    source32 = training.load(SOURCE_32)
    normal_training = [
        Scenario.from_dict(r["scenario"])
        for r in source32["dataset_partitions"]["groups"]["training"]
    ]
    if (
        content_hash([scenario_record(s) for s in normal_training])
        != saved["protocol"]["training_hash"]
    ):
        raise ValueError("Normal training partition changed")
    if not leakage_check(
        {
            "training": normal_training,
            "known": pools[selected["pool"]],
            "final_nominal": groups["final_nominal"],
            "final_stress": groups["final_stress"],
        }
    )["passed"]:
        raise ValueError("Final physical or seed leakage detected")
    used = manifest["preparation_seconds"] + manifest["screening_seconds"]
    started = time.perf_counter()
    deadline = started + max(1, config["wall_seconds"] - used)
    manifest["status"] = "confirming"
    write_json(directory / "manifest.json", manifest)
    records = []
    try:
        for seed in CONFIRM_SEEDS:
            baseline = metadata_from_dict(saved["baseline_verification"][seed]["checkpoint"])
            verify_checkpoint(baseline.path, baseline.sha256)
            ppo = PPOConfig.from_dict(baseline.ppo_config)
            for arm, ratio in (("normal", 0.0), ("selected", selected["stress_ratio"])):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Phase 3.4 confirmatory wall deadline reached")
                env = PumpGymEnv(
                    normal_training,
                    seed=int(seed) + 5000,
                    observation_schema=OBSERVATION,
                    stress_scenarios=pools[selected["pool"]],
                    stress_ratio=ratio,
                    policy_action_schema=ACTION,
                    reward_schema=REWARD,
                    termination_schema=TERMINATION,
                )
                fingerprint = {}

                def check_start(
                    model,
                    target=fingerprint,
                    expected=saved["baseline_verification"][seed]["fingerprint"]["sha256"],
                ):
                    target.update(model_fingerprint(model))
                    if target["sha256"] != expected:
                        raise ValueError("Confirmatory branch did not restore identical PPO state")

                segment = time.perf_counter()
                model = train_ppo(
                    env,
                    ppo,
                    seed=int(seed) + 5000,
                    timesteps=config["confirm_steps"],
                    checkpoint=baseline.path,
                    expected_sha256=baseline.sha256,
                    wall_seconds=deadline - time.perf_counter(),
                    on_model=check_start,
                )
                elapsed = time.perf_counter() - segment
                meta = _checkpoint(
                    model,
                    directory,
                    run_id,
                    f"confirm_{seed}_{arm}",
                    {
                        "ppo": ppo.to_dict(),
                        "policy_action_schema": ACTION,
                        "reward_schema": REWARD,
                        "termination_schema": TERMINATION,
                    },
                    int(seed) + 5000,
                    OBSERVATION,
                )
                verify_checkpoint(meta.path, meta.sha256)
                manifest["checkpoints"][meta.checkpoint_id] = meta.to_dict()
                actual = model.num_timesteps - baseline.training_timesteps
                record = {
                    "seed": int(seed),
                    "arm": arm,
                    "starting_checkpoint": baseline.checkpoint_id,
                    "starting_sha256": baseline.sha256,
                    "starting_fingerprint": fingerprint,
                    "checkpoint": meta.to_dict(),
                    "requested_additional_steps": config["confirm_steps"],
                    "actual_additional_steps": actual,
                    "training_seconds": elapsed,
                    "continuation_seed": int(seed) + 5000,
                    "ppo_config": ppo.to_dict(),
                    "contracts": saved["protocol"]["contracts"],
                    "stress_ratio": ratio,
                    "sampled_episodes": dict(env.sample_counts),
                    "sampled_scenarios": dict(env.sample_counts_by_scenario_id),
                }
                records.append(record)
                write_json(directory / "continuation_runs.json", {"records": records})
                write_json(directory / "manifest.json", manifest)
                env.close()
                if actual != config["confirm_steps"]:
                    raise TimeoutError("Partial confirmatory arm cannot open final holdouts")
                print(
                    f"{run_id}: seed={seed} arm={arm} steps={actual} seconds={elapsed:.1f}",
                    file=sys.stderr,
                    flush=True,
                )
            left, right = records[-2:]
            for field in (
                "starting_sha256",
                "starting_fingerprint",
                "actual_additional_steps",
                "continuation_seed",
                "ppo_config",
                "contracts",
            ):
                if left[field] != right[field]:
                    raise ValueError(f"Matched confirmatory arms differ in {field}")
        if (
            len(records) != 6
            or sum(r["actual_additional_steps"] for r in records)
            + sum(
                r["actual_additional_steps"] for r in saved["development_results"]["screening_runs"]
            )
            > config["total_training_step_cap"]
        ):
            raise ValueError("Complete matched branches or total interaction budget missing")
        manifest["status"] = "evaluating"
        write_json(directory / "manifest.json", manifest)
        evaluations = {
            "set_hashes": frozen["final_benchmark_hashes"],
            "seeds": {},
            "references": {},
        }
        replays = []
        for seed in CONFIRM_SEEDS:
            evaluations["seeds"][seed] = {}
            for arm in ("normal", "selected"):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Final evaluation deadline reached")
                meta = metadata_from_dict(manifest["checkpoints"][f"confirm_{seed}_{arm}"])
                model = PPOController.load(meta.path, meta)._model
                evaluations["seeds"][seed][arm] = {}
                for set_name in ("final_nominal", "final_stress"):
                    evaluations["seeds"][seed][arm][set_name] = _evaluate(
                        model, meta, groups[set_name], set_name
                    )
                    write_json(directory / "final_evaluation.json", evaluations)
                    record = run_and_save_experiment(
                        experiments,
                        groups[set_name][0],
                        "ppo",
                        max_steps=200,
                        controller_config={"checkpoint": meta.to_dict()},
                    )
                    replays.append(
                        {
                            "seed": int(seed),
                            "arm": arm,
                            "set": set_name,
                            "experiment_id": record.experiment_id,
                            "replay": replay_experiment(
                                experiments, record.experiment_id
                            ).to_dict(),
                        }
                    )
                    write_json(directory / "replay_verification.json", {"records": replays})
        for controller_id, factory, schema in (
            ("pump-rule", PumpRuleController, "pump-minimal:1"),
            ("pump-analytical", AnalyticalPumpController, OBSERVATION),
        ):
            evaluations["references"][controller_id] = {}
            for set_name in ("final_nominal", "final_stress"):
                evaluations["references"][controller_id][set_name] = evaluate_controller(
                    factory,
                    groups[set_name],
                    set_name,
                    max_steps=200,
                    observation_schema=schema,
                    reward_schema=REWARD,
                ).to_dict()
                write_json(directory / "final_evaluation.json", evaluations)
                record = run_and_save_experiment(
                    experiments, groups[set_name][0], controller_id, max_steps=200
                )
                replays.append(
                    {
                        "controller": controller_id,
                        "set": set_name,
                        "experiment_id": record.experiment_id,
                        "replay": replay_experiment(experiments, record.experiment_id).to_dict(),
                    }
                )
                write_json(directory / "replay_verification.json", {"records": replays})
        if any(x["replay"]["status"] not in ("exact_match", "tolerance_match") for x in replays):
            raise ValueError("Final representative replay failed")
        strata = [row["stratum"] for row in saved["stress_benchmark"]["groups"]["final_stress"]]
        result = classify_final(evaluations, strata, frozen["analysis"])
        write_json(directory / "paired_analysis.json", result)
        write_json(
            directory / "regression.json",
            {
                "classification": result["classification"],
                "reason": result["reason"],
                "pairs": result["pairs"],
            },
        )
        manifest["status"] = "completed"
        summary = {
            "study_id": run_id,
            "status": "completed",
            "classification": result["classification"],
            "reason": result["reason"],
            "confirmatory_pairs": 3,
            "additional_confirmation_steps": sum(r["actual_additional_steps"] for r in records),
            "representative_replays": len(replays),
            "failed_replays": sum(
                x["replay"]["status"] not in ("exact_match", "tolerance_match") for x in replays
            ),
            "final_evaluated": True,
        }
    except BaseException as exc:
        manifest["status"] = (
            "interrupted" if isinstance(exc, (KeyboardInterrupt, TimeoutError)) else "failed"
        )
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        summary = {
            "study_id": run_id,
            "status": manifest["status"],
            "classification": "inconclusive",
            "reason": str(exc),
            "completed_arms": len(records),
            "final_evaluated": (directory / "final_evaluation.json").exists(),
        }
        if not isinstance(exc, (KeyboardInterrupt, TimeoutError)):
            raise
    finally:
        manifest["confirmation_seconds"] = time.perf_counter() - started
        manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
        manifest["memory"] = _memory()
        manifest["artifact_bytes"] = sum(
            p.stat().st_size for p in directory.rglob("*") if p.is_file()
        )
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "summary.json", summary)
    return summary
