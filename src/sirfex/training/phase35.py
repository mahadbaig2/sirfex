"""Exposure-controlled, development-gated continuation study."""

from __future__ import annotations

import json
import platform
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
from sirfex.training.checkpoints import verify_checkpoint, write_json
from sirfex.training.config import PPOConfig
from sirfex.training.controller import PPOController
from sirfex.training.evaluation import evaluate_controller
from sirfex.training.exposure import DEFAULT_GATE, exposure_adequacy
from sirfex.training.matched_study import (
    ACTION,
    OBSERVATION,
    REWARD,
    TERMINATION,
    _evaluate,
    _selected_baselines,
    model_fingerprint,
)
from sirfex.training.phase34 import (
    CONFIRM_SEEDS,
    SOURCE_32,
    _groups,
    _memory,
    _pools,
    classify_final,
)
from sirfex.training.ppo import dependency_versions, train_ppo
from sirfex.training.protocol import content_hash, leakage_check, scenario_record
from sirfex.training.pump_gym import PumpGymEnv
from sirfex.training.remediation_diagnostics import metadata_from_dict
from sirfex.training.runner import _git_commit
from sirfex.training.validation import _checkpoint

SOURCE_34 = "20261027T135821_f307f4551c"
EXPECTED_FINAL = {
    "final_stress": "c992525e1f9e382432dfe3fceebef0a0e9339a16432a4767fb12455687816226",
    "final_nominal": "da86e081c624fe65009d89505c48c7760d0a2692070b6c7e7d463dfa886f1f94",
}
POLICY = {
    "nominal_relative_mae_max": 0.10,
    "nominal_violation_increase_max": 0,
    "stress_violation_increase_max": 0,
    "execution_failure_increase_max": 0,
    "development_stress_mae_gain_min": 0.005,
}
SCREEN_SEEDS = ("3", "17")
CANDIDATES = (
    {"id": "normal", "target_stress_interaction_fraction": 0.0},
    {"id": "expanded_5", "target_stress_interaction_fraction": 0.05},
    {"id": "expanded_10", "target_stress_interaction_fraction": 0.10},
)


def _read(directory, name):
    return json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))


def _source(training):
    old = training.load(SOURCE_34)
    if old["manifest"]["status"] != "no_eligible_curriculum":
        raise ValueError("Phase 3.4 must have stopped before final evaluation")
    forbidden = ("final_protocol", "final_evaluation", "continuation_runs", "paired_analysis")
    if any(name in old for name in forbidden):
        raise ValueError("Phase 3.4 final partitions were opened")
    groups = _groups(old["stress_benchmark"])
    if any(
        old["stress_benchmark"]["hashes"][key] != value for key, value in EXPECTED_FINAL.items()
    ):
        raise ValueError("Published final partition hash changed")
    source32 = training.load(SOURCE_32)
    baselines = _selected_baselines(source32)
    training_pool = [
        Scenario.from_dict(row["scenario"])
        for row in source32["dataset_partitions"]["groups"]["training"]
    ]
    nominal = [
        Scenario.from_dict(row["scenario"])
        for row in source32["dataset_partitions"]["groups"]["validation"]
    ]
    pool = _pools(old["candidate_curricula"])["expanded"]
    if len(pool) != 7:
        raise ValueError("Expanded discovery pool must have seven physical scenarios")
    if not leakage_check(
        {
            "training": training_pool,
            "discovery": pool,
            "development_nominal": nominal,
            "development_stress": groups["screening_stress"],
            "final_nominal": groups["final_nominal"],
            "final_stress": groups["final_stress"],
        }
    )["passed"]:
        raise ValueError("Physical or seed leakage into final partitions")
    screen_ids = {
        key
        for row in old["development_results"]["screening_runs"]
        for key in row["sampled_scenarios"]
    }
    final_ids = {
        row["scenario_id"]
        for name in EXPECTED_FINAL
        for row in old["stress_benchmark"]["groups"][name]
    }
    if screen_ids & final_ids:
        raise ValueError("Phase 3.4 training touched final partitions")
    discovery_ids = {
        scenario_record(Scenario.from_dict(value["scenario"]))["scenario_id"]: key
        for category in ("original", "new")
        for key, value in old["candidate_curricula"][category].items()
    }
    return old, groups, baselines, training_pool, nominal, pool, discovery_ids


def _calibrate(training_pool, stress_pool, discovery_ids):
    """Measure neutral transition throughput and reset frequency, not policy outcomes."""
    rows = []
    for fraction in (0.05, 0.10):
        env = PumpGymEnv(
            training_pool,
            seed=3500,
            observation_schema=OBSERVATION,
            stress_scenarios=stress_pool,
            discovery_ids=discovery_ids,
            stress_target_fraction=fraction,
            policy_action_schema=ACTION,
            reward_schema=REWARD,
            termination_schema=TERMINATION,
        )
        started = time.perf_counter()
        count = 0
        env.reset()
        while count < 4096:
            _, _, done, truncated, _ = env.step(np.array([0.5]))
            count += 1
            if (done or truncated) and count < 4096:
                env.reset()
        elapsed = time.perf_counter() - started
        summary = env.exposure_summary(scheduled_interactions=4096, model_interactions=4096)
        env.close()
        if not summary["reconciliation_valid"]:
            raise ValueError("Neutral calibration accounting failed")
        rows.append(
            {
                "target": fraction,
                "neutral_steps": count,
                "seconds": elapsed,
                "neutral_steps_per_second": count / elapsed,
                "exposure": summary,
            }
        )
    return {
        "schema_version": "phase-3.5-calibration:1",
        "neutral_action": 0.5,
        "rows": rows,
        "note": "No candidate policy performance evaluated; PPO throughput is lower.",
    }


def prepare(value, training):
    if not isinstance(value, dict) or set(value) - {
        "screen_steps",
        "confirm_steps",
        "wall_seconds",
    }:
        raise ValueError("Unknown Phase 3.5 configuration")
    screen_steps = value.get("screen_steps", 51200)
    confirm_steps = value.get("confirm_steps", 51200)
    wall = value.get("wall_seconds", 5400)
    if any(type(x) is not int or x <= 0 for x in (screen_steps, confirm_steps, wall)):
        raise ValueError("Positive integer step and wall limits required")
    if screen_steps % 128 or confirm_steps % 128:
        raise ValueError("PPO interaction budgets must be multiples of 128")
    if screen_steps < 25600:
        raise ValueError("Exposure target requires at least 25,600 screening steps")
    started = time.perf_counter()
    old, groups, baselines, train_pool, nominal, pool, discovery_ids = _source(training)
    calibration = _calibrate(train_pool, pool, discovery_ids)
    # With a 200-step maximum episode, 51,200 steps at 5% target implies
    # at least 12 completed stress episodes, barring early physical termination.
    protocol = {
        "schema_version": "phase-3.5-protocol:1",
        "source_phase_3_4": SOURCE_34,
        "source_phase_3_2": SOURCE_32,
        "selection_uses_final_outcomes": False,
        "sampler": "interaction-debt:1",
        "sampler_reason": "Phase 3.4 probability arms saw one stress episode each at 5120 steps; "
        "episode-boundary interaction targets reduce exposure variance.",
        "candidates": list(CANDIDATES),
        "screen_seeds": list(SCREEN_SEEDS),
        "confirm_seeds": list(CONFIRM_SEEDS),
        "continuation_seed_offset": 5000,
        "screen_steps_per_arm": screen_steps,
        "confirm_steps_per_arm": confirm_steps,
        "total_step_cap": 6 * screen_steps + 6 * confirm_steps,
        "wall_seconds": wall,
        "exposure_gate": DEFAULT_GATE,
        "selection_policy": POLICY,
        "selection_rule": "Each candidate must pass exposure and all development guards "
        "on both screen seeds; choose lowest mean stress violation, then "
        "lowest mean stress MAE, then stable ID.",
        "baseline_sha256": {s: baselines[s]["checkpoint"]["sha256"] for s in CONFIRM_SEEDS},
        "baseline_fingerprint": {s: baselines[s]["fingerprint"]["sha256"] for s in CONFIRM_SEEDS},
        "training_hash": content_hash([scenario_record(s) for s in train_pool]),
        "development_nominal_hash": content_hash([scenario_record(s) for s in nominal]),
        "development_stress_hash": old["stress_benchmark"]["hashes"]["screening_stress"],
        "final_hashes": EXPECTED_FINAL,
        "discovery_pool_hash": content_hash(old["candidate_curricula"]),
        "discovery_ids": discovery_ids,
        "contracts": {
            "observation": OBSERVATION,
            "action": ACTION,
            "reward": REWARD,
            "termination": TERMINATION,
        },
        "final_analysis": old["protocol"]["final_analysis"],
        "final_primary": "final feasible-stress violation frequency",
        "final_co_primary": "final feasible-stress mean absolute tracking error",
        "final_independent_unit": "baseline policy training seed",
    }
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:10]
    manifest = {
        "schema_version": "2",
        "phase": "3.5",
        "kind": "exposure-controlled-curriculum-study",
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
            "config.json": value,
            "protocol.json": protocol,
            "exposure_audit.json": {
                "source_status": old["manifest"]["status"],
                "source_final_files_absent": True,
                "final_hashes": EXPECTED_FINAL,
                "source_screen_sampled_final_count": 0,
                "leakage_check_passed": True,
            },
            "calibration.json": calibration,
            "stress_benchmark.json": old["stress_benchmark"],
        },
    )
    return {
        "study_id": run_id,
        "status": "prepared",
        "protocol_sha256": manifest["protocol_sha256"],
        "calibration": [
            {k: r[k] for k in ("target", "neutral_steps_per_second")} for r in calibration["rows"]
        ],
    }


def _load(run_id, training, statuses):
    saved = training.load(run_id, allow_partial=True)
    if saved["manifest"].get("phase") != "3.5" or saved["manifest"]["status"] not in statuses:
        raise ValueError(f"Phase 3.5 status must be one of {sorted(statuses)}")
    if content_hash(saved["protocol"]) != saved["manifest"]["protocol_sha256"]:
        raise ValueError("Frozen Phase 3.5 protocol hash changed")
    return saved


def _run_arm(
    saved, training, run_id, seed, candidate, stage, train_pool, pool, discovery_ids, deadline
):
    protocol = saved["protocol"]
    baseline = metadata_from_dict(saved["baseline_verification"][seed]["checkpoint"])
    verify_checkpoint(baseline.path, protocol["baseline_sha256"][seed])
    fraction = candidate["target_stress_interaction_fraction"]
    sampler_seed = int(seed) + protocol["continuation_seed_offset"]
    env = PumpGymEnv(
        train_pool,
        seed=sampler_seed,
        observation_schema=OBSERVATION,
        stress_scenarios=pool if fraction else [],
        stress_target_fraction=fraction,
        discovery_ids=discovery_ids if fraction else {},
        policy_action_schema=ACTION,
        reward_schema=REWARD,
        termination_schema=TERMINATION,
    )
    ppo = PPOConfig.from_dict(baseline.ppo_config)
    fingerprint = {}

    def check_start(model):
        fingerprint.update(model_fingerprint(model))
        if fingerprint["sha256"] != protocol["baseline_fingerprint"][seed]:
            raise ValueError("Starting PPO weights or optimizer state changed")

    steps = protocol[f"{stage}_steps_per_arm"]
    started = time.perf_counter()
    try:
        model = train_ppo(
            env,
            ppo,
            seed=sampler_seed,
            timesteps=steps,
            checkpoint=baseline.path,
            expected_sha256=baseline.sha256,
            wall_seconds=deadline - started,
            on_model=check_start,
        )
        actual = model.num_timesteps - baseline.training_timesteps
        exposure = env.exposure_summary(scheduled_interactions=steps, model_interactions=actual)
        if not exposure["reconciliation_valid"]:
            raise ValueError("PPO and physical transition accounting disagree")
        meta = _checkpoint(
            model,
            training.directory(run_id),
            run_id,
            f"{stage}_{seed}_{candidate['id']}",
            {
                "ppo": ppo.to_dict(),
                "policy_action_schema": ACTION,
                "reward_schema": REWARD,
                "termination_schema": TERMINATION,
            },
            sampler_seed,
            OBSERVATION,
        )
        verify_checkpoint(meta.path, meta.sha256)
        return (
            model,
            meta,
            {
                "seed": int(seed),
                "candidate_id": candidate["id"],
                "target_stress_interaction_fraction": fraction,
                "starting_sha256": baseline.sha256,
                "starting_fingerprint": fingerprint,
                "checkpoint": meta.to_dict(),
                "continuation_seed": sampler_seed,
                "requested_additional_steps": steps,
                "actual_additional_steps": actual,
                "training_seconds": time.perf_counter() - started,
                "exposure": exposure,
                "adequacy": (
                    exposure_adequacy(exposure, len(pool), trial_complete=actual == steps)
                    if fraction
                    else None
                ),
            },
        )
    finally:
        env.close()


def select(rows):
    """Apply the frozen rule per seed; no final outcomes enter this decision."""
    by = {(str(r["seed"]), r["candidate_id"]): r for r in rows}
    if any((seed, candidate["id"]) not in by for seed in SCREEN_SEEDS for candidate in CANDIDATES):
        return {
            "status": "invalid_study",
            "selected_candidate_id": None,
            "reason": "Incomplete screening matrix",
            "decisions": [],
        }
    decisions, eligible = [], []
    for candidate in CANDIDATES[1:]:
        per_seed = []
        for seed in SCREEN_SEEDS:
            n, r = by[seed, "normal"], by[seed, candidate["id"]]
            adequate = r["adequacy"]["status"] == "adequate_exposure"
            equal = (
                n["actual_additional_steps"]
                == r["actual_additional_steps"]
                == n["requested_additional_steps"]
                == r["requested_additional_steps"]
            )
            nom_n, nom_r = n["development_nominal"], r["development_nominal"]
            stress_n, stress_r = n["development_stress"], r["development_stress"]
            checks = {
                "exposure": adequate,
                "equal_budget": equal,
                "nominal_mae": nom_r["mean_tracking_error"] <= nom_n["mean_tracking_error"] * 1.10,
                "nominal_violations": nom_r["constraint_violation_frequency"]
                <= nom_n["constraint_violation_frequency"],
                "stress_violations": stress_r["constraint_violation_frequency"]
                <= stress_n["constraint_violation_frequency"],
                "execution": all(
                    r[k]["execution_failures"] <= n[k]["execution_failures"]
                    for k in ("development_nominal", "development_stress")
                ),
                "stress_gain": (
                    stress_r["constraint_violation_frequency"]
                    < stress_n["constraint_violation_frequency"]
                    or stress_r["mean_tracking_error"] <= stress_n["mean_tracking_error"] - 0.005
                ),
            }
            per_seed.append({"seed": int(seed), "checks": checks, "eligible": all(checks.values())})
        decision = {
            "candidate_id": candidate["id"],
            "per_seed": per_seed,
            "eligible": all(item["eligible"] for item in per_seed),
        }
        decisions.append(decision)
        if decision["eligible"]:
            eligible.append(candidate)
    winner = min(
        eligible,
        key=lambda c: (
            np.mean(
                [
                    by[s, c["id"]]["development_stress"]["constraint_violation_frequency"]
                    for s in SCREEN_SEEDS
                ]
            ),
            np.mean(
                [by[s, c["id"]]["development_stress"]["mean_tracking_error"] for s in SCREEN_SEEDS]
            ),
            c["id"],
        ),
        default=None,
    )
    statuses = [by[s, c["id"]]["adequacy"]["status"] for s in SCREEN_SEEDS for c in CANDIDATES[1:]]
    status = (
        "invalid_study"
        if "invalid_exposure" in statuses
        else "selected_curriculum"
        if winner
        else "insufficient_exposure"
        if "insufficient_exposure" in statuses
        else "no_eligible_curriculum"
    )
    return {
        "status": status,
        "selected_candidate_id": winner["id"] if winner else None,
        "selected_configuration": winner,
        "decisions": decisions,
    }


def screen(run_id, training):
    saved = _load(run_id, training, {"prepared"})
    old, groups, baselines, train_pool, nominal, pool, discovery_ids = _source(training)
    saved["baseline_verification"] = baselines
    protocol, manifest = saved["protocol"], saved["manifest"]
    if content_hash(old["candidate_curricula"]) != protocol["discovery_pool_hash"]:
        raise ValueError("Frozen discovery pool changed")
    if (
        content_hash([scenario_record(s) for s in train_pool]) != protocol["training_hash"]
        or content_hash([scenario_record(s) for s in nominal])
        != protocol["development_nominal_hash"]
    ):
        raise ValueError("Frozen Phase 3.5 development partitions changed")
    directory = training.directory(run_id)
    deadline = time.perf_counter() + protocol["wall_seconds"] - manifest["preparation_seconds"]
    rows = []
    manifest["status"] = "screening"
    write_json(directory / "manifest.json", manifest)
    try:
        for seed in SCREEN_SEEDS:
            for candidate in CANDIDATES:
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Screening wall deadline reached")
                model, meta, row = _run_arm(
                    saved,
                    training,
                    run_id,
                    seed,
                    candidate,
                    "screen",
                    train_pool,
                    pool,
                    discovery_ids,
                    deadline,
                )
                manifest["checkpoints"][meta.checkpoint_id] = meta.to_dict()
                rows.append(row)
                write_json(directory / "screening_runs.json", {"records": rows})
                write_json(directory / "manifest.json", manifest)
                if row["actual_additional_steps"] != protocol["screen_steps_per_arm"]:
                    raise TimeoutError("Partial screening arm")
                if row["adequacy"] and row["adequacy"]["status"] == "invalid_exposure":
                    raise ValueError("Invalid curriculum exposure")
                row["development_nominal"] = _evaluate(model, meta, nominal, "development_nominal")
                row["development_stress"] = _evaluate(
                    model, meta, groups["screening_stress"], "development_stress"
                )
                write_json(directory / "screening_runs.json", {"records": rows})
                print(
                    f"{run_id} screen seed={seed} {candidate['id']} steps="
                    f"{row['actual_additional_steps']} "
                    f"stress={row['exposure']['stress_interactions']} "
                    f"gate={row['adequacy']['status'] if row['adequacy'] else 'control'}",
                    file=sys.stderr,
                    flush=True,
                )
        selection = select(rows)
        manifest["status"] = selection["status"]
        write_json(directory / "selected_curriculum.json", selection)
        if selection["status"] == "selected_curriculum":
            frozen = {
                "schema_version": "phase-3.5-final:1",
                "protocol_sha256": manifest["protocol_sha256"],
                "selected_configuration": selection["selected_configuration"],
                "final_hashes": protocol["final_hashes"],
                "confirm_seeds": protocol["confirm_seeds"],
                "confirm_steps_per_arm": protocol["confirm_steps_per_arm"],
                "final_analysis": protocol["final_analysis"],
            }
            manifest["final_protocol_sha256"] = content_hash(frozen)
            write_json(directory / "final_protocol.json", frozen)
    except BaseException as exc:
        manifest["status"] = (
            "interrupted" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "failed"
        )
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if not isinstance(exc, (TimeoutError, KeyboardInterrupt)):
            raise
    finally:
        manifest["memory"] = _memory()
        write_json(directory / "manifest.json", manifest)
        write_json(
            directory / "summary.json",
            {
                "study_id": run_id,
                "status": manifest["status"],
                "screening_arms_completed": len(rows),
            },
        )
    return {
        "study_id": run_id,
        "status": manifest["status"],
        "selection": _read(directory, "selected_curriculum")
        if (directory / "selected_curriculum.json").exists()
        else None,
    }


def confirm(run_id, training, experiments):
    saved = _load(run_id, training, {"selected_curriculum"})
    old, groups, baselines, train_pool, _, pool, discovery_ids = _source(training)
    saved["baseline_verification"] = baselines
    protocol, manifest = saved["protocol"], saved["manifest"]
    if content_hash(old["candidate_curricula"]) != protocol["discovery_pool_hash"]:
        raise ValueError("Frozen discovery pool changed")
    directory = training.directory(run_id)
    frozen = saved["final_protocol"]
    if (
        content_hash(frozen) != manifest["final_protocol_sha256"]
        or frozen["final_hashes"] != EXPECTED_FINAL
    ):
        raise ValueError("Final selection or holdout hash changed")
    selected = frozen["selected_configuration"]
    deadline = (
        time.perf_counter()
        + protocol["wall_seconds"]
        - manifest["preparation_seconds"]
        - sum(row["training_seconds"] for row in saved["screening_runs"]["records"])
    )
    records = []
    manifest["status"] = "confirming"
    write_json(directory / "manifest.json", manifest)
    try:
        for seed in CONFIRM_SEEDS:
            for candidate in (CANDIDATES[0], selected):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Confirmation wall deadline reached")
                model, meta, row = _run_arm(
                    saved,
                    training,
                    run_id,
                    seed,
                    candidate,
                    "confirm",
                    train_pool,
                    pool,
                    discovery_ids,
                    deadline,
                )
                manifest["checkpoints"][meta.checkpoint_id] = meta.to_dict()
                records.append(row)
                write_json(directory / "confirmation_runs.json", {"records": records})
                write_json(directory / "manifest.json", manifest)
                if row["actual_additional_steps"] != protocol["confirm_steps_per_arm"]:
                    raise TimeoutError("Partial confirmation arm")
                if row["adequacy"] and row["adequacy"]["status"] != "adequate_exposure":
                    raise ValueError("Confirmatory stress exposure below frozen gate")
                print(
                    f"{run_id} confirm seed={seed} {candidate['id']} complete",
                    file=sys.stderr,
                    flush=True,
                )
            left, right = records[-2:]
            if any(
                left[key] != right[key]
                for key in (
                    "starting_sha256",
                    "starting_fingerprint",
                    "continuation_seed",
                    "actual_additional_steps",
                )
            ):
                raise ValueError("Matched confirmation state or budget differs")
        if (
            sum(r["actual_additional_steps"] for r in records)
            + sum(r["actual_additional_steps"] for r in saved["screening_runs"]["records"])
            > protocol["total_step_cap"]
        ):
            raise ValueError("Training step cap exceeded")
        manifest["status"] = "evaluating"
        write_json(directory / "manifest.json", manifest)
        evaluation = {"set_hashes": EXPECTED_FINAL, "seeds": {}, "references": {}}
        replays = []
        known = pool
        for seed in CONFIRM_SEEDS:
            evaluation["seeds"][seed] = {}
            for arm, candidate in (("normal", CANDIDATES[0]), ("selected", selected)):
                meta = metadata_from_dict(
                    manifest["checkpoints"][f"confirm_{seed}_{candidate['id']}"]
                )
                model = PPOController.load(meta.path, meta)._model
                evaluation["seeds"][seed][arm] = {}
                for name, scenarios in (
                    ("final_nominal", groups["final_nominal"]),
                    ("final_stress", groups["final_stress"]),
                    ("known_stress", known),
                ):
                    evaluation["seeds"][seed][arm][name] = _evaluate(model, meta, scenarios, name)
                    write_json(directory / "final_evaluation.json", evaluation)
                record = run_and_save_experiment(
                    experiments,
                    groups["final_stress"][0],
                    "ppo",
                    max_steps=200,
                    controller_config={"checkpoint": meta.to_dict()},
                )
                replays.append(
                    {
                        "seed": int(seed),
                        "arm": arm,
                        "experiment_id": record.experiment_id,
                        "replay": replay_experiment(experiments, record.experiment_id).to_dict(),
                    }
                )
                write_json(directory / "replay_verification.json", {"records": replays})
        for name, cls, schema in (
            ("pump-rule", PumpRuleController, "pump-minimal:1"),
            ("pump-analytical", AnalyticalPumpController, OBSERVATION),
        ):
            evaluation["references"][name] = {}
            for set_name in EXPECTED_FINAL:
                evaluation["references"][name][set_name] = evaluate_controller(
                    cls,
                    groups[set_name],
                    set_name,
                    max_steps=200,
                    observation_schema=schema,
                    reward_schema=REWARD,
                ).to_dict()
            write_json(directory / "final_evaluation.json", evaluation)
        analysis = classify_final(
            evaluation,
            [row["stratum"] for row in saved["stress_benchmark"]["groups"]["final_stress"]],
            protocol["final_analysis"],
        )
        write_json(directory / "paired_analysis.json", analysis)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = (
            "interrupted" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "failed"
        )
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if not isinstance(exc, (TimeoutError, KeyboardInterrupt)):
            raise
    finally:
        manifest["memory"] = _memory()
        write_json(directory / "manifest.json", manifest)
        write_json(
            directory / "summary.json",
            {
                "study_id": run_id,
                "status": manifest["status"],
                "confirmation_arms_completed": len(records),
            },
        )
    return {"study_id": run_id, "status": manifest["status"]}
