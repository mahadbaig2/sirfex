"""Checkpoint 2A input-integrity preflight. Never opens bundled final partitions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sirfex.core import Scenario
from sirfex.environments.pump import PumpEnvironment
from sirfex.robustness.disturbance import DIRECTIONS, prepare_grid
from sirfex.robustness.identity import scenario_identity, semantic_hash
from sirfex.training.checkpoints import CheckpointMetadata, sha256_file, write_json
from sirfex.training.feasibility import analyze_feasibility
from sirfex.training.phase34 import STRATA, _candidate, configuration, physical_difficulty
from sirfex.training.protocol import content_hash, leakage_check, scenario_record

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
RUN32 = "20261025T205339_64b622c146"
RUN34 = "20261027T135821_f307f4551c"
RUN35 = "20261027T143749_d88d2dd37f"


def _safe_nominal_prefix() -> dict[str, list[Scenario]]:
    """Generate only the first two Phase 3.2 partitions, then stop the RNG."""
    rng = np.random.default_rng(3101)
    groups = {}
    for index, name in enumerate(("training", "validation")):
        group = []
        for j in range(12):
            capacity, efficiency = float(rng.uniform(0.75, 1.0)), float(rng.uniform(0.8, 1.0))
            leak = float(rng.uniform(0, 0.03))
            demand = float(rng.uniform(0.45, 0.65)) * capacity * efficiency - leak * 0.5
            group.append(
                Scenario(
                    "pump",
                    {
                        "dt": 0.1,
                        "horizon": 200,
                        "min_level": 0.05,
                        "max_level": 0.95,
                        "initial_level": float(rng.uniform(0.3, 0.7)),
                        "target_level": float(rng.uniform(0.4, 0.6)),
                        "pump_capacity": capacity,
                        "efficiency": efficiency,
                        "demand": demand,
                        "leak": leak,
                    },
                    seed=100000 + index * 10000 + j,
                    environment_version="sirfex-pump:1",
                    provenance={"partition": name, "generator": "pump-partitions:1"},
                )
            )
        groups[name] = group
    return groups


def _safe_screening_prefix() -> tuple[list[Scenario], list[dict]]:
    """Generate only Phase 3.4 screening draws; verify against its published hash."""
    config = configuration({"benchmark_seed": 3401})
    rng = np.random.default_rng(config["benchmark_seed"])
    scenarios, records = [], []
    serial = 0
    for stratum in STRATA:
        accepted = attempted = 0
        while accepted < config["development_stress_per_stratum"] and attempted < 600:
            attempted += 1
            serial += 1
            scenario = _candidate(rng, stratum, 3400000 + serial, "screening_stress")
            try:
                analysis = analyze_feasibility(scenario)
            except ValueError:
                continue
            if analysis["finite_horizon_viable"] is not True:
                continue
            if not leakage_check({"seen": scenarios, "candidate": [scenario]})["passed"]:
                continue
            scenarios.append(scenario)
            records.append(
                {
                    **scenario_record(scenario),
                    "stratum": stratum,
                    "difficulty": physical_difficulty(scenario),
                    "feasibility": analysis,
                }
            )
            accepted += 1
        if accepted != 6:
            raise ValueError("Could not reconstruct screening prefix")
    return scenarios, records


def scenarios() -> dict:
    phase35 = json.loads((ROOT / "artifacts/training" / RUN35 / "protocol.json").read_text())
    prefix = _safe_nominal_prefix()
    hashes = {
        name: content_hash([scenario_record(s) for s in group]) for name, group in prefix.items()
    }
    if hashes != {
        "training": phase35["training_hash"],
        "validation": phase35["development_nominal_hash"],
    }:
        raise ValueError(f"Development nominal reconstruction hash mismatch: {hashes}")
    screening, records = _safe_screening_prefix()
    stress_hash = content_hash(records)
    if stress_hash != phase35["development_stress_hash"]:
        raise ValueError(f"Development stress reconstruction hash mismatch: {stress_hash}")
    historical = []
    historical_sources = []
    for search_run in ("20261026T090912_05c97a1dec", "20261027T135757_2ba26db7df"):
        path = ROOT / "artifacts/searches" / search_run / "discoveries.json"
        discoveries = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(discoveries, list):
            raise ValueError("Historical discovery source is not a list")
        for discovery in discoveries:
            if discovery.get("search_id") != search_run or discovery.get("held_out"):
                raise ValueError("Historical discovery is not development-only")
            scenario = Scenario.from_dict(discovery["scenario"])
            if (
                phase35["discovery_ids"].get(scenario_record(scenario)["scenario_id"])
                != discovery["discovery_id"]
            ):
                raise ValueError("Historical discovery identity mismatch")
            historical.append(scenario)
            historical_sources.append(
                {
                    "search_run": search_run,
                    "discovery_id": discovery["discovery_id"],
                    "source_file_sha256": sha256_file(path),
                }
            )
    if len(historical) != len(phase35["discovery_ids"]) or len(historical) != 7:
        raise ValueError("Historical discovery exclusion set incomplete")
    groups = {**prefix, "screening_stress": screening, "historical_discoveries": historical}
    check = leakage_check(groups)
    if not check["passed"]:
        raise ValueError(f"Development partitions overlap: {check}")
    rows = []
    for name, group in groups.items():
        for index, scenario in enumerate(group):
            analysis = analyze_feasibility(scenario)
            historical_source = (
                historical_sources[index] if name == "historical_discoveries" else None
            )
            rows.append(
                {
                    "scenario_id": scenario_identity(scenario),
                    "physical_key": scenario_record(scenario)["scenario_id"],
                    "scenario": scenario.to_dict(),
                    "source_phase": "search"
                    if historical_source
                    else "3.4"
                    if name == "screening_stress"
                    else "3.2",
                    "source_run": historical_source["search_run"]
                    if historical_source
                    else RUN34
                    if name == "screening_stress"
                    else RUN32,
                    "partition": name,
                    "source_partition_hash": historical_source["source_file_sha256"]
                    if historical_source
                    else stress_hash
                    if name == "screening_stress"
                    else hashes[name],
                    "discovery_id": historical_source["discovery_id"]
                    if historical_source
                    else None,
                    "physical_validation": "valid",
                    "feasibility": analysis["classification"],
                    "intended_use": "disturbance_anchor"
                    if name == "validation"
                    else "differential_exclusion_and_context",
                }
            )
    value = {
        "schema_version": "phase37-development-manifest:1",
        "scope": "development_only",
        "provenance": (
            "Generator prefix only; cross-checked with Phase 3.5 development hashes. "
            "Bundled final records unopened."
        ),
        "source_protocol": f"artifacts/training/{RUN35}/protocol.json",
        "source_hashes": {
            **hashes,
            "screening_stress": stress_hash,
            "historical_discovery_pool": phase35["discovery_pool_hash"],
        },
        "leakage": check,
        "anchors": [scenario_identity(s) for s in prefix["validation"]],
        "scenarios": rows,
    }
    value["semantic_sha256"] = semantic_hash(value)
    return value


def controllers() -> dict:
    from sirfex.experiments.factories import make_controller, make_environment

    draft = json.loads(
        (ROOT / "configs/robustness/phase37_checkpoint2_protocol.draft.json").read_text()
    )
    result = []
    for row in draft["controller_inventory"]:
        if row["group"] == "reference":
            env = make_environment("pump")
            try:
                controller = make_controller(row["controller_id"], env, {})
                result.append(
                    {
                        **row,
                        "status": "reference_factory_verified",
                        "factory_compatible": controller is not None,
                    }
                )
            finally:
                env.close()
            continue
        source = ROOT / "artifacts/training" / row["source_run"] / "checkpoints"
        metadata_path = source / f"{row['source_key']}.metadata.json"
        checkpoint_path = source / f"{row['source_key']}.zip"
        record = {
            k: v
            for k, v in row.items()
            if k not in {"file_available", "sha256_verified", "loadability_tested"}
        }
        record.update(
            {
                "source_artifact": (
                    f"artifacts/training/{row['source_run']}/checkpoints/{row['source_key']}.zip"
                ),
                "metadata_verified": False,
                "file_available": checkpoint_path.is_file(),
                "sha256_verified": False,
                "loadability_tested": False,
                "deterministic_inference_verified": False,
            }
        )
        try:
            raw = json.loads(metadata_path.read_text())
            metadata = CheckpointMetadata(
                **{
                    **raw,
                    "path": str(checkpoint_path),
                    "observation_shape": tuple(raw["observation_shape"]),
                    "action_shape": tuple(raw["action_shape"]),
                }
            )
            if (
                metadata.checkpoint_id != row["source_key"]
                or metadata.training_run_id != row["source_run"]
                or metadata.sha256 != row["sha256"]
                or metadata.environment_id != "pump"
            ):
                raise ValueError("Checkpoint metadata identity mismatch")
            record["metadata_verified"] = True
            if not record["file_available"]:
                raise FileNotFoundError(checkpoint_path.name)
            record["actual_sha256"] = sha256_file(checkpoint_path)
            if record["actual_sha256"] != row["sha256"]:
                raise ValueError("Checkpoint SHA-256 mismatch")
            record["sha256_verified"] = True
            env = make_environment("pump")
            try:
                controller = make_controller("ppo", env, {"checkpoint": metadata.to_dict()})
                record["loadability_tested"] = True
                record["observation_compatible"] = (
                    tuple(env.observation_space.shape) == metadata.observation_shape
                )
                record["action_compatible"] = (
                    tuple(controller._model.action_space.shape) == metadata.action_shape
                )
                observation = np.zeros(metadata.observation_shape, dtype=np.float32)
                if not env.observation_space.contains(observation):
                    raise ValueError("Synthetic observation outside declared observation space")
                a = controller.act(observation, deterministic=True)
                b = controller.act(observation, deterministic=True)
                if not (
                    np.array_equal(a, b) and np.all(np.isfinite(a)) and env.action_space.contains(a)
                ):
                    raise ValueError("Deterministic action contract failed")
                record["deterministic_inference_verified"] = True
            finally:
                env.close()
                if "controller" in locals():
                    del controller
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        result.append(record)
        print(
            f"controller {row['source_key']}: {record['deterministic_inference_verified']}",
            flush=True,
        )
    return {
        "schema_version": "phase37-controller-preflight:1",
        "scope": "input_integrity_only",
        "controllers": result,
    }


def domain(manifest: dict) -> dict:
    anchors = [
        Scenario.from_dict(row["scenario"])
        for row in manifest["scenarios"]
        if row["partition"] == "validation"
    ]
    grid = tuple(round(i * 0.025, 3) for i in range(25))
    rows = []
    for anchor in anchors:
        p = PumpEnvironment().validate_scenario(anchor)
        for direction in DIRECTIONS:
            prepared = prepare_grid(anchor, direction, grid)
            statuses = [point.domain_status for point in prepared.points]
            viable = [point.viable for point in prepared.points if point.domain_status == "valid"]
            nonmonotone = any(
                a is False and b is True for a, b in zip(viable, viable[1:], strict=False)
            )
            valid = [point for point in prepared.points if point.domain_status == "valid"]
            refinement = (
                direction != ("surplus_demand",)
                and not nonmonotone
                and all(
                    0 <= p["dt"] * PumpEnvironment().validate_scenario(point.scenario)["leak"] <= 1
                    for point in valid
                )
            )
            rows.append(
                {
                    "anchor_id": scenario_identity(anchor),
                    "direction": list(direction),
                    "valid": statuses.count("valid"),
                    "censored": statuses.count("censored_bound"),
                    "undefined": statuses.count("undefined_axis"),
                    "infeasible": viable.count(False),
                    "unknown": viable.count(None),
                    "validity_reopened": False,
                    "feasibility_nonmonotone": nonmonotone,
                    "refinement_eligible_numerically": refinement,
                    "prepared_grid_sha256": prepared.semantic_sha256,
                }
            )
    return {
        "schema_version": "phase37-disturbance-domain-preflight:1",
        "scope": "scenario_physics_only",
        "anchors": len(anchors),
        "directions": len(DIRECTIONS),
        "grid_points": len(grid),
        "cells": rows,
        "totals": {
            key: sum(row[key] for row in rows)
            for key in ("valid", "censored", "undefined", "infeasible", "unknown")
        },
        "estimated_controller_evaluations_upper": 17 * len(anchors) * len(DIRECTIONS) * len(grid),
        "refinement_protocol_decision": "disabled_pending_formal_monotonicity_review",
    }


def main() -> None:
    manifest = scenarios()
    write_json(EVIDENCE / "phase_3_7_development_scenario_manifest.json", manifest)
    print("development scenario hashes verified", flush=True)
    inventory = controllers()
    write_json(EVIDENCE / "phase_3_7_controller_preflight.json", inventory)
    result = domain(manifest)
    write_json(EVIDENCE / "phase_3_7_disturbance_domain_preflight.json", result)
    print(f"domain totals: {result['totals']}", flush=True)


if __name__ == "__main__":
    main()
