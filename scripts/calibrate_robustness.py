"""Bounded, synthetic Checkpoint 1.1 cost measurement; no training or study run."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from sirfex import __version__
from sirfex.core import EvaluationRunner, Scenario
from sirfex.experiments.factories import make_controller, make_environment
from sirfex.experiments.manifests import ExperimentManifest
from sirfex.experiments.storage import ExperimentStore
from sirfex.robustness.disturbance import prepare_grid
from sirfex.robustness.identity import scenario_identity, semantic_hash
from sirfex.robustness.margins import evaluate_margin
from sirfex.robustness.report import build_report, render_markdown
from sirfex.training.checkpoints import verify_checkpoint


def timed(function):
    started = time.perf_counter()
    value = function()
    return value, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metadata = json.loads(Path(args.checkpoint_metadata).read_text(encoding="utf-8"))
    if metadata.get("environment_id") != "pump":
        raise ValueError("Calibration requires a pump checkpoint")
    checkpoint = Path(metadata["path"])
    startup = []
    for _ in range(3):
        _, elapsed = timed(
            lambda: subprocess.run([sys.executable, "-c", "pass"], check=True, capture_output=True)
        )
        startup.append(elapsed)
    import_times = {}
    _, import_times["torch_seconds"] = timed(lambda: __import__("torch"))
    _, import_times["stable_baselines3_seconds"] = timed(lambda: __import__("stable_baselines3"))
    _, verification_seconds = timed(lambda: verify_checkpoint(checkpoint, metadata["sha256"]))
    scenario = Scenario(
        "pump",
        parameters={
            "dt": 0.1,
            "horizon": 200,
            "initial_level": 0.5,
            "target_level": 0.5,
            "demand": 0.35,
            "pump_capacity": 0.8,
            "efficiency": 0.95,
            "leak": 0.01,
            "min_level": 0.05,
            "max_level": 0.95,
        },
        seed=371100,
        provenance={"partition": "synthetic_software_calibration"},
    )
    specs = [
        {"id": "pump-rule", "config": {}},
        {"id": "pump-analytical", "config": {"dt": 0.1, "response": 0.25}},
        {"id": "ppo", "config": {"checkpoint": metadata}},
    ]
    measurements = {}
    sample_directions = (("demand",), ("capacity",), ("leak",), ("surplus_demand",))
    sample_grid = [0.0, 0.025, 0.05, 0.075, 0.1]
    _, grid_preparation_seconds = timed(
        lambda: [prepare_grid(scenario, direction, sample_grid) for direction in sample_directions]
    )
    first_result = None
    for spec in specs:
        environment = make_environment("pump")
        controller, init_seconds = timed(
            lambda spec=spec, environment=environment: make_controller(
                spec["id"], environment, spec["config"]
            )
        )
        runner = EvaluationRunner(200)

        def episode(runner=runner, environment=environment, controller=controller):
            return runner.run_episode(environment, controller, scenario, record_trace=True)

        warmup, warmup_seconds = timed(episode)
        results = []
        elapsed = []
        for _ in range(3):
            result, seconds = timed(episode)
            results.append(result)
            elapsed.append(seconds)
        observation, _ = environment.reset(scenario)
        controller.reset()
        _, acts_seconds = timed(
            lambda controller=controller, observation=observation: [
                controller.act(observation) for _ in range(100)
            ]
        )
        transitions = sum(row.episode_length for row in results)
        measurements[spec["id"]] = {
            "initialization_seconds": init_seconds,
            "warmup_episode_seconds": warmup_seconds,
            "warmup_transitions": warmup.episode_length,
            "measured_episode_seconds": elapsed,
            "measured_transitions": transitions,
            "mean_seconds_per_transition_with_trace": sum(elapsed) / transitions,
            "isolated_100_actions_seconds": acts_seconds,
            "isolated_seconds_per_action": acts_seconds / 100,
            "full_horizon_episodes": sum(row.episode_length == 200 for row in results),
        }
        first_result = first_result or results[0]
        environment.close()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    experiment_store = ExperimentStore(output.parent / "checkpoint1_1_persistence_probe")
    experiment_id = "robust_calibration_" + semantic_hash(scenario.to_dict())[:20]
    manifest = ExperimentManifest(
        experiment_id=experiment_id,
        created_at_utc=datetime.now(UTC).isoformat(),
        sirfex_version=__version__,
        environment_id="pump",
        environment_version=scenario.environment_version,
        controller_id=first_result.controller_id,
        controller_metadata={},
        controller_config={},
        scenario=scenario.to_dict(),
        evaluation_config={"max_steps": 200, "deterministic": True},
        dependencies={},
        execution_status=first_result.execution_status.value,
        outcome=first_result.outcome,
    )
    # A fresh experiment ID keeps the probe append-only on repeated calibration.
    experiment_id += "_" + str(time.time_ns())[-8:]
    manifest.experiment_id = experiment_id
    directory, persistence_seconds = timed(
        lambda: experiment_store.save(manifest, scenario, first_result)
    )
    protocol = {
        "schema_version": "pump-robustness-protocol:1",
        "profile": "software_calibration_only",
        "calibration_only": True,
    }
    margin = evaluate_margin(first_result).to_dict()
    report, generation_seconds = timed(
        lambda: build_report(
            profile="software_calibration_only",
            protocol=protocol,
            controllers=[{"id": first_result.controller_id}],
            scenarios=[
                {"scenario_id": scenario_identity(scenario), "scenario": scenario.to_dict()}
            ],
            margin_results=[margin],
            disturbance_cells=[],
            differential_results=[],
            reproducibility={"command": "scripts/calibrate_robustness.py"},
        )
    )
    _, rendering_seconds = timed(lambda: render_markdown(report))
    payload = {
        "schema_version": "pump-robustness-cost-calibration:1",
        "profile": "software_calibration_only",
        "scientific_result": False,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "scenario_id": scenario_identity(scenario),
        "horizon": 200,
        "checkpoint_sha256": metadata["sha256"],
        "process_startup_seconds": startup,
        "ml_import_seconds": import_times,
        "checkpoint_verification_seconds": verification_seconds,
        "controllers": measurements,
        "grid_preparation": {
            "directions": 4,
            "points_per_direction": 5,
            "seconds": grid_preparation_seconds,
        },
        "persistence_seconds": persistence_seconds,
        "persistence_bytes": sum(
            path.stat().st_size for path in directory.iterdir() if path.is_file()
        ),
        "report_generation_seconds": generation_seconds,
        "report_rendering_seconds": rendering_seconds,
    }
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
