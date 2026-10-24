"""Reproducible Phase 3 baseline and matched-continuation orchestration."""

from __future__ import annotations

import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sirfex import __version__
from sirfex.controllers import PumpRuleController
from sirfex.core.interfaces import Scenario
from sirfex.search.persistence import SearchStore
from sirfex.training.checkpoints import CheckpointMetadata, sha256_file, write_json
from sirfex.training.config import TrainingConfig
from sirfex.training.controller import PPOController
from sirfex.training.curricula import ScenarioCurriculum, select_discoveries
from sirfex.training.evaluation import evaluate_controller
from sirfex.training.persistence import TrainingStore
from sirfex.training.ppo import dependency_versions, train_ppo
from sirfex.training.pump_gym import PumpGymEnv
from sirfex.training.regression import classify_regression


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def normal_scenarios(seeds: list[int]) -> list[Scenario]:
    values = [
        {
            "dt": 0.1,
            "horizon": 80,
            "target_level": 0.5,
            "initial_level": 0.5,
            "demand": 0.45,
            "pump_capacity": 0.8,
            "efficiency": 0.95,
            "leak": 0.01,
        },
        {
            "dt": 0.1,
            "horizon": 80,
            "target_level": 0.55,
            "initial_level": 0.5,
            "demand": 0.5,
            "pump_capacity": 0.8,
            "efficiency": 0.95,
            "leak": 0.01,
        },
        {
            "dt": 0.1,
            "horizon": 80,
            "target_level": 0.45,
            "initial_level": 0.55,
            "demand": 0.4,
            "pump_capacity": 0.8,
            "efficiency": 1.0,
            "leak": 0.0,
        },
    ]
    return [
        Scenario(
            "pump",
            parameters=value,
            seed=seed,
            description="Phase 3 normal operating scenario",
            tags=("phase-3", "normal"),
        )
        for seed in seeds
        for value in values
    ]


def held_out_stress_scenarios() -> list[Scenario]:
    values = [
        {
            "dt": 0.1,
            "horizon": 100,
            "target_level": 0.5,
            "initial_level": 0.5,
            "demand": 0.9,
            "pump_capacity": 0.8,
            "efficiency": 0.55,
            "leak": 0.04,
        },
        {
            "dt": 0.1,
            "horizon": 100,
            "target_level": 0.6,
            "initial_level": 0.55,
            "demand": 0.82,
            "pump_capacity": 0.8,
            "efficiency": 0.6,
            "leak": 0.03,
        },
    ]
    return [
        Scenario(
            "pump",
            parameters=value,
            seed=700 + index,
            description="Phase 3 held-out stress scenario",
            tags=("phase-3", "held-out"),
        )
        for index, value in enumerate(values)
    ]


def _checkpoint(
    model: Any, directory: Path, run_id: str, name: str, config: TrainingConfig
) -> CheckpointMetadata:
    path_without_suffix = directory / "checkpoints" / name
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path_without_suffix))
    path = path_without_suffix.with_suffix(".zip")
    metadata = CheckpointMetadata(
        checkpoint_id=f"{run_id}:{name}",
        training_run_id=run_id,
        path=str(path.resolve()),
        sha256=sha256_file(path),
        environment_id="pump",
        observation_shape=(3,),
        action_shape=(1,),
        ppo_config=config.ppo.to_dict(),
        training_seed=config.training_seed,
        training_timesteps=config.training_timesteps,
        dependencies=dependency_versions(),
    )
    write_json(directory / "checkpoints" / f"{name}.metadata.json", metadata.to_dict())
    return metadata


def _ppo_factory(metadata: CheckpointMetadata):
    return lambda: PPOController.load(metadata.path, metadata)


def run_baseline(config: TrainingConfig, store: TrainingStore) -> dict[str, Any]:
    config.validate()
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:10]}"
    directory = store.directory(run_id)
    directory.mkdir(parents=True)
    (directory / ".incomplete").write_text("incomplete", encoding="utf-8")
    started = time.perf_counter()
    scenarios = list(config.normal_scenarios) or normal_scenarios(
        [config.training_seed, config.training_seed + 1]
    )
    environment = PumpGymEnv(scenarios, seed=config.training_seed)
    try:
        model = train_ppo(
            environment, config.ppo, seed=config.training_seed, timesteps=config.training_timesteps
        )
        metadata = _checkpoint(model, directory, run_id, "baseline", config)
    finally:
        environment.close()
    validation = scenarios[: min(4, len(scenarios))]
    ppo_summary = evaluate_controller(
        _ppo_factory(metadata), validation, "nominal", max_steps=config.max_episode_steps
    )
    rule_summary = evaluate_controller(
        lambda: PumpRuleController(), validation, "nominal", max_steps=config.max_episode_steps
    )
    manifest = {
        "schema_version": "1",
        "training_run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed",
        "sirfex_version": __version__,
        "git_commit": _git_commit(),
        "environment_id": "pump",
        "environment_version": "sirfex-pump:1",
        "controller_architecture": "MlpPolicy [64, 64]",
        "training_seed": config.training_seed,
        "training_timesteps": config.training_timesteps,
        "checkpoint": metadata.to_dict(),
        "execution_status": "completed",
        "wall_clock_seconds": time.perf_counter() - started,
    }
    files = {
        "manifest.json": manifest,
        "config.json": config.to_dict(),
        "curriculum.json": {
            "normal_scenarios": [item.to_dict() for item in scenarios],
            "stress_scenarios": [],
            "ratio": 0.0,
        },
        "evaluation.json": {"ppo": ppo_summary.to_dict(), "rule_based": rule_summary.to_dict()},
        "regression.json": {
            "classification": "inconclusive",
            "reason": "Baseline run has no matched continuation yet.",
        },
        "summary.json": {
            "training_run_id": run_id,
            "best_available_controller": "ppo",
            "training_seconds": manifest["wall_clock_seconds"],
        },
    }
    store.save(run_id, files)
    return {"training_run_id": run_id, "manifest": manifest, "evaluation": files["evaluation.json"]}


def run_improvement(
    config: TrainingConfig,
    store: TrainingStore,
    baseline_run_id: str,
    search_id: str,
    search_store: SearchStore,
) -> dict[str, Any]:
    config.validate()
    baseline = store.load(baseline_run_id)
    baseline_meta = CheckpointMetadata(
        **baseline["manifest"]["checkpoint"]
        | {
            "observation_shape": tuple(baseline["manifest"]["checkpoint"]["observation_shape"]),
            "action_shape": tuple(baseline["manifest"]["checkpoint"]["action_shape"]),
        }
    )
    selected = select_discoveries(search_store, search_id, config.discovery_limit)
    normal = list(config.normal_scenarios) or normal_scenarios(
        [config.training_seed + 10, config.training_seed + 11]
    )
    stress = [Scenario.from_dict(item.scenario) for item in selected]
    curriculum = ScenarioCurriculum(normal, stress, config.curriculum_ratio, config.training_seed)
    robust_training = curriculum.scenarios(max(8, len(normal) + len(stress)))
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:10]}"
    directory = store.directory(run_id)
    directory.mkdir(parents=True)
    (directory / ".incomplete").write_text("incomplete", encoding="utf-8")

    def train_continuation(name: str, scenarios: list[Scenario]) -> CheckpointMetadata:
        environment = PumpGymEnv(scenarios, seed=config.training_seed)
        try:
            model = train_ppo(
                environment,
                config.ppo,
                seed=config.training_seed,
                timesteps=config.training_timesteps,
                checkpoint=baseline_meta.path,
            )
            return _checkpoint(model, directory, run_id, name, config)
        finally:
            environment.close()

    normal_meta = train_continuation("normal_continuation", normal)
    robust_meta = train_continuation("robust_continuation", robust_training)
    known = stress or held_out_stress_scenarios()[:1]
    held_out = held_out_stress_scenarios()
    evaluations: dict[str, Any] = {}
    for label, scenarios in (
        ("nominal", normal[:4]),
        ("known_stress", known),
        ("held_out_stress", held_out),
    ):
        evaluations[f"normal_continuation_{label}"] = evaluate_controller(
            _ppo_factory(normal_meta), scenarios, label, max_steps=config.max_episode_steps
        ).to_dict()
        evaluations[f"robust_continuation_{label}"] = evaluate_controller(
            _ppo_factory(robust_meta), scenarios, label, max_steps=config.max_episode_steps
        ).to_dict()
        evaluations[f"baseline_{label}"] = evaluate_controller(
            _ppo_factory(baseline_meta), scenarios, label, max_steps=config.max_episode_steps
        ).to_dict()
        evaluations[f"rule_based_{label}"] = evaluate_controller(
            lambda: PumpRuleController(), scenarios, label, max_steps=config.max_episode_steps
        ).to_dict()
    normal_control = evaluations["normal_continuation_nominal"]
    robust = evaluations["robust_continuation_nominal"]
    regression = classify_regression(
        normal_control,
        robust,
        evaluations["normal_continuation_held_out_stress"],
        evaluations["robust_continuation_held_out_stress"],
        evaluations["normal_continuation_known_stress"],
        evaluations["robust_continuation_known_stress"],
        config.regression_tolerances,
    )
    manifest = {
        "schema_version": "1",
        "training_run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed",
        "sirfex_version": __version__,
        "git_commit": _git_commit(),
        "environment_id": "pump",
        "environment_version": "sirfex-pump:1",
        "baseline_training_run_id": baseline_run_id,
        "source_search_id": search_id,
        "training_seed": config.training_seed,
        "training_timesteps": config.training_timesteps,
        "checkpoints": {
            "baseline": baseline_meta.to_dict(),
            "normal_continuation": normal_meta.to_dict(),
            "robust_continuation": robust_meta.to_dict(),
        },
        "selected_discovery_ids": [item.discovery_id for item in selected],
        "selected_experiment_ids": [
            experiment_id for item in selected for experiment_id in item.experiment_ids
        ],
        "execution_status": "completed",
    }
    files = {
        "manifest.json": manifest,
        "config.json": config.to_dict(),
        "curriculum.json": curriculum.to_dict(),
        "evaluation.json": evaluations,
        "regression.json": regression,
        "summary.json": {
            "training_run_id": run_id,
            "classification": regression["classification"],
            "known_stress_scenarios": len(known),
            "held_out_stress_scenarios": len(held_out),
        },
    }
    store.save(run_id, files)
    return {
        "training_run_id": run_id,
        "manifest": manifest,
        "regression": regression,
        "evaluation": evaluations,
    }
