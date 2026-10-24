"""Experiment execution service used by the CLI."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from datetime import UTC, datetime
from uuid import uuid4

from sirfex import __version__
from sirfex.core import EvaluationRunner, Scenario
from sirfex.experiments.factories import make_controller, make_environment
from sirfex.experiments.manifests import ExperimentManifest
from sirfex.experiments.storage import ExperimentStore


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dependencies() -> dict[str, str]:
    result = {}
    for package in ("sirfex", "numpy", "gymnasium", "box2d", "torch", "stable-baselines3"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def run_and_save_experiment(
    store: ExperimentStore,
    scenario: Scenario,
    controller_id: str,
    *,
    max_steps: int = 1_000,
    deterministic: bool = True,
    controller_config: dict[str, object] | None = None,
) -> ExperimentManifest:
    environment = make_environment(scenario.environment_id)
    controller = make_controller(controller_id, environment, controller_config)
    try:
        result = EvaluationRunner(max_steps=max_steps).run_episode(
            environment,
            controller,
            scenario,
            deterministic=deterministic,
            record_trace=True,
        )
        experiment_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:10]}"
        manifest = ExperimentManifest(
            experiment_id=experiment_id,
            created_at_utc=datetime.now(UTC).isoformat(),
            sirfex_version=__version__,
            environment_id=environment.id,
            environment_version=environment.version,
            controller_id=controller.id,
            controller_metadata=controller.metadata(),
            controller_config=controller_config or {},
            scenario=scenario.to_dict(),
            evaluation_config={
                "max_steps": max_steps,
                "deterministic": deterministic,
                "record_trace": True,
            },
            dependencies=_dependencies(),
            execution_status=result.execution_status.value,
            outcome=result.outcome,
            failures=[failure.to_dict() for failure in result.failures],
            artifact_refs={
                "scenario": "scenario.json",
                "result": "result.json",
                "trajectory": "trajectory.jsonl",
            },
            git_commit=_git_commit(),
        )
        store.save(manifest, scenario, result)
        return manifest
    finally:
        environment.close()


def load_scenario_file(path: str) -> Scenario:
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    return Scenario.from_dict(value)
