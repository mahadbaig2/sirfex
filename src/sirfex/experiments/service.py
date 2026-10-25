"""Experiment execution service used by the CLI."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from datetime import UTC, datetime
from uuid import uuid4

from sirfex import __version__
from sirfex.core import (
    EpisodeResult,
    EvaluationError,
    EvaluationRunner,
    FailureCategory,
    FailureRecord,
    Scenario,
)
from sirfex.core.failures import ExecutionStatus
from sirfex.experiments.factories import make_controller, make_environment
from sirfex.experiments.manifests import ExperimentManifest
from sirfex.experiments.storage import ExperimentStore


class ExecutionPersistenceError(RuntimeError):
    """Both the execution error and persistence error remain available to callers."""

    def __init__(self, execution_error: BaseException, persistence_error: BaseException) -> None:
        self.execution_error = execution_error
        self.persistence_error = persistence_error
        super().__init__(
            f"Execution failed ({execution_error}); persistence also failed ({persistence_error})"
        )


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
    effective_config = dict(controller_config or {})
    if controller_id == "pump-analytical":
        dt = float(scenario.parameters.get("dt", 0.1))
        if "dt" in effective_config and effective_config["dt"] != dt:
            raise ValueError("Analytical controller dt must match the scenario")
        effective_config["dt"] = dt
    if controller_id == "random":
        effective_config.setdefault("seed", scenario.seed)
    experiment_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:10]}"
    environment = make_environment(scenario.environment_id)
    controller = None
    result = None
    try:
        controller = make_controller(controller_id, environment, effective_config)
        result = EvaluationRunner(max_steps=max_steps).run_episode(
            environment,
            controller,
            scenario,
            deterministic=deterministic,
            record_trace=True,
        )
        manifest = ExperimentManifest(
            experiment_id=experiment_id,
            created_at_utc=datetime.now(UTC).isoformat(),
            sirfex_version=__version__,
            environment_id=environment.id,
            environment_version=environment.version,
            controller_id=controller.id,
            controller_metadata=controller.metadata(),
            controller_config=effective_config,
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
    except BaseException as execution_error:
        if isinstance(execution_error, EvaluationError):
            record = execution_error.record
            partial_trace = execution_error.partial_trace
            execution_error.experiment_id = experiment_id
        else:
            record = FailureRecord(
                FailureCategory.ENVIRONMENT_EXCEPTION,
                f"Unhandled execution error: {type(execution_error).__name__}: {execution_error}",
                evidence_source="service",
            )
            partial_trace = []
        failed_result = EpisodeResult(
            environment_id=environment.id,
            controller_id=controller.id if controller is not None else controller_id,
            scenario=scenario.to_dict(),
            seed=scenario.seed,
            episode_reward=0.0,
            episode_length=len(partial_trace),
            terminated=False,
            truncated=False,
            outcome="execution_failed",
            metrics={"execution_error_type": type(execution_error).__name__},
            trace=partial_trace,
            execution_status=ExecutionStatus.FAILED,
            observed_outcome="execution_error",
            failures=[record],
        )
        failed_manifest = ExperimentManifest(
            experiment_id=experiment_id,
            created_at_utc=datetime.now(UTC).isoformat(),
            sirfex_version=__version__,
            environment_id=environment.id,
            environment_version=environment.version,
            controller_id=failed_result.controller_id,
            controller_metadata=controller.metadata() if controller is not None else {},
            controller_config=effective_config,
            scenario=scenario.to_dict(),
            evaluation_config={
                "max_steps": max_steps,
                "deterministic": deterministic,
                "record_trace": True,
            },
            dependencies=_dependencies(),
            execution_status=ExecutionStatus.FAILED.value,
            outcome=failed_result.outcome,
            failures=[record.to_dict()],
            artifact_refs={
                "scenario": "scenario.json",
                "result": "result.json",
                "trajectory": "trajectory.jsonl",
            },
            git_commit=_git_commit(),
        )
        try:
            store.save(failed_manifest, scenario, failed_result)
        except BaseException as persistence_error:
            raise ExecutionPersistenceError(execution_error, persistence_error) from execution_error
        raise
    finally:
        environment.close()


def load_scenario_file(path: str) -> Scenario:
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    return Scenario.from_dict(value)
