"""Experiment replay and evidence comparison."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from sirfex import __version__
from sirfex.core import EvaluationRunner, Scenario
from sirfex.experiments.factories import make_controller, make_environment
from sirfex.experiments.storage import ExperimentStore, StorageError


@dataclass
class ReplayReport:
    status: str
    experiment_id: str
    explanation: str
    mismatches: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compare_value(
    left: Any, right: Any, path: str, tolerance: float, mismatches: list[str]
) -> None:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance):
            mismatches.append(f"{path}: {left!r} != {right!r}")
        return
    if type(left) is not type(right):
        mismatches.append(f"{path}: types {type(left).__name__} != {type(right).__name__}")
        return
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                mismatches.append(f"{path}.{key}: field missing")
            else:
                _compare_value(left[key], right[key], f"{path}.{key}", tolerance, mismatches)
        return
    if isinstance(left, list):
        if len(left) != len(right):
            mismatches.append(f"{path}: lengths {len(left)} != {len(right)}")
            return
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _compare_value(left_item, right_item, f"{path}[{index}]", tolerance, mismatches)
        return
    if left != right:
        mismatches.append(f"{path}: {left!r} != {right!r}")


def compare_results(
    original: dict[str, Any], replayed: dict[str, Any], tolerance: float = 1e-9
) -> ReplayReport:
    mismatches: list[str] = []
    fields = (
        "environment_id",
        "controller_id",
        "seed",
        "episode_reward",
        "episode_length",
        "terminated",
        "truncated",
        "outcome",
        "observed_outcome",
        "failures",
        "trace",
    )
    for field_name in fields:
        _compare_value(
            original.get(field_name), replayed.get(field_name), field_name, tolerance, mismatches
        )
    if not mismatches:
        return ReplayReport("exact_match", "", "All recorded fields matched exactly.")
    tolerance_mismatches: list[str] = []
    for field_name in fields:
        _compare_value(
            original.get(field_name),
            replayed.get(field_name),
            field_name,
            1e-5,
            tolerance_mismatches,
        )
    status = "tolerance_match" if not tolerance_mismatches else "mismatch"
    explanation = (
        "Differences were within numerical tolerance."
        if status == "tolerance_match"
        else "Recorded evidence differed."
    )
    return ReplayReport(status, "", explanation, mismatches[:50])


def replay_experiment(
    store: ExperimentStore, experiment_id: str, tolerance: float = 1e-9
) -> ReplayReport:
    try:
        stored = store.load(experiment_id)
        environment = make_environment(stored.manifest.environment_id)
        controller = make_controller(
            stored.manifest.controller_id,
            environment,
            stored.manifest.controller_config,
        )
    except (StorageError, ValueError, ImportError, OSError, KeyError, TypeError) as exc:
        return ReplayReport("unavailable", experiment_id, str(exc))
    try:
        scenario = Scenario.from_dict(stored.manifest.scenario)
        result = EvaluationRunner(
            max_steps=int(stored.manifest.evaluation_config.get("max_steps", 1_000))
        ).run_episode(
            environment,
            controller,
            scenario,
            deterministic=bool(stored.manifest.evaluation_config.get("deterministic", True)),
            record_trace=True,
        )
        report = compare_results(stored.result, result.to_dict(), tolerance)
        report.experiment_id = experiment_id
        if stored.manifest.sirfex_version != __version__:
            report.explanation += " Manifest was created by a different SIRFEX version."
        return report
    except Exception as exc:
        return ReplayReport("unavailable", experiment_id, f"Replay execution failed: {exc}")
    finally:
        environment.close()
