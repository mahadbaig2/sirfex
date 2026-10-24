"""Atomic, file-based local experiment storage."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sirfex.core.interfaces import Scenario
from sirfex.core.results import EpisodeResult
from sirfex.experiments.manifests import ExperimentManifest

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{5,63}$")


class StorageError(RuntimeError):
    """Raised for malformed, incomplete, unsafe, or conflicting artifacts."""


@dataclass
class StoredExperiment:
    manifest: ExperimentManifest
    scenario: Scenario
    result: dict[str, Any]
    trajectory: list[dict[str, Any]]


class ExperimentStore:
    def __init__(self, root: Path | str = Path("artifacts/experiments")) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _experiment_dir(self, experiment_id: str) -> Path:
        if not _ID_PATTERN.fullmatch(experiment_id):
            raise StorageError("Invalid experiment ID")
        path = (self.root / experiment_id).resolve()
        if path.parent != self.root:
            raise StorageError("Unsafe experiment path")
        return path

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            Path(temporary_name).replace(path)
        finally:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()

    def save(
        self,
        manifest: ExperimentManifest,
        scenario: Scenario,
        result: EpisodeResult,
    ) -> Path:
        directory = self._experiment_dir(manifest.experiment_id)
        if directory.exists():
            raise StorageError(f"Experiment already exists: {manifest.experiment_id}")
        directory.mkdir(parents=True)
        marker = directory / ".incomplete"
        marker.write_text("incomplete", encoding="utf-8")
        try:
            self._atomic_write(
                directory / "scenario.json", json.dumps(scenario.to_dict(), indent=2)
            )
            result_value = result.to_dict()
            trajectory = result_value.pop("trace", [])
            self._atomic_write(
                directory / "result.json", json.dumps(result_value, indent=2, allow_nan=False)
            )
            trajectory_text = "".join(
                json.dumps(step, allow_nan=False, separators=(",", ":")) + "\n"
                for step in trajectory
            )
            self._atomic_write(directory / "trajectory.jsonl", trajectory_text)
            self._atomic_write(
                directory / "manifest.json",
                json.dumps(manifest.to_dict(), indent=2, allow_nan=False),
            )
            marker.unlink()
        except Exception:
            raise
        return directory

    def load(self, experiment_id: str) -> StoredExperiment:
        directory = self._experiment_dir(experiment_id)
        if not directory.is_dir():
            raise StorageError(f"Experiment not found: {experiment_id}")
        if (directory / ".incomplete").exists():
            raise StorageError(f"Experiment is incomplete: {experiment_id}")
        required = ["manifest.json", "scenario.json", "result.json", "trajectory.jsonl"]
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise StorageError(f"Experiment artifacts missing: {', '.join(missing)}")
        try:
            manifest = ExperimentManifest.from_dict(
                json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            )
            scenario = Scenario.from_dict(
                json.loads((directory / "scenario.json").read_text(encoding="utf-8"))
            )
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            trajectory = [
                json.loads(line)
                for line in (directory / "trajectory.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise StorageError(f"Malformed experiment artifacts: {experiment_id}") from exc
        if manifest.experiment_id != experiment_id:
            raise StorageError("Manifest experiment ID does not match its directory")
        result["trace"] = trajectory
        return StoredExperiment(manifest, scenario, result, trajectory)

    def list(
        self, *, environment_id: str | None = None, controller_id: str | None = None
    ) -> list[ExperimentManifest]:
        manifests = []
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir() or not _ID_PATTERN.fullmatch(directory.name):
                continue
            try:
                manifest = self.load(directory.name).manifest
            except StorageError:
                continue
            if environment_id and manifest.environment_id != environment_id:
                continue
            if controller_id and manifest.controller_id != controller_id:
                continue
            manifests.append(manifest)
        return manifests

    def delete(self, experiment_id: str) -> None:
        directory = self._experiment_dir(experiment_id)
        if not directory.is_dir():
            raise StorageError(f"Experiment not found: {experiment_id}")
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()
        directory.rmdir()
