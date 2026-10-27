"""Versioned local persistence for Phase 3 training runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sirfex.training.checkpoints import write_json

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{5,80}$")


class TrainingStorageError(RuntimeError):
    pass


class TrainingStore:
    def __init__(self, root: Path | str = "artifacts/training") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, run_id: str) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise TrainingStorageError("Invalid training run ID")
        path = (self.root / run_id).resolve()
        if path.parent != self.root:
            raise TrainingStorageError("Unsafe training run path")
        return path

    def save(self, run_id: str, files: dict[str, dict[str, Any]]) -> Path:
        directory = self.directory(run_id)
        if any(Path(name).name != name or not name.endswith(".json") for name in files):
            raise TrainingStorageError("Unsafe training artifact filename")
        marker = directory / ".incomplete"
        if directory.exists():
            if (directory / "manifest.json").exists() or not marker.exists():
                raise TrainingStorageError(f"Training run already exists: {run_id}")
        else:
            directory.mkdir(parents=True)
            marker.write_text("incomplete", encoding="utf-8")
        try:
            for name, value in files.items():
                write_json(directory / name, value)
            marker.unlink()
        except Exception:
            raise
        return directory

    def load(self, run_id: str, *, allow_partial: bool = False) -> dict[str, Any]:
        directory = self.directory(run_id)
        if not directory.is_dir():
            raise TrainingStorageError(f"Training run not found: {run_id}")
        if (directory / ".incomplete").exists() and not allow_partial:
            raise TrainingStorageError(f"Training run is incomplete: {run_id}")
        result = {}
        for name in (
            "manifest.json",
            "config.json",
            "curriculum.json",
            "evaluation.json",
            "regression.json",
            "summary.json",
            "protocol.json",
            "feasibility_report.json",
            "reward_diagnostics.json",
            "learning_curve.json",
            "dataset_partitions.json",
            "baseline_selection.json",
            "continuation_results.json",
            "uncertainty.json",
            "policy_diagnostics.json",
            "policy_trace_diagnostics.json",
            "selected_policy_diagnostics.json",
            "partitions.json",
            "baseline_verification.json",
            "known_stress_precheck.json",
            "continuation_runs.json",
            "paired_comparison.json",
            "replay_verification.json",
            "diagnostics.json",
            "stress_benchmark.json",
            "difficulty_strata.json",
            "candidate_curricula.json",
            "development_results.json",
            "selected_curriculum.json",
            "final_protocol.json",
            "final_evaluation.json",
            "paired_analysis.json",
        ):
            path = directory / name
            if path.resolve().parent != directory:
                raise TrainingStorageError("Unsafe training artifact path")
            if path.is_file():
                try:
                    result[name.removesuffix(".json")] = json.loads(
                        path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise TrainingStorageError(f"Malformed training artifact: {name}") from exc
        if "manifest" not in result:
            raise TrainingStorageError("Training manifest is missing")
        if result["manifest"].get("schema_version") not in ("1", "2"):
            raise TrainingStorageError("Unsupported training manifest schema")
        if result["manifest"].get("training_run_id") != run_id:
            raise TrainingStorageError("Training manifest identity mismatch")
        return result

    def list(self) -> list[dict[str, Any]]:
        values = []
        for directory in sorted(self.root.iterdir()):
            if directory.is_dir() and _RUN_ID.fullmatch(directory.name):
                try:
                    values.append(self.load(directory.name, allow_partial=True)["manifest"])
                except TrainingStorageError:
                    continue
        return values
