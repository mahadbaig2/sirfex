"""Publish frozen Phase 3.3 protocol or compact completed evidence for review."""

import argparse
from pathlib import Path

from sirfex.training.checkpoints import write_json
from sirfex.training.persistence import TrainingStore


def compact(value):
    if isinstance(value, list):
        return [compact(item) for item in value]
    if isinstance(value, dict):
        return {
            key: Path(item).name if key == "path" and isinstance(item, str) else compact(item)
            for key, item in value.items()
            if key not in ("applied_actions", "per_step")
        }
    return value


def export(run_id, store, *, complete):
    saved = store.load(run_id, allow_partial=True)
    if saved["manifest"].get("phase") != "3.3":
        raise ValueError("Expected a Phase 3.3 study")
    expected_status = "completed" if complete else "prepared"
    if saved["manifest"]["status"] != expected_status:
        raise ValueError(f"Expected a {expected_status} study")
    names = (
        "manifest",
        "config",
        "protocol",
        "partitions",
        "curriculum",
        "baseline_verification",
        "known_stress_precheck",
    )
    if complete:
        names += (
            "continuation_runs",
            "evaluation",
            "paired_comparison",
            "uncertainty",
            "regression",
            "replay_verification",
            "summary",
        )
    return {
        "schema_version": "phase-3.3-public-evidence:1",
        "export_kind": "completed" if complete else "preregistered",
        "note": "Checkpoint paths reduced to filenames; per-step action arrays omitted. "
        "Original complete evidence remains in local stores.",
        "training_run_id": run_id,
        **{name: compact(saved[name]) for name in names},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--store", default="artifacts/training")
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, export(args.run_id, TrainingStore(args.store), complete=args.complete))
