"""Export reviewable Phase 3.4 evidence without binary checkpoints or action traces."""

import argparse
from pathlib import Path

from sirfex.training.checkpoints import write_json
from sirfex.training.persistence import TrainingStore


def compact(value):
    if isinstance(value, list):
        return [compact(v) for v in value]
    if isinstance(value, dict):
        return {
            k: Path(v).name if k == "path" and isinstance(v, str) else compact(v)
            for k, v in value.items()
            if k not in ("applied_actions", "per_step", "commands")
        }
    return value


def export(run_id, store):
    saved = store.load(run_id, allow_partial=True)
    if saved["manifest"].get("phase") != "3.4":
        raise ValueError("Expected a Phase 3.4 study")
    names = (
        "manifest",
        "config",
        "protocol",
        "diagnostics",
        "stress_benchmark",
        "difficulty_strata",
        "candidate_curricula",
        "baseline_verification",
        "development_results",
        "selected_curriculum",
        "final_protocol",
        "continuation_runs",
        "final_evaluation",
        "paired_analysis",
        "regression",
        "replay_verification",
        "summary",
    )
    return {
        "schema_version": "phase-3.4-public-evidence:1",
        "study_id": run_id,
        "status": saved["manifest"]["status"],
        "note": (
            "Checkpoint paths reduced to filenames; per-step action and witness "
            "arrays omitted. Full artifacts remain local."
        ),
        **{name: compact(saved[name]) for name in names if name in saved},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_id")
    parser.add_argument("--store", default="artifacts/training")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, export(args.study_id, TrainingStore(args.store)))
