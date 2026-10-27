"""Publish compact Phase 3.5 evidence without local checkpoint paths or traces."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from sirfex.training.checkpoints import write_json
from sirfex.training.persistence import TrainingStore
from sirfex.training.protocol import content_hash


def _evaluation(value):
    keys = (
        "scenarios",
        "episodes",
        "constraint_violation_count",
        "constraint_violation_frequency",
        "mean_tracking_error",
        "maximum_violation_magnitude",
        "mean_violation_magnitude",
        "execution_failures",
        "mean_episode_length",
        "full_horizon_frequency",
        "action_statistics",
        "checkpoint_sha256",
    )
    result = {key: value.get(key) for key in keys if key in value}
    by_stratum = {}
    for row in value.get("episode_results", []):
        label = row.get("scenario", {}).get("provenance", {}).get("difficulty_stratum")
        if label:
            by_stratum.setdefault(label, []).append(row)
    if by_stratum:
        result["difficulty_strata"] = {
            label: {
                "scenarios": len(rows),
                "mean_tracking_error": statistics.mean(row["mean_tracking_error"] for row in rows),
                "violation_frequency": statistics.mean(
                    bool(row["constraint_violation"]) for row in rows
                ),
            }
            for label, rows in sorted(by_stratum.items())
        }
    return result


def _run(row):
    keys = (
        "seed",
        "candidate_id",
        "target_stress_interaction_fraction",
        "starting_sha256",
        "starting_fingerprint",
        "continuation_seed",
        "requested_additional_steps",
        "actual_additional_steps",
        "training_seconds",
        "exposure",
        "adequacy",
    )
    result = {key: row.get(key) for key in keys if key in row}
    result["checkpoint"] = {
        key: row["checkpoint"][key] for key in ("checkpoint_id", "sha256", "training_timesteps")
    }
    for key in ("development_nominal", "development_stress"):
        if key in row:
            result[key] = _evaluation(row[key])
    return result


def export(study_id, store, path):
    saved = store.load(study_id, allow_partial=True)
    if saved["manifest"].get("phase") != "3.5":
        raise ValueError("Expected Phase 3.5 study")
    if content_hash(saved["protocol"]) != saved["manifest"]["protocol_sha256"]:
        raise ValueError("Frozen protocol hash mismatch")
    result = {
        "schema_version": "phase-3.5-public-evidence:1",
        "study_id": study_id,
        "status": saved["manifest"]["status"],
        "protocol_sha256": saved["manifest"]["protocol_sha256"],
        "source_checkpoint_sha256": saved["protocol"]["baseline_sha256"],
        "partition_hashes": {
            key: saved["protocol"][key]
            for key in (
                "training_hash",
                "development_nominal_hash",
                "development_stress_hash",
                "final_hashes",
            )
        },
        "discovery_pool_hash": saved["protocol"]["discovery_pool_hash"],
        "exposure_audit": saved["exposure_audit"],
        "calibration": saved["calibration"],
        "screening_runs": [_run(row) for row in saved.get("screening_runs", {}).get("records", [])],
        "selection": saved.get("selected_curriculum"),
        "confirmation_runs": [
            _run(row) for row in saved.get("confirmation_runs", {}).get("records", [])
        ],
        "final_protocol_sha256": saved["manifest"].get("final_protocol_sha256"),
        "paired_analysis": saved.get("paired_analysis"),
        "replay_verification": saved.get("replay_verification"),
        "resource_memory": saved["manifest"].get("memory"),
        "note": "Full checkpoint binaries, scenario traces and local paths are not published.",
    }
    if "final_evaluation" in saved:
        final = saved["final_evaluation"]
        result["final_evaluation"] = {
            "set_hashes": final["set_hashes"],
            "seeds": {
                seed: {
                    arm: {name: _evaluation(summary) for name, summary in values.items()}
                    for arm, values in arms.items()
                }
                for seed, arms in final["seeds"].items()
            },
            "references": {
                name: {set_name: _evaluation(summary) for set_name, summary in sets.items()}
                for name, sets in final["references"].items()
            },
        }
    write_json(Path(path), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study_id")
    parser.add_argument("--store", default="artifacts/training")
    parser.add_argument("--output", default="docs/evidence/phase_3_5_results.json")
    args = parser.parse_args()
    data = export(args.study_id, TrainingStore(args.store), args.output)
    print(data["status"])
