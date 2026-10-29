"""Run a bounded synthetic Checkpoint 2A pilot; never load the scientific cohort."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import tracemalloc
from pathlib import Path

import sirfex.robustness.evidence as evidence_module
from sirfex.robustness.study import build_plan
from sirfex.robustness.study_execute import StudyEngine
from tests.integration.test_robustness_study_execution import synthetic_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="docs/evidence/phase_3_7_study_pilot_2a1.json",
        help="Repository-relative output for this bounded synthetic pilot",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol, manifest = synthetic_package()
    plan = build_plan(protocol, manifest, root, synthetic=True)
    durations = {
        "grid_preparation": 0.0,
        "evidence_archival_including_replay": 0.0,
        "independent_replay": 0.0,
    }
    counts = {"independent_replay": 0}
    started = time.monotonic()
    tracemalloc.start()
    with tempfile.TemporaryDirectory(prefix="sirfex_synthetic_pilot_") as directory:
        engine = StudyEngine(protocol, manifest, plan, root, directory, synthetic=True)
        original_prepare = engine.prepare
        original_archive = engine._archive_pair
        original_replay = evidence_module.replay_experiment

        def timed_prepare():
            before = time.monotonic()
            try:
                return original_prepare()
            finally:
                durations["grid_preparation"] += time.monotonic() - before

        def timed_archive(pair):
            before = time.monotonic()
            try:
                return original_archive(pair)
            finally:
                durations["evidence_archival_including_replay"] += time.monotonic() - before

        def timed_replay(*args, **kwargs):
            before = time.monotonic()
            try:
                return original_replay(*args, **kwargs)
            finally:
                durations["independent_replay"] += time.monotonic() - before
                counts["independent_replay"] += 1

        engine.prepare = timed_prepare
        engine._archive_pair = timed_archive
        evidence_module.replay_experiment = timed_replay
        try:
            fixed_start = time.monotonic()
            engine.execute_fixed_and_grid()
            durations["fixed_and_grid_total"] = time.monotonic() - fixed_start
            differential_start = time.monotonic()
            engine.execute_differential()
            durations["differential_total"] = time.monotonic() - differential_start
            report_start = time.monotonic()
            report = engine.execute_report()
            durations["report"] = time.monotonic() - report_start
            status = engine.store.status()
            archive_item = next(i for i in plan["work_items"] if i["stage"] == "archive")
            archive = engine.store.read(archive_item["work_id"])["result"]["result"]
            _, peak_python_bytes = tracemalloc.get_traced_memory()
            result = {
                "schema_version": "phase37-synthetic-pilot:1",
                "scope": "synthetic_test_only",
                "protocol_sha256": protocol["semantic_sha256"],
                "plan_sha256": plan["semantic_sha256"],
                "report_sha256": report["semantic_sha256"],
                "report_status": report["execution_status"],
                "work_counts": plan["work_counts"],
                "measured_seconds": {key: round(value, 6) for key, value in durations.items()},
                "controller_evaluation_seconds": status["resource_usage"]["controller_seconds"],
                "independent_replay_count": counts["independent_replay"],
                "selected_class_b_count": len(archive["findings"]["class_b"]),
                "artifact_bytes": status["artifact_bytes"],
                "peak_python_allocated_bytes": peak_python_bytes,
                "total_seconds": round(time.monotonic() - started, 6),
                "limitations": (
                    "Synthetic classical controllers only; Python heap peak excludes native "
                    "allocations and is not a scientific runtime estimate."
                ),
            }
            if result["report_status"] != "complete" or result["independent_replay_count"] < 4:
                raise RuntimeError("Synthetic pilot did not complete the full evidence path")
        finally:
            evidence_module.replay_experiment = original_replay
            engine.close()
            tracemalloc.stop()
    destination = (root / args.output).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("Synthetic pilot output must stay within the repository")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
