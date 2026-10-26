"""Persist validation-policy traces through the existing experiment/replay services.

This diagnostic never evaluates final holdouts or changes a training run's protocol.
"""

import argparse
import json
from pathlib import Path

from sirfex.core import Scenario
from sirfex.experiments.replay import replay_experiment
from sirfex.experiments.service import run_and_save_experiment
from sirfex.experiments.storage import ExperimentStore
from sirfex.training.checkpoints import write_json
from sirfex.training.persistence import TrainingStore


def inspect(run_id, store, experiments, *, selected_only=False):
    saved = store.load(run_id)
    scenarios = [
        Scenario.from_dict(row["scenario"])
        for row in saved["dataset_partitions"]["groups"]["validation"]
    ]
    rows = []
    selected = {
        meta["sha256"] for meta in saved.get("baseline_selection", {}).get("selected", {}).values()
    }
    for point in saved["learning_curve"]["points"]:
        if selected_only and point["checkpoint"]["sha256"] not in selected:
            continue
        actions, ids = [], []
        for scenario in scenarios:
            manifest = run_and_save_experiment(
                experiments,
                scenario,
                "ppo",
                max_steps=200,
                controller_config={"checkpoint": point["checkpoint"]},
            )
            ids.append(manifest.experiment_id)
            stored = experiments.load(manifest.experiment_id)
            actions.extend(row["action"][0] for row in stored.trajectory)
        rows.append(
            {
                "checkpoint_id": point["checkpoint"]["checkpoint_id"],
                "sha256": point["checkpoint"]["sha256"],
                "experiment_ids": ids,
                "action_minimum": min(actions),
                "action_maximum": max(actions),
                "zero_action_fraction": sum(a == 0 for a in actions) / len(actions),
                "action_count": len(actions),
                "representative_replay": replay_experiment(experiments, ids[0]).to_dict(),
            }
        )
    result = {
        "schema_version": "validation-policy-diagnostics:1",
        "training_run_id": run_id,
        "partition": "validation",
        "checkpoints": rows,
        "note": "Validation traces inspected; selection and holdouts unchanged",
    }
    if not rows:
        raise ValueError("No eligible validation checkpoints")
    filename = (
        "policy_trace_diagnostics.json"
        if saved["manifest"].get("phase") == "3.2"
        else "policy_diagnostics.json"
    )
    write_json(store.directory(run_id) / filename, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--store", default="artifacts/training")
    parser.add_argument("--experiment-store", default="artifacts/experiments")
    parser.add_argument("--selected-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            inspect(
                args.run_id,
                TrainingStore(Path(args.store)),
                ExperimentStore(args.experiment_store),
                selected_only=args.selected_only,
            ),
            indent=2,
        )
    )
