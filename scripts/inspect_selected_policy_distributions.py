"""Inspect fixed validation observations for each preselected competent policy."""

import argparse
import json

from sirfex.core import Scenario
from sirfex.training.checkpoints import write_json
from sirfex.training.persistence import TrainingStore
from sirfex.training.policy_diagnostics import diagnose_policy
from sirfex.training.remediation_diagnostics import metadata_from_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--store", default="artifacts/training")
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    store = TrainingStore(args.store)
    saved = store.load(args.run_id)
    cases = [
        Scenario.from_dict(row["scenario"])
        for row in saved["dataset_partitions"]["groups"]["validation"]
    ]
    records = [
        diagnose_policy(metadata_from_dict(meta), cases, seed=args.seed, samples=args.samples)
        for meta in saved["baseline_selection"]["selected"].values()
    ]
    report = {
        "schema_version": "selected-ppo-actions:1",
        "training_run_id": args.run_id,
        "partition": "validation",
        "sampling_seed": args.seed,
        "samples_per_observation": args.samples,
        "records": records,
    }
    write_json(store.directory(args.run_id) / "selected_policy_diagnostics.json", report)
    print(
        json.dumps(
            {
                "training_run_id": args.run_id,
                "checkpoints": [r["checkpoint"]["checkpoint_id"] for r in records],
                "output": "selected_policy_diagnostics.json",
            },
            indent=2,
        )
    )
