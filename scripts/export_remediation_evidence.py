"""Export compact review evidence; complete trajectories/checkpoints stay in local stores."""

import argparse
import json
from pathlib import Path

from sirfex.search.persistence import SearchStore
from sirfex.training.checkpoints import write_json
from sirfex.training.persistence import TrainingStore


def compact(value):
    if isinstance(value, list):
        return [compact(item) for item in value]
    if isinstance(value, dict):
        return {
            key: Path(item).name if key == "path" and isinstance(item, str) else compact(item)
            for key, item in value.items()
            if key not in ("per_step", "applied_actions")
        }
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--search")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = TrainingStore()
    runs = {}
    for run_id in args.run:
        saved = training.load(run_id)
        if "policy_diagnostics" not in saved:
            saved["policy_diagnostics"] = {
                "records": [
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in sorted(training.directory(run_id).glob("policy_[0-9]*.json"))
                ]
            }
        runs[run_id] = compact(saved)
    write_json(
        args.output,
        {
            "schema_version": "phase-3.2-review-evidence:1",
            "note": "Per-step rewards/actions omitted; checkpoint paths reduced to filenames. "
            "Original complete evidence remains in local artifact stores.",
            "training_runs": runs,
            "search": compact(SearchStore().load(args.search)) if args.search else None,
        },
    )
