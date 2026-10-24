"""Export reviewable Phase 3.1 evidence without machine-specific checkpoint paths."""

import argparse
from pathlib import Path, PureWindowsPath

from sirfex.training.checkpoints import write_json
from sirfex.training.persistence import TrainingStore


def portable(value):
    if isinstance(value, dict):
        return {
            key: PureWindowsPath(item).name if key == "path" else portable(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [portable(item) for item in value]
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--store", default="artifacts/training")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = TrainingStore(args.store).load(args.run_id, allow_partial=True)
    write_json(
        args.output,
        {
            "export_schema": "validation-review:1",
            "note": "Review export; checkpoint filenames are references, not portable model files",
            **portable(saved),
        },
    )
