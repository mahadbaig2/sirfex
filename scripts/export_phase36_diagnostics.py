"""Publish compact numerical diagnostics without checkpoint binaries or paths."""

from __future__ import annotations

import argparse
from pathlib import Path

from sirfex.research.store import ResearchStore
from sirfex.training.checkpoints import write_json
from sirfex.training.protocol import content_hash


def export(run_id: str, store: ResearchStore, destination: Path) -> dict:
    manifest = store.read(run_id, "manifest.json")
    protocol = store.read(run_id, "protocol.json")
    if manifest["kind"] != "policy-regression-diagnostics":
        raise ValueError("Expected numerical diagnostic run")
    if content_hash(protocol) != manifest["protocol_sha256"]:
        raise ValueError("Diagnostic protocol hash mismatch")
    value = {
        "schema_version": "sirfex-diagnostics-public:1",
        "run_id": run_id,
        "protocol": protocol,
        "protocol_sha256": manifest["protocol_sha256"],
        "status": manifest["status"],
        "summary": store.read(run_id, "summary.json"),
        "action_differences": store.read(run_id, "action_differences.json"),
        "tracking_differences": store.read(run_id, "tracking_differences.json"),
        "policy_shift": store.read(run_id, "policy_shift.json"),
        "hypotheses": store.read(run_id, "hypotheses.json"),
        "scenario_set": store.read(run_id, "scenario_set.json"),
        "checkpoint_hashes": {
            str(row["seed"]): row["checkpoint_sha256"]
            for row in store.read(run_id, "controller_pairs.json")["pairs"]
        },
        "note": "All actions are fixed-observation diagnostics; historical trajectory metrics "
        "come from saved independent development evaluations.",
    }
    write_json(destination, value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--store", default="artifacts/research")
    parser.add_argument("--output", default="docs/evidence/phase_3_6_diagnostics.json")
    args = parser.parse_args()
    result = export(args.run_id, ResearchStore(args.store), Path(args.output))
    print(result["status"])
