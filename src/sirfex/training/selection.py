"""Audited Phase 2 discovery interpretation, including rejected discoveries."""

from collections import Counter

from sirfex.core import Scenario
from sirfex.experiments.replay import replay_experiment
from sirfex.training.feasibility import analyze_feasibility
from sirfex.training.protocol import content_hash, physical_key


def discovery_report(search_store, experiment_store, search_ids, limit=8):
    records, selected, seen = [], [], set()
    for search_id in search_ids:
        loaded = search_store.load(search_id)
        if loaded["manifest"].get("schema_version") != "1":
            raise ValueError("Unsupported search schema")
        for discovery in loaded["discoveries"]:
            row = {
                "search_id": search_id,
                "discovery_id": discovery["discovery_id"],
                "valid": False,
                "replay_verified": False,
                "eligible": False,
                "experiment_ids": discovery.get("experiment_ids", []),
            }
            records.append(row)
            try:
                scenario = Scenario.from_dict(discovery["scenario"])
                if (
                    scenario.schema_version != "1"
                    or scenario.environment_version != "sirfex-pump:1"
                ):
                    raise ValueError("Explicit compatible scenario versions required")
                row["feasibility"] = analyze_feasibility(scenario)
                row["valid"] = True
                row["scenario"] = scenario.to_dict()
                if (
                    not discovery.get("observed_failure")
                    or discovery.get("evidence_status") != "observed"
                ):
                    raise ValueError("Discovery is not an observed physical failure")
                if (
                    discovery.get("objective", {}).get("objective_id")
                    != "pump.constraint_violation"
                ):
                    raise ValueError("Discovery objective is not pump constraint violation")
                if not row["experiment_ids"]:
                    raise ValueError("Missing supporting experiments")
                reports = []
                for experiment_id in row["experiment_ids"]:
                    stored = experiment_store.load(experiment_id)
                    if physical_key(stored.scenario) != physical_key(scenario):
                        raise ValueError("Supporting scenario differs from discovery")
                    if stored.result.get("execution_status") != "completed" or not any(
                        f.get("category") == "constraint_violation"
                        for f in stored.result.get("failures", [])
                    ):
                        raise ValueError("No supporting observed physical failure")
                    reports.append(replay_experiment(experiment_store, experiment_id).to_dict())
                row["replay_reports"] = reports
                row["replay_verified"] = all(
                    x["status"] in ("exact_match", "tolerance_match") for x in reports
                )
                key = content_hash(physical_key(scenario))
                row["eligible"] = (
                    row["replay_verified"]
                    and row["feasibility"]["classification"]
                    in ("target_feasible", "bounded_operation_feasible")
                    and key not in seen
                )
                if row["eligible"] and len(selected) < limit:
                    selected.append(row)
                    seen.add(key)
            except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
                row["exclusion_reason"] = str(exc)
    counts = Counter(row.get("feasibility", {}).get("classification", "invalid") for row in records)
    return {
        "schema_version": "discovery-feasibility:1",
        "total_discoveries": len(records),
        "valid_discoveries": sum(r["valid"] for r in records),
        "replay_verified": sum(r["replay_verified"] for r in records),
        "classifications": dict(counts),
        "records": records,
        "selected": selected,
    }
