"""Build a minimized, development-only evidence context for proposal selection."""

from __future__ import annotations

from sirfex.core import Scenario
from sirfex.training.feasibility import analyze_feasibility
from sirfex.training.phase35 import SOURCE_32, SOURCE_34
from sirfex.training.protocol import content_hash, scenario_record


def verified_discoveries(training) -> dict[str, dict]:
    old = training.load(SOURCE_34)
    if old["manifest"]["status"] != "no_eligible_curriculum":
        raise ValueError("Frozen discovery source is incomplete")
    result = {}
    for category in ("original", "new"):
        for discovery_id, row in old["candidate_curricula"][category].items():
            scenario = Scenario.from_dict(row["scenario"])
            feasible = analyze_feasibility(scenario)
            replays = (
                [item for witness in row["witnesses"] for item in witness["replay"]]
                if category == "original"
                else row["replay"]
            )
            if (
                feasible["finite_horizon_viable"] is not True
                or not replays
                or any(item["status"] not in ("exact_match", "tolerance_match") for item in replays)
            ):
                raise ValueError(f"Discovery {discovery_id} lacks feasible replay evidence")
            result[discovery_id] = {
                "scenario_id": scenario_record(scenario)["scenario_id"],
                "scenario": scenario,
                "feasibility": feasible,
                "replay_statuses": [item["status"] for item in replays],
            }
    if len(result) != 7:
        raise ValueError("Expected seven verified avoidable discoveries")
    return result


def evidence_context(
    training, research, *, diagnostic_run_id: str, starting_seed: int, training_timesteps: int
) -> tuple[dict, dict]:
    diagnostic = research.read(diagnostic_run_id, "manifest.json")
    protocol = research.read(diagnostic_run_id, "protocol.json")
    if (
        diagnostic.get("kind") != "policy-regression-diagnostics"
        or diagnostic.get("status") != "completed"
        or content_hash(protocol) != diagnostic.get("protocol_sha256")
        or starting_seed not in protocol["seeds"]
    ):
        raise ValueError("Diagnostic run is incomplete, changed or lacks this seed")
    source35 = training.load("20261027T143749_d88d2dd37f")
    source32 = training.load(SOURCE_32)
    if source35["manifest"]["status"] != "no_eligible_curriculum":
        raise ValueError("Phase 3.5 development source is incomplete")
    actions = research.read(diagnostic_run_id, "action_differences.json")[str(starting_seed)]
    measured = research.read(diagnostic_run_id, "tracking_differences.json")[str(starting_seed)]
    discoveries = verified_discoveries(training)
    context = {
        "schema_version": "sirfex-research-context:1",
        "diagnostic_run_id": diagnostic_run_id,
        "historical_study_id": source35["manifest"]["training_run_id"],
        "starting_seed": starting_seed,
        "training_timesteps": training_timesteps,
        "nominal_guardrail_relative_mae": 0.10,
        "diagnostic_observation_count": actions["summary"]["baseline"]["count"],
        "normal_to_stress_action_shift": {
            arm: actions["shifts_from_baseline"][f"normal_to_{arm}"]
            for arm in ("expanded_5", "expanded_10")
        },
        "development_metrics": {
            arm: {
                partition: {
                    "mean_tracking_error": measured[arm][partition]["mean_tracking_error"],
                    "constraint_violation_frequency": measured[arm][partition][
                        "constraint_violation_frequency"
                    ],
                }
                for partition in ("development_nominal", "development_stress")
            }
            for arm in ("normal", "expanded_5", "expanded_10")
        },
        "verified_discovery_ids": sorted(discoveries),
        "allowed_interventions": [
            "scheduled_curriculum",
            "nominal_rehearsal",
            "discovery_weighting",
        ],
        "evaluation_policy": "development_only",
    }
    local = {
        "evidence_ids": {diagnostic_run_id, source35["manifest"]["training_run_id"]},
        "discoveries": discoveries,
        "baseline_metadata": source32["baseline_selection"]["selected"][str(starting_seed)],
        "development_hashes": {
            "nominal": source35["protocol"]["development_nominal_hash"],
            "stress": source35["protocol"]["development_stress_hash"],
        },
    }
    if len(str(context)) > 8000:
        raise ValueError("Evidence context is not minimized")
    return context, local
