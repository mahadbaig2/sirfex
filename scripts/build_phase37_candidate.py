"""Build the review-only Phase 3.7 protocol from verified development inputs."""

import json
from pathlib import Path

from sirfex.robustness.identity import semantic_hash
from sirfex.training.checkpoints import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def build():
    controllers = read("phase_3_7_controller_preflight.json")["controllers"]
    scenarios = read("phase_3_7_development_scenario_manifest.json")
    domain = read("phase_3_7_disturbance_domain_preflight.json")
    if len(controllers) != 17 or len(scenarios["anchors"]) != 12 or len(domain["cells"]) != 192:
        raise ValueError("Incomplete preflight package")
    if any(not row.get("deterministic_inference_verified") for row in controllers[:15]):
        raise ValueError("An intended PPO checkpoint is not loadable")
    pairs = []
    for seed in (3, 17):
        for arm in ("expanded_5", "expanded_10"):
            pairs.append(
                {
                    "id": f"primary_{seed}_{arm}",
                    "class": "primary_equal_budget_screening",
                    "x": f"screen_{seed}_{arm}",
                    "y": f"screen_{seed}_normal",
                    "source_phase": "3.5",
                    "training_seed": seed,
                    "direction_reason": (
                        "Stress continuation investigated against matched normal continuation"
                    ),
                    "reverse_search": False,
                }
            )
    for seed in (3, 17, 41):
        pairs.append(
            {
                "id": f"secondary_{seed}_robust30",
                "class": "secondary_historical",
                "x": f"seed_{seed}_robust",
                "y": f"seed_{seed}_normal",
                "source_phase": "3.3",
                "training_seed": seed,
                "direction_reason": (
                    "Historical stress continuation investigated against within-seed "
                    "normal continuation"
                ),
                "reverse_search": False,
            }
        )
    context = [
        {
            "x": f"ppo_{seed}_{10240 if seed != 41 else 25600}",
            "y": ref,
            "class": "context_not_training_effect",
        }
        for seed in (3, 17, 41)
        for ref in ("pump-analytical", "pump-rule")
    ]
    search_path = ROOT / "configs/search/pump_avoidable.json"
    screening_path = ROOT / "artifacts/training/20261027T143749_d88d2dd37f/screening_runs.json"
    screening_records = json.loads(screening_path.read_text(encoding="utf-8"))["records"]
    exposures = []
    for record in screening_records:
        e = record["exposure"]
        if not e["reconciliation_valid"]:
            raise ValueError("Screening exposure ledger did not reconcile")
        exposures.append(
            {
                "seed": record["seed"],
                "arm": record["candidate_id"],
                "checkpoint_id": record["checkpoint"]["checkpoint_id"],
                "actual_environment_interactions": e["actual_environment_interactions"],
                "stress_interactions": e["stress_interactions"],
                "stress_interaction_fraction": e["stress_interaction_fraction"],
                "stress_completed_episodes": e["stress_completed_episodes"],
                "distinct_discoveries": e["distinct_discoveries"],
            }
        )
    if len(exposures) != 6:
        raise ValueError("Missing screening exposure ledger")
    grid_episodes = 17 * 12 * 16 * 25
    stress_context_episodes = 17 * 24
    # All 200 proposals might be viable Class B candidates and require in-memory repeat.
    differential_first = len(pairs) * 200 * 2
    differential_repeat_upper = differential_first
    headline_archive_upper = len(pairs) * 2 * 5 * 4
    headline_replay_upper = headline_archive_upper
    budget = {
        "schema_version": "phase37-execution-budget:1",
        "grid_episodes_per_controller_upper": 12 * 16 * 25,
        "grid_ppo_episodes_upper": 15 * 12 * 16 * 25,
        "grid_reference_episodes_upper": 2 * 12 * 16 * 25,
        "grid_episodes_upper": grid_episodes,
        "development_stress_context_episodes": stress_context_episodes,
        "grid_valid_scenario_points": domain["totals"]["valid"],
        "grid_physics_preprocessing_points": 12 * 16 * 25,
        "differential_pairs": len(pairs),
        "differential_first_pass_episodes_upper": differential_first,
        "differential_in_memory_repeat_episodes_upper": differential_repeat_upper,
        "headline_archive_episodes_upper": headline_archive_upper,
        "headline_independent_replay_episodes_upper": headline_replay_upper,
        "total_episode_upper": (
            grid_episodes
            + stress_context_episodes
            + 2 * differential_first
            + 2 * headline_archive_upper
        ),
        "maximum_transitions": (
            grid_episodes
            + stress_context_episodes
            + 2 * differential_first
            + 2 * headline_archive_upper
        )
        * 200,
        "checkpoint_loads_nominal": 15,
        "warm_ppo_seconds_grid_range": [72000 * 0.193, 72000 * 0.254],
        "warm_ppo_seconds_differential_upper_range": [
            (differential_first * 2 + headline_archive_upper * 2) * 0.193,
            (differential_first * 2 + headline_archive_upper * 2) * 0.254,
        ],
        "cold_model_load_seconds_observed_range": [15 * 3.9, 15 * 4.52],
        "compact_grid_case_count": 17 * 12 * 16,
        "compact_grid_bytes_at_observed_case_mean": round(17 * 12 * 16 * 8066.45),
        "headline_trajectory_bytes_at_observed_example": headline_archive_upper * 2571,
        "wall_seconds_cap": 8 * 3600,
        "per_ppo_seconds_cap": 30 * 60,
        "artifact_bytes_cap": 2 * 1024**3,
        "storage_policy": "compact all grid cells; full trajectory only selected headlines",
        "storage_estimate_note": (
            "Case and trajectory byte projections use 11 prior calibration cases "
            "and one prior 2571-byte trajectory; they are examples, not a worst-case bound"
        ),
    }
    write_json(EVIDENCE / "phase_3_7_execution_budget.json", budget)
    candidate = {
        "schema_version": "pump-robustness-checkpoint2-candidate:1",
        "status": "CANDIDATE — NOT FROZEN — NOT AUTHORIZED FOR EXECUTION",
        "executable": False,
        "scope": "development_only",
        "environment_id": "pump",
        "environment_version": "sirfex-pump:1",
        "controller_inventory": [
            {k: row[k] for k in ("group", "source_run", "source_key", "sha256", "source_artifact")}
            if row["group"] != "reference"
            else {"group": "reference", "controller_id": row["controller_id"]}
            for row in controllers
        ],
        "scenario_manifest_sha256": scenarios["semantic_sha256"],
        "authorized_partitions": [
            "training",
            "validation",
            "screening_stress",
            "historical_discoveries",
        ],
        "anchors": scenarios["anchors"],
        "disturbance": {
            "directions": 16,
            "grid": [round(i * 0.025, 3) for i in range(25)],
            "censoring": "parameter bounds reported, never clamped",
            "refinement": "disabled_pending_formal_monotonicity_review",
        },
        "differential_pairs": pairs,
        "context_pairs": context,
        "phase35_realized_exposure": {
            "source_file_sha256": sha256_file(screening_path),
            "arms": exposures,
        },
        "search": {
            "space": "configs/search/pump_avoidable.json",
            "space_file_sha256": sha256_file(search_path),
            "method": "seeded_random_nonadaptive",
            "search_seed": 3701,
            "scenario_seed_start": 37000000,
            "proposals_per_pair": 200,
            "top_per_class": 5,
            "class_a_margin_gap": 0.05,
            "exclusion": (
                "development training/validation/screening/historical discoveries; "
                "fixed proposals count excluded; no replacement; physical tolerance "
                "1e-4 and seed disjoint"
            ),
        },
        "endpoints": [
            "physical_violation",
            "signed_post_initial_margin",
            "first_observed_failure_delta",
            "nominal_tracking_mae",
            "stress_tracking_mae",
        ],
        "summary": [
            "per_controller_distribution",
            "per_scenario_paired_difference",
            "minimum_margin",
            "p10_margin",
            "median_margin",
            "censoring_count",
            "class_a_count",
            "verified_class_b_count",
            "replay_success",
        ],
        "budget": budget,
        "resume": (
            "verified immutable completed cases only; failed work explicit recovery; "
            "deterministic work order"
        ),
        "missing_evidence": "pair incomplete; no imputation or inferred ranking",
        "approval": "No approval token issued at Checkpoint 2A",
    }
    candidate["semantic_sha256"] = semantic_hash(candidate)
    write_json(ROOT / "configs/robustness/phase37_checkpoint2_protocol.candidate.json", candidate)
    return candidate


if __name__ == "__main__":
    value = build()
    print(value["semantic_sha256"])
