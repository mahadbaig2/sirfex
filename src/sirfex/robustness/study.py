"""Deterministic, inspectable work planning for a future authorized pump study.

Planning is read-only. The synthetic profile is capped independently and never
constitutes approval for the development controller cohort.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from sirfex.core import Scenario
from sirfex.robustness.disturbance import DIRECTIONS, validate_direction, validate_grid
from sirfex.robustness.identity import scenario_identity, semantic_hash
from sirfex.robustness.scientific import StudyGateError, _checked_hash, dry_run

PLAN_SCHEMA = "pump-robustness-study-plan:1"


def _controller_key(row: dict) -> str:
    return row.get("source_key") or row.get("controller_id") or row["id"]


def build_plan(
    protocol: dict,
    manifest: dict,
    root: Path | str,
    *,
    synthetic: bool = False,
    inputs_verified: bool = False,
) -> dict:
    """Bind every fixed work item to the exact protocol and manifest identity."""
    if synthetic:
        if protocol.get("profile") != "synthetic_test_only" or protocol.get("scope") != "synthetic":
            raise StudyGateError("Synthetic plan requires isolated test-only profile")
        if (
            protocol.get("status") != "SYNTHETIC TEST FIXTURE"
            or protocol.get("executable") is not False
        ):
            raise StudyGateError("Synthetic fixtures cannot authorize scientific execution")
        _checked_hash(protocol)
        if _checked_hash(manifest) != protocol.get("scenario_manifest_sha256"):
            raise StudyGateError("Synthetic manifest identity mismatch")
        if len(protocol["controller_inventory"]) > 3 or len(protocol["anchors"]) > 2:
            raise StudyGateError("Synthetic controller or anchor cap exceeded")
        if len(protocol["disturbance"]["directions"]) > 4:
            raise StudyGateError("Synthetic direction cap exceeded")
        if len(protocol["disturbance"]["grid"]) > 5:
            raise StudyGateError("Synthetic grid cap exceeded")
        if protocol["search"]["proposals_per_pair"] > 5:
            raise StudyGateError("Synthetic differential proposal cap exceeded")
    elif not inputs_verified:
        dry_run(protocol, manifest, root)
    controllers = [_controller_key(row) for row in protocol["controller_inventory"]]
    if len(controllers) != len(set(controllers)):
        raise StudyGateError("Duplicate controller identity")
    nominal_partition = "synthetic_nominal" if synthetic else "validation"
    stress_partition = "synthetic_stress" if synthetic else "screening_stress"
    rows = manifest["scenarios"]
    nominal = [row for row in rows if row["partition"] == nominal_partition]
    stress = [row for row in rows if row["partition"] == stress_partition]
    if not nominal or not stress or [row["scenario_id"] for row in nominal] != protocol["anchors"]:
        raise StudyGateError("Nominal or development-stress scenario plan is incomplete")
    for row in nominal + stress:
        scenario = Scenario.from_dict(row["scenario"])
        if scenario_identity(scenario) != row["scenario_id"]:
            raise StudyGateError("Scenario identity changed while planning")
        if not synthetic and scenario.environment_version != "sirfex-pump:1":
            raise StudyGateError("Unsupported physical environment")
    directions = (
        [
            validate_direction(row)
            for row in protocol["disturbance"].get(
                "directions", DIRECTIONS if not synthetic else ()
            )
        ]
        if synthetic
        else list(DIRECTIONS)
    )
    if not directions or len(directions) != len(set(directions)):
        raise StudyGateError("Duplicate or empty disturbance directions")
    if not synthetic and protocol["disturbance"]["directions"] != len(DIRECTIONS):
        raise StudyGateError("Scientific direction set changed")
    grid = validate_grid(protocol["disturbance"]["grid"])
    if len(grid) > (5 if synthetic else 25):
        raise StudyGateError("Grid exceeds profile cap")
    pairs = protocol["differential_pairs"]
    if not pairs or len({pair["id"] for pair in pairs}) != len(pairs):
        raise StudyGateError("Duplicate or missing differential pairs")
    if any(
        pair["x"] not in controllers or pair["y"] not in controllers or pair["x"] == pair["y"]
        for pair in pairs
    ):
        raise StudyGateError("Differential pair names an unavailable controller")
    base = {
        "schema_version": PLAN_SCHEMA,
        "profile": "synthetic_test_only" if synthetic else "development_scientific",
        "protocol_sha256": protocol["semantic_sha256"],
        "scenario_manifest_sha256": manifest["semantic_sha256"],
        "controllers": controllers,
        "nominal_scenarios": protocol["anchors"],
        "stress_scenarios": [row["scenario_id"] for row in stress],
        "directions": [list(row) for row in directions],
        "grid": list(grid),
        "pairs": pairs,
        "search": protocol["search"],
        "resource_limits": protocol["budget"],
    }
    base_sha = semantic_hash(base)
    items = []

    def append(stage: str, **fields) -> None:
        payload = {"stage": stage, **fields}
        items.append(
            {"work_id": semantic_hash({"plan_base_sha256": base_sha, **payload}), **payload}
        )

    for controller in controllers:
        for scenario_id in base["stress_scenarios"]:
            append("fixed_stress", controller=controller, scenario_id=scenario_id)
        for anchor_id in base["nominal_scenarios"]:
            for direction in base["directions"]:
                append("grid", controller=controller, anchor_id=anchor_id, direction=direction)
    for pair_index, pair in enumerate(pairs):
        for proposal_index in range(protocol["search"]["proposals_per_pair"]):
            append(
                "differential",
                pair_id=pair["id"],
                proposal_index=proposal_index,
                search_seed=protocol["search"]["search_seed"] + pair_index,
                scenario_seed=protocol["search"]["scenario_seed_start"]
                + pair_index * protocol["search"]["proposals_per_pair"]
                + proposal_index,
            )
        append("archive", pair_id=pair["id"])
    append("report")
    identifiers = [row["work_id"] for row in items]
    if len(identifiers) != len(set(identifiers)):
        raise StudyGateError("Duplicate work item identity")
    counts = dict(sorted(Counter(row["stage"] for row in items).items()))
    plan = {**base, "plan_base_sha256": base_sha, "work_items": items, "work_counts": counts}
    plan["semantic_sha256"] = semantic_hash(plan)
    return plan
