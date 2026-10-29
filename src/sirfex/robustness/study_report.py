"""Deterministic scientific summaries reconstructed from verified study artifacts."""

from __future__ import annotations

import statistics
from collections import defaultdict

from sirfex.experiments.storage import ExperimentStore
from sirfex.robustness.evidence import _validate_stored
from sirfex.robustness.identity import semantic_hash
from sirfex.robustness.store import RobustnessStorageError, controller_identity
from sirfex.robustness.study_store import StudyStore

REPORT_SCHEMA = "pump-scientific-study-report:2"
VIABLE_FEASIBILITY = {"target_feasible", "bounded_operation_feasible"}


def _grid_feasibility_counts(points: list[dict]) -> dict:
    """Count only analyzer-proven finite-horizon viable cases as avoidable."""
    groups = {
        key: {"scenarios": 0, "evaluated": 0, "failures": 0, "survivals": 0, "missing_evidence": 0}
        for key in ("viable", "infeasible", "unknown", "invalid")
    }
    censored = 0
    for point in points:
        if point.get("domain_status") != "valid":
            censored += 1
            continue
        feasibility = point.get("feasibility")
        group = (
            "viable"
            if feasibility in VIABLE_FEASIBILITY
            else "infeasible"
            if feasibility == "infeasible"
            else "unknown"
            if feasibility == "undetermined"
            else "invalid"
        )
        counts = groups[group]
        counts["scenarios"] += 1
        failure = point.get("failure")
        if type(failure) is bool:
            counts["evaluated"] += 1
            counts["failures" if failure else "survivals"] += 1
        else:
            counts["missing_evidence"] += 1
    return {
        "groups": groups,
        "censored": censored,
        "valid_evaluated": sum(
            groups[key]["evaluated"] for key in ("viable", "infeasible", "unknown")
        ),
    }


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "minimum": None, "p10": None, "median": None}
    ordered = sorted(values)
    index = (len(ordered) - 1) * 0.1
    lo = int(index)
    frac = index - lo
    p10 = ordered[lo] * (1 - frac) + ordered[min(lo + 1, len(ordered) - 1)] * frac
    return {
        "count": len(values),
        "minimum": min(values),
        "p10": p10,
        "median": statistics.median(values),
    }


def _paired(left: dict[str, dict], right: dict[str, dict]) -> dict:
    if set(left) != set(right) or not left:
        return {
            "status": "incomplete",
            "shared": len(set(left) & set(right)),
            "left": len(left),
            "right": len(right),
            "differences": [],
        }
    differences = [
        {
            "scenario_id": scenario_id,
            "tracking_mae_x_minus_y": left[scenario_id]["tracking_mae"]
            - right[scenario_id]["tracking_mae"],
            "post_initial_margin_x_minus_y": (
                left[scenario_id]["margin"]["post_initial_minimum_margin"]
                - right[scenario_id]["margin"]["post_initial_minimum_margin"]
            ),
            "physical_violation_x": left[scenario_id]["margin"]["constraint_violated"],
            "physical_violation_y": right[scenario_id]["margin"]["constraint_violated"],
        }
        for scenario_id in sorted(left)
    ]
    return {
        "status": "complete",
        "shared": len(differences),
        "left": len(left),
        "right": len(right),
        "differences": differences,
    }


def build_study_report(store: StudyStore, protocol: dict, manifest: dict, specs: dict) -> dict:
    """Verify every present item and archive before deriving any comparison."""
    plan = store.plan
    if (
        plan["protocol_sha256"] != protocol["semantic_sha256"]
        or plan["scenario_manifest_sha256"] != manifest["semantic_sha256"]
    ):
        raise RobustnessStorageError("Report inputs differ from stored study plan")
    stage_counts = store.status()["stages"]
    nominal: dict[str, dict[str, dict]] = defaultdict(dict)
    stress: dict[str, dict[str, dict]] = defaultdict(dict)
    cells: dict[str, list[dict]] = defaultdict(list)
    differential = []
    unverified_selected = False
    for item in plan["work_items"]:
        if item["stage"] == "report":
            continue
        saved = store.read(item["work_id"])
        if saved is None or saved["status"] not in {"completed", "censored"}:
            continue
        result = saved["result"]
        if item["stage"] == "fixed_stress":
            if result["scenario_id"] != item["scenario_id"]:
                raise RobustnessStorageError("Fixed-scenario result identity changed")
            stress[item["controller"]][item["scenario_id"]] = result
        elif item["stage"] == "grid":
            cell = result["cell"]
            if cell["anchor_id"] != item["anchor_id"] or cell["direction"] != item["direction"]:
                raise RobustnessStorageError("Grid-cell identity changed")
            cells[item["controller"]].append(cell)
            if result.get("nominal"):
                nominal[item["controller"]][item["anchor_id"]] = result["nominal"]
        elif item["stage"] == "archive":
            archive = result["result"]
            unverified_selected |= bool(archive["findings"]["class_b_unverified"])
            pair = next(row for row in plan["pairs"] if row["id"] == item["pair_id"])
            for finding in archive["findings"]["class_b"] + archive["findings"]["class_a"]:
                durable = finding.get("durable_evidence")
                if not durable or durable["verification_status"] != "independently_replayed":
                    raise RobustnessStorageError(
                        "Selected headline lacks verified durable evidence"
                    )
                experiment_store = ExperimentStore(store.directory / pair["id"] / "experiments")
                for role, episode in durable["experiments"].items():
                    key = pair["x"] if role.startswith("x_") else pair["y"]
                    spec = specs[key]
                    _validate_stored(
                        experiment_store,
                        episode["experiment_id"],
                        finding["scenario_id"],
                        spec["id"],
                        episode["episode_sha256"],
                        controller_identity(spec),
                        plan["protocol_sha256"],
                    )
            differential.append(
                {
                    "pair_id": pair["id"],
                    "coverage": archive["coverage"],
                    "findings": archive["findings"],
                    "durable_headline_selection": archive["durable_headline_selection"],
                }
            )
    controllers = []
    for key in plan["controllers"]:
        nominal_rows = nominal[key]
        stress_rows = stress[key]
        grid_rows = cells[key]
        all_points = [point for cell in grid_rows for point in cell["grid_outcomes"]]
        all_grid = [point for point in all_points if point["domain_status"] == "valid"]
        feasibility = _grid_feasibility_counts(all_points)
        groups = feasibility["groups"]
        controllers.append(
            {
                "controller": key,
                "checkpoint_sha256": next(
                    (
                        row["sha256"]
                        for row in protocol["controller_inventory"]
                        if row.get("source_key") == key
                    ),
                    None,
                ),
                "nominal_coverage": len(nominal_rows),
                "stress_coverage": len(stress_rows),
                "grid_cell_coverage": len(grid_rows),
                "nominal_tracking_mae": _distribution(
                    [r["tracking_mae"] for r in nominal_rows.values()]
                ),
                "stress_tracking_mae": _distribution(
                    [r["tracking_mae"] for r in stress_rows.values()]
                ),
                "nominal_margin": _distribution(
                    [r["margin"]["post_initial_minimum_margin"] for r in nominal_rows.values()]
                ),
                "stress_margin": _distribution(
                    [r["margin"]["post_initial_minimum_margin"] for r in stress_rows.values()]
                ),
                "nominal_violations": sum(
                    r["margin"]["constraint_violated"] for r in nominal_rows.values()
                ),
                "stress_violations": sum(
                    r["margin"]["constraint_violated"] for r in stress_rows.values()
                ),
                "all_grid_points": len(all_grid),
                "requested_grid_points": len(all_points),
                "valid_evaluated_grid_points": feasibility["valid_evaluated"],
                "grid_missing_evidence": sum(
                    group["missing_evidence"] for group in groups.values()
                ),
                "grid_invalid_feasibility_points": groups["invalid"]["scenarios"],
                "grid_censored_points": feasibility["censored"],
                "feasibility_coverage": groups,
                "physically_viable_points": groups["viable"]["evaluated"],
                "physically_infeasible_points": groups["infeasible"]["evaluated"],
                "physically_unknown_points": groups["unknown"]["evaluated"],
                "avoidable_grid_violations": groups["viable"]["failures"],
                "infeasible_grid_violations": groups["infeasible"]["failures"],
                "unknown_feasibility_grid_violations": groups["unknown"]["failures"],
                "first_failure_thresholds": [cell["delta_first"] for cell in grid_rows],
                "physical_infeasibility_thresholds": [
                    cell["delta_infeasible_grid"] for cell in grid_rows
                ],
                "censored_cells": sum(cell["status"].startswith("censored") for cell in grid_rows),
                "nonmonotone_cells": sum(cell["nonmonotone"] for cell in grid_rows),
            }
        )
    pairs = []
    for pair in plan["pairs"]:
        pairs.append(
            {
                "id": pair["id"],
                "class": pair["class"],
                "x": pair["x"],
                "y": pair["y"],
                "nominal": _paired(nominal[pair["x"]], nominal[pair["y"]]),
                "stress": _paired(stress[pair["x"]], stress[pair["y"]]),
            }
        )
    mandatory = {stage: counts for stage, counts in stage_counts.items() if stage != "report"}
    complete = all(
        row["pending"] == row["running"] == row["failed"] == row["invalid"] == 0
        for row in mandatory.values()
    )
    complete &= all(
        pair["nominal"]["status"] == pair["stress"]["status"] == "complete" for pair in pairs
    )
    complete &= not unverified_selected
    complete &= all(
        row["grid_missing_evidence"] == row["grid_invalid_feasibility_points"] == 0
        for row in controllers
    )
    context = [
        {
            "x": pair["x"],
            "y": pair["y"],
            "class": pair["class"],
            "nominal": _paired(nominal[pair["x"]], nominal[pair["y"]]),
            "stress": _paired(stress[pair["x"]], stress[pair["y"]]),
        }
        for pair in protocol.get("context_pairs", [])
    ]
    report = {
        "schema_version": REPORT_SCHEMA,
        "profile": plan["profile"],
        "execution_status": "complete" if complete else "partial_incomplete",
        "protocol_sha256": plan["protocol_sha256"],
        "plan_sha256": plan["semantic_sha256"],
        "scenario_manifest_sha256": plan["scenario_manifest_sha256"],
        "environment_version": "sirfex-pump:1",
        "search_space_identity": protocol["search"].get("space_file_sha256", "synthetic_fixture"),
        "stage_coverage": mandatory,
        "controllers": controllers,
        "matched_pairs": pairs,
        "context_pairs": context,
        "fixed_scenario_results": {
            key: {
                "nominal": [nominal[key][scenario_id] for scenario_id in sorted(nominal[key])],
                "stress": [stress[key][scenario_id] for scenario_id in sorted(stress[key])],
            }
            for key in plan["controllers"]
        },
        "grid_thresholds": [
            {
                "controller": key,
                "anchor_id": cell["anchor_id"],
                "direction": cell["direction"],
                "status": cell["status"],
                "first_failure_delta": cell["delta_first"],
                "physical_infeasibility_delta": cell["delta_infeasible_grid"],
                "relative_failure_threshold": cell["relative_failure_threshold"],
                "nonmonotone": cell["nonmonotone"],
                "censored_points": cell["requested_grid_points"] - cell["valid_grid_points"],
            }
            for key in plan["controllers"]
            for cell in cells[key]
        ],
        "differential": sorted(differential, key=lambda row: row["pair_id"]),
        "seeds": {
            "search": protocol["search"]["search_seed"],
            "scenario_start": protocol["search"]["scenario_seed_start"],
        },
        "reproduction": {
            "command": (
                "sirfex robustness study --mode report --protocol <file> "
                "--manifest <file> --output-dir <run-root>"
            ),
            "replay_command": (
                "sirfex experiments --store <pair>/experiments replay <experiment-id>"
            ),
        },
        "limitations": [
            "Simplified pump simulator; no real-world safety guarantee.",
            "Development-only scenarios; bounded grid and search are not global minima.",
            "Related PPO checkpoints and three training seeds are not independent trajectories.",
            "Null, adverse, tied, and censored outcomes remain reportable.",
        ],
    }
    report["semantic_sha256"] = semantic_hash(report)
    return report


def render_study_markdown(report: dict) -> str:
    expected = semantic_hash({k: v for k, v in report.items() if k != "semantic_sha256"})
    if report.get("schema_version") != REPORT_SCHEMA or report.get("semantic_sha256") != expected:
        raise ValueError("Scientific report identity mismatch")
    lines = [
        "# SIRFEX physical robustness study",
        "",
        f"Status: **{report['execution_status']}**; profile: `{report['profile']}`.",
        f"Protocol: `{report['protocol_sha256']}`; plan: `{report['plan_sha256']}`.",
        f"Report SHA-256: `{report['semantic_sha256']}`.",
        "",
        "## Coverage",
        "",
    ]
    for stage, counts in sorted(report["stage_coverage"].items()):
        lines.append(f"- {stage}: {counts}")
    lines += [
        "",
        "## Controllers",
        "",
        "| Controller | Nominal | Stress | Grid cells | Viable grid | "
        "Infeasible grid | Unknown grid | Avoidable grid violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["controllers"]:
        lines.append(
            f"| `{row['controller']}` | {row['nominal_coverage']} | "
            f"{row['stress_coverage']} | {row['grid_cell_coverage']} | "
            f"{row['physically_viable_points']} | {row['physically_infeasible_points']} | "
            f"{row['physically_unknown_points']} | "
            f"{row['avoidable_grid_violations']} |"
        )
    lines += ["", "## Matched comparisons", ""]
    for pair in report["matched_pairs"]:
        lines.append(
            f"- `{pair['id']}` ({pair['class']}): nominal {pair['nominal']['status']}, "
            f"stress {pair['stress']['status']}."
        )
    lines += ["", "## Differential evidence", ""]
    for row in report["differential"]:
        lines.append(
            f"- `{row['pair_id']}`: {row['coverage']}; "
            f"verified B={len(row['findings']['class_b'])}; "
            f"A={len(row['findings']['class_a'])}."
        )
    lines += [
        "",
        "## Reproducibility and limits",
        "",
        f"Regenerate: `{report['reproduction']['command']}`.",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"
