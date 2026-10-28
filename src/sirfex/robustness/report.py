"""JSON-authoritative robustness evidence and deterministic Markdown rendering."""

from __future__ import annotations

import statistics
from typing import Any

from sirfex.robustness.identity import semantic_hash

SCHEMA = "pump-robustness-report:1"


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def build_report(
    *,
    profile: str,
    protocol: dict,
    controllers: list[dict],
    scenarios: list[dict],
    margin_results: list[dict],
    disturbance_cells: list[dict],
    differential_results: list[dict],
    reproducibility: dict,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    if profile not in ("synthetic", "software_calibration_only"):
        raise ValueError("Checkpoint 1 reports only synthetic or software calibration evidence")
    if protocol.get("schema_version") != "pump-robustness-protocol:1":
        raise ValueError("Report requires a versioned robustness protocol")
    if any(item.get("schema_version") != "pump-signed-margin:1" for item in margin_results):
        raise ValueError("Report contains unrecognized signed margin evidence")
    if any(item.get("schema_version") != "pump-disturbance-cell:1" for item in disturbance_cells):
        raise ValueError("Report contains unrecognized disturbance evidence")
    if any(
        item.get("schema_version") != "pump-differential-search:1" for item in differential_results
    ):
        raise ValueError("Report contains unrecognized differential evidence")
    valid_margins = [item for item in margin_results if item["evidence_status"] == "valid"]
    result = {
        "schema_version": SCHEMA,
        "profile": profile,
        "scientific_protocol_frozen": False,
        "final_holdouts_accessed": False,
        "protocol": protocol,
        "protocol_sha256": semantic_hash(protocol),
        "controllers": controllers,
        "scenarios": scenarios,
        "margin_results": margin_results,
        "margin_summary": {
            "valid_episodes": len(valid_margins),
            "incomplete_episodes": len(margin_results) - len(valid_margins),
            "full_episode_minimum": _distribution(
                [item["full_episode_minimum_margin"] for item in valid_margins]
            ),
            "post_initial_minimum": _distribution(
                [item["post_initial_minimum_margin"] for item in valid_margins]
            ),
            "observed_violations": sum(item["constraint_violated"] for item in valid_margins),
        },
        "disturbance_cells": disturbance_cells,
        "disturbance_coverage": {
            "cells": len(disturbance_cells),
            "requested_grid_points": sum(c["requested_grid_points"] for c in disturbance_cells),
            "evaluated_grid_points": sum(c["evaluated_grid_points"] for c in disturbance_cells),
            "by_status": {
                status: sum(c["status"] == status for c in disturbance_cells)
                for status in (
                    "fails_at_nominal",
                    "failed",
                    "censored_bound",
                    "censored_max",
                    "undefined_axis",
                )
            },
        },
        "differential_results": differential_results,
        "differential_coverage": {
            "searches": len(differential_results),
            "class_a": sum(len(r["findings"]["class_a"]) for r in differential_results),
            "class_b": sum(len(r["findings"]["class_b"]) for r in differential_results),
        },
        "reproducibility": reproducibility,
        "warnings": list(warnings or []),
        "scientific_limitations": [
            "Simplified simulated pump; no real-world safety certification.",
            "Bounded search does not prove global robustness.",
            "Development observations are not independent final holdouts.",
            "Related PPO checkpoints are not statistically independent.",
            "No successful controller-improvement claim is made.",
        ],
    }
    result["semantic_sha256"] = semantic_hash(result)
    return result


def verify_report(record: dict) -> None:
    if record.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported robustness report schema")
    digest = record.get("semantic_sha256")
    body = {key: value for key, value in record.items() if key != "semantic_sha256"}
    if digest != semantic_hash(body) or record.get("protocol_sha256") != semantic_hash(
        record["protocol"]
    ):
        raise ValueError("Robustness report evidence hash mismatch")


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def render_markdown(record: dict) -> str:
    verify_report(record)
    summary = record["margin_summary"]
    coverage = record["disturbance_coverage"]
    differential = record["differential_coverage"]
    lines = [
        "# Pump robustness evidence — Checkpoint 1",
        "",
        f"**Profile:** `{record['profile']}`. **Scientific protocol frozen:** no.",
        f"**Evidence SHA-256:** `{record['semantic_sha256']}`.",
        "",
        "## Controllers and scenarios",
        "",
        f"Controllers: {len(record['controllers'])}; scenarios: {len(record['scenarios'])}.",
    ]
    for controller in record["controllers"]:
        lines.append(
            f"- `{controller['id']}` — checkpoint `{controller.get('checkpoint_sha256') or 'none'}`"
        )
    lines += [
        "",
        "## Signed margins",
        "",
        "| Measure | Count | Minimum | Median | Maximum |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, key in (
        ("Full episode M_min", "full_episode_minimum"),
        ("Post-initial M_post", "post_initial_minimum"),
    ):
        values = summary[key]
        lines.append(
            f"| {name} | {values['count']} | {_number(values['minimum'])} | "
            f"{_number(values['median'])} | {_number(values['maximum'])} |"
        )
    lines += [
        "",
        f"Observed violations: {summary['observed_violations']}; "
        f"incomplete episodes: {summary['incomplete_episodes']}.",
        "",
        "| Controller | Scenario SHA-256 | M0 | M_min | M_post | ΔM | Critical t/boundary |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for margin in record["margin_results"]:
        lines.append(
            f"| `{margin['controller_id']}` | `{margin['scenario_id']}` | "
            f"{_number(margin['initial_margin'])} | "
            f"{_number(margin['full_episode_minimum_margin'])} | "
            f"{_number(margin['post_initial_minimum_margin'])} | "
            f"{_number(margin['signed_change_from_initial'])} | "
            f"{margin['critical_timestep']}/{margin['critical_boundary']} |"
        )
    lines += [
        "",
        "## Disturbance tolerance",
        "",
        f"Cells: {coverage['cells']}; evaluated grid points: "
        f"{coverage['evaluated_grid_points']}/{coverage['requested_grid_points']}.",
        "",
        "| Controller | Direction | Outcome | First observed failure | "
        "First grid infeasibility | Relative threshold | Nonmonotone |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for cell in record["disturbance_cells"]:
        lines.append(
            f"| `{cell['controller_id']}` | {'+'.join(cell['direction'])} | "
            f"`{cell['status']}` | {_number(cell['delta_first'])} | "
            f"{_number(cell['delta_infeasible_grid'])} | "
            f"{_number(cell['relative_failure_threshold'])} | "
            f"{cell['nonmonotone']} / physics {cell['physics_nonmonotone']} |"
        )
    lines += [
        "",
        "## Differential findings",
        "",
        f"Class B actual avoidable failures: {differential['class_b']}; "
        f"Class A near-misses: {differential['class_a']}.",
    ]
    if differential["class_b"] + differential["class_a"] == 0:
        lines.append("No qualifying differential finding was observed in this bounded run.")
    for search in record["differential_results"]:
        for classification in ("class_b", "class_a"):
            for finding in search["findings"][classification]:
                lines.append(
                    f"- `{classification}` `{finding['scenario_id']}`: "
                    f"post-initial margin gap {_number(finding['signed_margin_gap'])}; "
                    f"replay `{','.join(finding['replay_statuses']) or 'not_run'}`."
                )
    lines += [
        "",
        "## Reproducibility",
        "",
        f"Git commit: `{record['reproducibility'].get('git_commit') or 'unavailable'}`.",
        f"Configuration hash: `{record['protocol_sha256']}`.",
        f"Command: `{record['reproducibility'].get('command', 'unavailable')}`.",
        f"Replay: `{record['reproducibility'].get('replay_status', 'not_run')}`.",
        "",
        "## Limits",
        "",
    ]
    lines.extend(f"- {text}" for text in record["scientific_limitations"])
    lines.extend(f"- Warning: {text}" for text in record["warnings"])
    return "\n".join(lines) + "\n"
