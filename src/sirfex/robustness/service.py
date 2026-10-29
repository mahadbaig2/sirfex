"""Explicit, bounded Checkpoint 1 orchestration over existing SIRFEX adapters."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from sirfex import __version__
from sirfex.core import EvaluationRunner, Scenario
from sirfex.environments.pump import PumpEnvironment
from sirfex.robustness.differential import sample_differentials
from sirfex.robustness.disturbance import (
    evaluate_direction,
    prepare_grid,
    validate_direction,
    validate_grid,
)
from sirfex.robustness.evidence import archive_selected_findings
from sirfex.robustness.identity import scenario_identity, semantic_hash
from sirfex.robustness.margins import evaluate_margin
from sirfex.robustness.report import build_report, render_markdown
from sirfex.robustness.store import (
    RobustnessStore,
    _atomic_json,
    atomic_text,
    controller_identity,
    run_items,
)
from sirfex.search.space import SearchSpace


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _controllers(value: object, *, max_count: int) -> list[dict]:
    if not isinstance(value, list) or not 1 <= len(value) <= max_count:
        raise ValueError(f"Calibration requires 1..{max_count} controllers")
    allowed = {"pump-rule", "pump-analytical", "pump-fixed", "ppo"}
    result = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {"id", "config"}:
            raise ValueError("Controller spec requires exactly id and config")
        if row["id"] not in allowed or not isinstance(row["config"], dict):
            raise ValueError("Unsupported calibration controller")
        if row["id"] == "ppo":
            checkpoint = row["config"].get("checkpoint")
            if not isinstance(checkpoint, dict) or not checkpoint.get("sha256"):
                raise ValueError("PPO calibration requires trusted checkpoint metadata")
        result.append(row)
    if len({controller_identity(row) for row in result}) != len(result):
        raise ValueError("Duplicate calibration controller specs")
    return result


def _scenario(value: dict) -> Scenario:
    scenario = Scenario.from_dict(value)
    if any(
        word in str(scenario.provenance).lower() or word in str(scenario.tags).lower()
        for word in ("final", "holdout", "held_out")
    ):
        raise ValueError("Final or held-out scenarios are outside Checkpoint 1")
    p = PumpEnvironment().validate_scenario(scenario)
    if not 1 <= p["horizon"] <= 200:
        raise ValueError("Checkpoint 1 calibration horizon must be <=200")
    return scenario


def _profile(value: dict, allowed: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != allowed:
        raise ValueError("Calibration configuration fields do not match the versioned schema")
    if (
        value["schema_version"] != "pump-robustness-calibration:1"
        or value["profile"] != "software_calibration_only"
    ):
        raise ValueError("Checkpoint 1 accepts software calibration only")


class _Session:
    def __init__(self, spec: dict) -> None:
        from sirfex.experiments.factories import make_controller, make_environment

        self.environment = make_environment("pump")
        self.controller = make_controller(spec["id"], self.environment, spec["config"])
        self.spec = spec
        self.episodes = 0
        self.transitions = 0
        self.seconds = 0.0

    def evaluate(self, scenario: Scenario):
        horizon = PumpEnvironment().validate_scenario(scenario)["horizon"]
        started = time.perf_counter()
        result = EvaluationRunner(horizon).run_episode(
            self.environment, self.controller, scenario, record_trace=True
        )
        self.seconds += time.perf_counter() - started
        self.episodes += 1
        self.transitions += result.episode_length
        return result

    def close(self):
        self.environment.close()


def _controller_record(spec: dict) -> dict:
    config = json.loads(json.dumps(spec["config"], allow_nan=False))
    checkpoint = config.get("checkpoint")
    sha = checkpoint.get("sha256") if isinstance(checkpoint, dict) else None
    checkpoint_id = checkpoint.get("checkpoint_id") if isinstance(checkpoint, dict) else None
    if isinstance(checkpoint, dict):
        checkpoint.pop("path", None)
    return {
        "type": spec["id"],
        "id": spec["id"],
        "config": config,
        "identity_sha256": controller_identity(spec),
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": sha,
        "environment_id": "pump",
        "environment_version": PumpEnvironment.version,
    }


def _rss() -> int | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except ImportError:
        return None


def _finish(
    store: RobustnessStore,
    run_id: str,
    protocol: dict,
    controllers: list[dict],
    scenarios: list[Scenario],
    margins: list[dict],
    cells: list[dict],
    differential: list[dict],
    calibration: dict,
    command: str,
) -> dict:
    directory = store.directory(run_id)
    report = build_report(
        profile="software_calibration_only",
        protocol=protocol,
        controllers=[_controller_record(spec) for spec in controllers],
        scenarios=[
            {"scenario_id": scenario_identity(s), "scenario": s.to_dict()} for s in scenarios
        ],
        margin_results=margins,
        disturbance_cells=cells,
        differential_results=differential,
        reproducibility={
            "sirfex_version": __version__,
            "git_commit": _git_commit(),
            "config_sha256": semantic_hash(protocol),
            "command": command,
            "source_evidence_hashes": [
                spec["config"].get("checkpoint", {}).get("sha256")
                for spec in controllers
                if spec["id"] == "ppo"
            ],
            "deterministic_seeds": [scenario.seed for scenario in scenarios],
            "replay_status": "see_per_finding_durable_evidence" if differential else "not_run",
        },
        case_status_counts=(
            {
                key: value
                for key, value in calibration["counts"].items()
                if key not in ("reused_completed", "semantic_results_sha256")
            }
            if "counts" in calibration
            else None
        ),
        warnings=["Software calibration only; thresholds are provisional."],
    )
    _atomic_json(directory / "report.json", report)
    atomic_text(directory / "report.md", render_markdown(report))
    calibration_path = directory / "calibration.json"
    if calibration.get("measured_episodes_this_invocation") == 0 and calibration_path.exists():
        original = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration = {
            **calibration,
            "first_measurement": original.get("first_measurement", original),
            "last_resume_reused_completed": calibration.get("counts", {}).get(
                "reused_completed", 0
            ),
        }
    _atomic_json(calibration_path, calibration)
    return {
        "run_id": run_id,
        "profile": "software_calibration_only",
        "report_path": str(directory / "report.md"),
        "report_sha256": report["semantic_sha256"],
        "calibration": calibration,
    }


def run_tolerance(config: dict, output_dir: Path | str) -> dict:
    _profile(
        config,
        {"schema_version", "profile", "anchor_scenarios", "directions", "grid", "controllers"},
    )
    anchors = [_scenario(row) for row in config["anchor_scenarios"]]
    if not 1 <= len(anchors) <= 2 or len({scenario_identity(s) for s in anchors}) != len(anchors):
        raise ValueError("Calibration requires one or two distinct permitted anchors")
    directions = [validate_direction(row) for row in config["directions"]]
    if not 1 <= len(directions) <= 4 or len(set(directions)) != len(directions):
        raise ValueError("Calibration requires one to four distinct directions")
    grid = validate_grid(config["grid"])
    if len(grid) > 5:
        raise ValueError("Checkpoint 1 calibration grid must have at most five points")
    controllers = _controllers(config["controllers"], max_count=3)
    protocol = {
        "schema_version": "pump-robustness-protocol:1",
        "profile": config["profile"],
        "anchors": [scenario_identity(s) for s in anchors],
        "directions": [list(d) for d in directions],
        "grid": list(grid),
        "controllers": [_controller_record(spec) for spec in controllers],
        "limits": {"anchors": 2, "directions": 4, "grid_points": 5, "controllers": 3},
    }
    store = RobustnessStore(output_dir)
    run_id = store.open_run(protocol)
    prepared = {
        (scenario_identity(anchor), direction): prepare_grid(
            anchor, direction, grid, protocol_sha256=semantic_hash(protocol)
        )
        for anchor in anchors
        for direction in directions
    }
    sessions = {}
    started = time.perf_counter()
    rss_before = _rss()

    def evaluate_case(spec: dict, anchor: Scenario) -> dict:
        key = controller_identity(spec)
        if key not in sessions:
            sessions[key] = _Session(spec)
        session = sessions[key]
        nominal = evaluate_margin(session.evaluate(anchor)).to_dict()
        cells = [
            evaluate_direction(
                anchor,
                direction,
                grid,
                session.controller.id,
                session.evaluate,
                prepared=prepared[(scenario_identity(anchor), direction)],
                refine_feasibility=True,
                protocol_sha256=semantic_hash(protocol),
            ).to_dict()
            for direction in directions
        ]
        return {
            "nominal_margin": nominal,
            "cells": cells,
            "censored": all(c["status"].startswith("censored") for c in cells),
        }

    try:
        counts = run_items(
            store,
            run_id,
            ((spec, anchor) for spec in controllers for anchor in anchors),
            evaluate_case,
        )
    finally:
        for session in sessions.values():
            session.close()
    cases = [
        store.read_case(run_id, store.case_id(run_id, spec, anchor))
        for spec in controllers
        for anchor in anchors
    ]
    margin_results = [case["result"]["nominal_margin"] for case in cases]
    cells = [cell for case in cases for cell in case["result"]["cells"]]
    seconds = time.perf_counter() - started
    episodes = sum(session.episodes for session in sessions.values())
    transitions = sum(session.transitions for session in sessions.values())
    calibration = {
        "schema_version": "pump-robustness-calibration-result:1",
        "profile": "software_calibration_only",
        "counts": counts,
        "measured_seconds_this_invocation": seconds,
        "measured_episodes_this_invocation": episodes,
        "measured_transitions_this_invocation": transitions,
        "episodes_per_second": episodes / seconds if seconds else None,
        "transitions_per_second": transitions / seconds if seconds else None,
        "observed_rss_bytes_start": rss_before,
        "observed_rss_bytes_end": _rss(),
        "per_controller": {
            session.controller.id: {
                "episodes": session.episodes,
                "transitions": session.transitions,
                "seconds": session.seconds,
                "episodes_per_second": (
                    session.episodes / session.seconds if session.seconds else None
                ),
            }
            for session in sessions.values()
        },
        "projected_82000_episode_seconds": (82000 * seconds / episodes if episodes else None),
        "projection_note": "Extrapolation from tiny calibration, not a measured full study.",
        "summary_case_storage_bytes": sum(
            p.stat().st_size for p in (store.directory(run_id) / "cases").glob("*.json")
        ),
    }
    return _finish(
        store,
        run_id,
        protocol,
        controllers,
        anchors,
        margin_results,
        cells,
        [],
        calibration,
        "sirfex robustness tolerance --config <config> --output-dir <directory> --mode calibration",
    )


def run_differential(config: dict, output_dir: Path | str) -> dict:
    _profile(
        config,
        {
            "schema_version",
            "profile",
            "search_space",
            "controllers",
            "proposals",
            "search_seed",
            "scenario_seed_start",
            "near_miss_delta",
            "excluded_scenarios",
        },
    )
    if type(config["proposals"]) is not int or not 1 <= config["proposals"] <= 5:
        raise ValueError("Checkpoint 1 permits at most five differential proposals")
    space = SearchSpace.from_dict(config["search_space"])
    horizon = space.fixed_parameters.get("horizon")
    if type(horizon) is not int or not 1 <= horizon <= 200:
        raise ValueError("Calibration search must fix horizon <=200")
    controllers = _controllers(config["controllers"], max_count=2)
    if len(controllers) != 2:
        raise ValueError("Differential calibration requires exactly two controllers")
    if controllers[0]["id"] == controllers[1]["id"]:
        raise ValueError("Differential calibration requires distinct controller IDs")
    excluded = tuple(_scenario(item) for item in config["excluded_scenarios"])
    protocol = {
        "schema_version": "pump-robustness-protocol:1",
        "profile": config["profile"],
        "search_space": space.to_dict(),
        "controllers": [_controller_record(spec) for spec in controllers],
        "proposals": config["proposals"],
        "search_seed": config["search_seed"],
        "scenario_seed_start": config["scenario_seed_start"],
        "near_miss_delta": config["near_miss_delta"],
        "excluded_scenario_ids": [scenario_identity(s) for s in excluded],
        "limits": {"proposals": 5, "controllers": 2},
    }
    store = RobustnessStore(output_dir)
    run_id = store.open_run(protocol)
    sessions = {}
    started = time.perf_counter()
    rss_before = _rss()

    def evaluate(controller_id: str, scenario: Scenario):
        spec = next(spec for spec in controllers if spec["id"] == controller_id)
        key = controller_identity(spec)
        if key not in sessions:
            sessions[key] = _Session(spec)
        return sessions[key].evaluate(scenario)

    path = store.directory(run_id) / "differential.json"
    try:
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("protocol_sha256") != semantic_hash(protocol):
                raise ValueError("Stored differential protocol changed")
            if result.get("result_sha256") != semantic_hash(result.get("result")):
                raise ValueError("Stored differential evidence hash changed")
            result = result["result"]
            if "durable_headline_selection" not in result:
                result = archive_selected_findings(
                    result, controllers, evaluate, store.directory(run_id), semantic_hash(protocol)
                )
                _atomic_json(
                    path,
                    {
                        "protocol_sha256": semantic_hash(protocol),
                        "result_sha256": semantic_hash(result),
                        "result": result,
                    },
                )
        else:
            result = sample_differentials(
                space,
                controllers[0]["id"],
                controllers[1]["id"],
                evaluate,
                proposals=config["proposals"],
                search_seed=config["search_seed"],
                scenario_seed_start=config["scenario_seed_start"],
                near_miss_delta=config["near_miss_delta"],
                excluded_scenarios=excluded,
            )
            result = archive_selected_findings(
                result, controllers, evaluate, store.directory(run_id), semantic_hash(protocol)
            )
            _atomic_json(
                path,
                {
                    "protocol_sha256": semantic_hash(protocol),
                    "result_sha256": semantic_hash(result),
                    "result": result,
                },
            )
    finally:
        for session in sessions.values():
            session.close()
    margin_results = [
        margin
        for row in result["records"]
        if row["status"] == "evaluated"
        for margin in (row["margin_x"], row["margin_y"])
    ]
    scenarios = [
        Scenario.from_dict(row["scenario"])
        for row in result["records"]
        if row["status"] == "evaluated"
    ]
    seconds = time.perf_counter() - started
    episodes = sum(session.episodes for session in sessions.values())
    transitions = sum(session.transitions for session in sessions.values())
    calibration = {
        "schema_version": "pump-robustness-calibration-result:1",
        "profile": "software_calibration_only",
        "coverage": result["coverage"],
        "measured_seconds_this_invocation": seconds,
        "measured_episodes_this_invocation": episodes,
        "measured_transitions_this_invocation": transitions,
        "episodes_per_second": episodes / seconds if seconds else None,
        "transitions_per_second": transitions / seconds if seconds else None,
        "observed_rss_bytes_start": rss_before,
        "observed_rss_bytes_end": _rss(),
        "projection_note": "No full-study performance inference from differential calibration.",
    }
    return _finish(
        store,
        run_id,
        protocol,
        controllers,
        scenarios,
        margin_results,
        [],
        [result],
        calibration,
        "sirfex robustness differential --config <config> "
        "--output-dir <directory> --mode calibration",
    )
