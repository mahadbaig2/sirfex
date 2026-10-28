"""Same-scenario controller differences with distinct failure and near-miss classes."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from sirfex.core import EpisodeResult, Scenario
from sirfex.experiments.replay import compare_results
from sirfex.robustness.identity import scenario_identity
from sirfex.robustness.margins import PumpMargin, evaluate_margin
from sirfex.search.space import SearchSpace
from sirfex.training.feasibility import analyze_feasibility
from sirfex.training.protocol import leakage_check

SCHEMA = "pump-differential:1"


@dataclass(frozen=True)
class DifferentialFinding:
    schema_version: str
    classification: str
    scenario_id: str
    controller_x: str
    controller_y: str
    margin_x_post: float
    margin_y_post: float
    signed_margin_gap: float
    x_violated: bool
    y_violated: bool
    finite_horizon_viable: bool | None
    replay_statuses: tuple[str, ...]
    reason: str
    units: str = "normalized_tank_level_fraction"

    def to_dict(self) -> dict:
        value = asdict(self)
        value["replay_statuses"] = list(self.replay_statuses)
        return value


def classify_pair(
    x: PumpMargin,
    y: PumpMargin,
    *,
    finite_horizon_viable: bool | None,
    near_miss_delta: float = 0.05,
    replay_statuses: tuple[str, str] | None = None,
) -> DifferentialFinding:
    if (
        not math.isfinite(near_miss_delta)
        or near_miss_delta <= 0
        or x.scenario_id != y.scenario_id
        or x.environment_id != "pump"
        or y.environment_id != "pump"
        or x.evidence_status != "valid"
        or y.evidence_status != "valid"
    ):
        raise ValueError("Differential comparison requires matching complete pump evidence")
    gap = y.post_initial_minimum_margin - x.post_initial_minimum_margin
    replay = tuple(replay_statuses or ())
    confirmed = bool(replay) and all(item in ("exact_match", "tolerance_match") for item in replay)
    if x.constraint_violated and not y.constraint_violated and finite_horizon_viable is True:
        classification = "class_b" if confirmed else "class_b_unverified"
        reason = (
            "X violated while Y survived a viable scenario; both deterministic repeats matched"
            if confirmed
            else "X violated while Y survived, but replay is not yet verified"
        )
    elif (
        not x.constraint_violated
        and not y.constraint_violated
        and finite_horizon_viable is True
        and gap >= near_miss_delta
    ):
        classification = "class_a"
        reason = "Both survived; X had a materially smaller post-initial margin"
    else:
        classification = "none"
        reason = "No qualifying, physically viable differential finding"
    return DifferentialFinding(
        SCHEMA,
        classification,
        x.scenario_id,
        x.controller_id,
        y.controller_id,
        x.post_initial_minimum_margin,
        y.post_initial_minimum_margin,
        gap,
        x.constraint_violated,
        y.constraint_violated,
        finite_horizon_viable,
        replay,
        reason,
    )


def rank_findings(findings: list[DifferentialFinding]) -> list[DifferentialFinding]:
    """Class B outranks A regardless of numerical margin-gap magnitude."""
    priority = {"class_b": 0, "class_a": 1, "class_b_unverified": 2, "none": 3}
    return sorted(
        findings,
        key=lambda item: (
            priority[item.classification],
            -item.signed_margin_gap,
            item.scenario_id,
        ),
    )


def sample_differentials(
    space: SearchSpace,
    controller_x: str,
    controller_y: str,
    evaluator: Callable[[str, Scenario], EpisodeResult],
    *,
    proposals: int,
    search_seed: int,
    scenario_seed_start: int,
    near_miss_delta: float = 0.05,
    excluded_scenarios: tuple[Scenario, ...] = (),
    top_per_class: int = 5,
) -> dict:
    """Seeded nonadaptive sampling using existing search-space and exclusion rules."""
    space.validate()
    if space.environment_id != "pump" or space.environment_version != "sirfex-pump:1":
        raise ValueError("Differential sampling requires the pump search space")
    if type(proposals) is not int or not 1 <= proposals <= 200:
        raise ValueError("Proposal count must be in 1..200")
    if type(search_seed) is not int or type(scenario_seed_start) is not int:
        raise ValueError("Seeds must be integers")
    if type(top_per_class) is not int or not 1 <= top_per_class <= 5:
        raise ValueError("Top-per-class must be in 1..5")
    rng = np.random.default_rng(search_seed)
    accepted: list[Scenario] = []
    findings: list[DifferentialFinding] = []
    rows = []
    for index in range(proposals):
        params = space.sample_parameters(rng)
        scenario = space.scenario(params, scenario_seed_start + index)
        scenario_id = scenario_identity(scenario)
        separate = leakage_check(
            {"excluded": excluded_scenarios + tuple(accepted), "candidate": [scenario]}
        )["passed"]
        if not separate:
            rows.append(
                {"proposal_index": index, "scenario_id": scenario_id, "status": "excluded_overlap"}
            )
            continue
        feasible = analyze_feasibility(scenario)
        if feasible["finite_horizon_viable"] is not True:
            rows.append(
                {
                    "proposal_index": index,
                    "scenario_id": scenario_id,
                    "status": "excluded_feasibility",
                    "feasibility": feasible["classification"],
                }
            )
            continue
        accepted.append(scenario)
        first = evaluator(controller_x, scenario)
        second = evaluator(controller_y, scenario)
        mx, my = evaluate_margin(first), evaluate_margin(second)
        if mx.scenario_id != scenario_id or my.scenario_id != scenario_id:
            raise ValueError("Evaluator changed the sampled scenario")
        initial = classify_pair(
            mx,
            my,
            finite_horizon_viable=True,
            near_miss_delta=near_miss_delta,
        )
        replay = None
        if initial.classification == "class_b_unverified":
            first_repeat = evaluator(controller_x, scenario)
            second_repeat = evaluator(controller_y, scenario)
            replay = (
                compare_results(first.to_dict(), first_repeat.to_dict()).status,
                compare_results(second.to_dict(), second_repeat.to_dict()).status,
            )
        finding = classify_pair(
            mx,
            my,
            finite_horizon_viable=True,
            near_miss_delta=near_miss_delta,
            replay_statuses=replay,
        )
        findings.append(finding)
        rows.append(
            {
                "proposal_index": index,
                "scenario_id": scenario_id,
                "scenario": scenario.to_dict(),
                "status": "evaluated",
                "feasibility": feasible["classification"],
                "finding": finding.to_dict(),
                "margin_x": mx.to_dict(),
                "margin_y": my.to_dict(),
            }
        )
    ranked = rank_findings(findings)
    return {
        "schema_version": "pump-differential-search:1",
        "method": "seeded_random_nonadaptive",
        "controller_x": controller_x,
        "controller_y": controller_y,
        "proposal_count": proposals,
        "search_seed": search_seed,
        "scenario_seed_start": scenario_seed_start,
        "near_miss_delta": near_miss_delta,
        "coverage": {
            "proposed": proposals,
            "evaluated": sum(row["status"] == "evaluated" for row in rows),
            "excluded_overlap": sum(row["status"] == "excluded_overlap" for row in rows),
            "excluded_feasibility": sum(row["status"] == "excluded_feasibility" for row in rows),
        },
        "findings": {
            "class_b": [item.to_dict() for item in ranked if item.classification == "class_b"][
                :top_per_class
            ],
            "class_a": [item.to_dict() for item in ranked if item.classification == "class_a"][
                :top_per_class
            ],
            "class_b_unverified": [
                item.to_dict() for item in ranked if item.classification == "class_b_unverified"
            ],
        },
        "records": rows,
    }
