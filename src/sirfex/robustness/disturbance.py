"""Bounded pump disturbances and observed, grid-based failure tolerance."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import combinations

from sirfex.core import EpisodeResult, Scenario
from sirfex.core.scenarios import ScenarioValidationError
from sirfex.environments.pump import PumpEnvironment
from sirfex.robustness.identity import scenario_identity
from sirfex.robustness.margins import PumpMargin, evaluate_margin
from sirfex.training.feasibility import analyze_feasibility

AXES = ("demand", "capacity", "efficiency", "leak")
DIRECTIONS = tuple(
    tuple(group) for length in range(1, len(AXES) + 1) for group in combinations(AXES, length)
) + (("surplus_demand",),)
SCHEMA = "pump-disturbance-cell:1"


class UndefinedAxis(ValueError):
    """The prescribed transformation has no meaningful reference scale."""


class FeasibilityDiscrepancy(RuntimeError):
    """A completed safe episode contradicts proven finite-horizon infeasibility."""

    def __init__(self, evidence: dict):
        self.evidence = evidence
        super().__init__("Controller survived a scenario proven physically infeasible")


@dataclass(frozen=True)
class GridPoint:
    delta: float
    domain_status: str
    scenario: Scenario | None
    scenario_id: str | None
    feasibility: str | None
    viable: bool | None


@dataclass(frozen=True)
class DisturbanceCell:
    schema_version: str
    controller_id: str
    anchor_id: str
    direction: tuple[str, ...]
    status: str
    delta_first: float | None
    delta_infeasible_grid: float | None
    relative_failure_threshold: float | None
    nonmonotone: bool
    physics_nonmonotone: bool
    grid_outcomes: tuple[dict, ...]
    valid_grid_points: int
    evaluated_grid_points: int
    requested_grid_points: int
    refinement: dict
    units: str = "dimensionless_delta"

    def to_dict(self) -> dict:
        value = asdict(self)
        value["direction"] = list(self.direction)
        value["grid_outcomes"] = list(self.grid_outcomes)
        return value


def validate_direction(direction: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    values = tuple(direction)
    if values not in DIRECTIONS:
        raise ValueError("Expected one of the 16 predefined pump disturbance directions")
    return values


def validate_grid(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 25:
        raise ValueError("Disturbance grid must contain 1..25 values")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        or not 0 <= item <= 0.60
        for item in values
    ):
        raise ValueError("Disturbance deltas must be finite within [0, 0.60]")
    grid = tuple(float(item) for item in values)
    if grid[0] != 0 or any(b <= a for a, b in zip(grid, grid[1:], strict=False)):
        raise ValueError("Grid must start at zero and increase strictly")
    return grid


def transform(anchor: Scenario, direction: tuple[str, ...], delta: float) -> Scenario:
    """Apply the prescribed transform without clamping invalid generated values."""
    direction = validate_direction(direction)
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta):
        raise ValueError("Disturbance delta must be finite")
    if delta < 0:
        raise ValueError("Disturbance delta must be nonnegative")
    p = PumpEnvironment().validate_scenario(anchor)
    if "leak" in direction and p["demand"] == 0 and p["leak"] == 0:
        raise UndefinedAxis("Leak scaling is undefined when both demand and leak are zero")
    changed = dict(p)
    if "surplus_demand" in direction:
        changed["demand"] = p["demand"] * (1 - delta)
    else:
        if "demand" in direction:
            changed["demand"] = p["demand"] * (1 + delta)
        if "capacity" in direction:
            changed["pump_capacity"] = p["pump_capacity"] * (1 - delta)
        if "efficiency" in direction:
            changed["efficiency"] = p["efficiency"] * (1 - delta)
        if "leak" in direction:
            midpoint = (p["min_level"] + p["max_level"]) / 2
            changed["leak"] = p["leak"] + delta * p["demand"] / midpoint
    candidate = Scenario(
        environment_id="pump",
        environment_version=PumpEnvironment.version,
        parameters=changed,
        seed=anchor.seed,
        schema_version=anchor.schema_version,
        provenance={"robustness_direction": "+".join(direction)},
    )
    PumpEnvironment().validate_scenario(candidate)
    return candidate


def prepare_grid(
    anchor: Scenario, direction: tuple[str, ...], grid: list[float] | tuple[float, ...]
) -> tuple[GridPoint, ...]:
    """Compute a shared domain and feasibility sequence before controller evaluation."""
    direction = validate_direction(direction)
    grid = validate_grid(grid)
    p = PumpEnvironment().validate_scenario(anchor)
    if "leak" in direction and p["demand"] == 0 and p["leak"] == 0:
        return tuple(GridPoint(delta, "undefined_axis", None, None, None, None) for delta in grid)
    points = []
    domain_ended = False
    for delta in grid:
        try:
            scenario = transform(anchor, direction, delta)
        except ScenarioValidationError:
            domain_ended = True
            points.append(GridPoint(delta, "censored_bound", None, None, None, None))
            continue
        if domain_ended:
            raise ValueError("Disturbance validity domain unexpectedly reopens")
        analysis = analyze_feasibility(scenario)
        points.append(
            GridPoint(
                delta,
                "valid",
                scenario,
                scenario_identity(scenario),
                analysis["classification"],
                analysis["finite_horizon_viable"],
            )
        )
    return tuple(points)


def _refine_ceiling(
    anchor: Scenario,
    direction: tuple[str, ...],
    points: tuple[GridPoint, ...],
    *,
    physics_nonmonotone: bool,
    iterations: int = 12,
) -> dict:
    if direction == ("surplus_demand",) or physics_nonmonotone:
        return {"status": "not_justified"}
    first = next((i for i, point in enumerate(points) if point.viable is False), None)
    if first is None or first == 0 or points[first - 1].viable is not True:
        return {"status": "not_bracketed"}
    low, high = points[first - 1].delta, points[first].delta
    p = PumpEnvironment().validate_scenario(anchor)
    for edge in (low, high):
        transformed = PumpEnvironment().validate_scenario(transform(anchor, direction, edge))
        if not 0 <= p["dt"] * transformed["leak"] <= 1:
            return {"status": "ceiling_unresolved_domain", "grid_bracket": [low, high]}
    for _ in range(iterations):
        midpoint = (low + high) / 2
        analysis = analyze_feasibility(transform(anchor, direction, midpoint))
        if analysis["finite_horizon_viable"] is None:
            return {"status": "ceiling_unresolved_domain", "grid_bracket": [low, high]}
        if analysis["finite_horizon_viable"]:
            low = midpoint
        else:
            high = midpoint
    return {
        "status": "restricted_monotone_bracket",
        "viable_lower": low,
        "infeasible_upper": high,
        "iterations": iterations,
        "assumptions": "adverse axes, nonnegative tank level, dt*leak <= 1 throughout bracket",
    }


def evaluate_direction(
    anchor: Scenario,
    direction: tuple[str, ...],
    grid: list[float] | tuple[float, ...],
    controller_id: str,
    evaluator: Callable[[Scenario], EpisodeResult | dict],
    *,
    prepared: tuple[GridPoint, ...] | None = None,
    refine_feasibility: bool = False,
) -> DisturbanceCell:
    direction = validate_direction(direction)
    grid = validate_grid(grid)
    points = prepared if prepared is not None else prepare_grid(anchor, direction, grid)
    if len(points) != len(grid) or any(p.delta != d for p, d in zip(points, grid, strict=True)):
        raise ValueError("Prepared shared grid does not match the requested grid")
    anchor_id = scenario_identity(anchor)
    if points[0].domain_status == "undefined_axis":
        return DisturbanceCell(
            SCHEMA,
            controller_id,
            anchor_id,
            direction,
            "undefined_axis",
            None,
            None,
            None,
            False,
            False,
            tuple(
                {"delta": p.delta, "domain_status": "undefined_axis", "failure": None}
                for p in points
            ),
            0,
            0,
            len(points),
            {"status": "not_applicable"},
        )
    rows = []
    seen_failure = False
    nonmonotone = False
    seen_infeasible = False
    physics_nonmonotone = False
    first_failure = None
    first_infeasible = None
    valid_count = 0
    for point in points:
        if point.domain_status != "valid":
            rows.append(
                {
                    "delta": point.delta,
                    "domain_status": point.domain_status,
                    "failure": None,
                    "feasibility": None,
                }
            )
            continue
        valid_count += 1
        if point.viable is False:
            seen_infeasible = True
            if first_infeasible is None:
                first_infeasible = point.delta
        elif point.viable is True and seen_infeasible:
            physics_nonmonotone = True
        result = evaluator(point.scenario)
        margin: PumpMargin = evaluate_margin(result)
        if margin.controller_id != controller_id or margin.scenario_id != point.scenario_id:
            raise ValueError("Controller or scenario identity changed during evaluation")
        if margin.evidence_status != "valid":
            raise ValueError("Partial episode cannot establish disturbance survival")
        failure = margin.constraint_violated
        if seen_failure and not failure:
            nonmonotone = True
        if failure:
            seen_failure = True
            if first_failure is None:
                first_failure = point.delta
        row = {
            "delta": point.delta,
            "domain_status": "valid",
            "scenario_id": point.scenario_id,
            "feasibility": point.feasibility,
            "failure": failure,
            "post_initial_margin": margin.post_initial_minimum_margin,
            "full_episode_margin": margin.full_episode_minimum_margin,
            "violation_magnitude": margin.maximum_violation_magnitude,
        }
        rows.append(row)
        if point.viable is False and not failure:
            raise FeasibilityDiscrepancy(
                {
                    "controller_id": controller_id,
                    "anchor_id": anchor_id,
                    "direction": list(direction),
                    "delta": point.delta,
                    "scenario_id": point.scenario_id,
                    "feasibility": point.feasibility,
                    "margin": margin.to_dict(),
                    "grid_prefix": rows,
                }
            )
    status = (
        "fails_at_nominal"
        if first_failure == 0
        else "failed"
        if first_failure is not None
        else "censored_bound"
        if valid_count < len(points)
        else "censored_max"
    )
    relative = (
        first_failure / first_infeasible
        if first_failure is not None and first_infeasible is not None and first_infeasible > 0
        else None
    )
    refinement = (
        _refine_ceiling(anchor, direction, points, physics_nonmonotone=physics_nonmonotone)
        if refine_feasibility
        else {"status": "not_requested"}
    )
    return DisturbanceCell(
        SCHEMA,
        controller_id,
        anchor_id,
        direction,
        status,
        first_failure,
        first_infeasible,
        relative,
        nonmonotone,
        physics_nonmonotone,
        tuple(rows),
        valid_count,
        valid_count,
        len(points),
        refinement,
    )
