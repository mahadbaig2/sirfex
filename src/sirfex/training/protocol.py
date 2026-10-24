"""Frozen dataset identities, competence gates and paired study interpretation."""

import hashlib
import json
from itertools import combinations

import numpy as np

from sirfex.core import Scenario
from sirfex.environments.pump import PumpEnvironment
from sirfex.training.feasibility import analyze_feasibility

GATE = {
    "maximum_violation_frequency": 0.0,
    "maximum_tracking_error_mean": 0.10,
    "maximum_execution_failures": 0,
    "require_full_horizons": True,
}


def content_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def physical_key(scenario):
    p = PumpEnvironment().validate_scenario(scenario)
    return {k: round(float(v), 5) for k, v in sorted(p.items())}


def scenario_record(scenario):
    return {"scenario_id": content_hash(physical_key(scenario)), "scenario": scenario.to_dict()}


def leakage_check(partitions):
    """Exclude seed, provenance and horizon from physical near-duplicate detection."""
    issues = []
    for (name_a, group_a), (name_b, group_b) in combinations(partitions.items(), 2):
        for a in group_a:
            for b in group_b:
                pa, pb = physical_key(a), physical_key(b)
                same = all(abs(pa[k] - pb[k]) <= 1e-4 for k in pa if k != "horizon")
                if same or a.seed == b.seed:
                    issues.append(
                        {
                            "first": name_a,
                            "second": name_b,
                            "physical_overlap": same,
                            "seed_overlap": a.seed == b.seed,
                        }
                    )
    return {
        "passed": not issues,
        "issues": issues,
        "rule": "all physical parameters except horizon within absolute 1e-4; seeds disjoint",
    }


def datasets(seed=3101, count=12):
    """Unseen physical parameters, constant dt/bounds/horizon; no seed-only replication."""
    rng = np.random.default_rng(seed)
    result = {}
    for index, name in enumerate(("training", "validation", "nominal_holdout", "held_out_stress")):
        group = []
        for j in range(count):
            stress = name == "held_out_stress"
            capacity, efficiency = float(rng.uniform(0.75, 1.0)), float(rng.uniform(0.8, 1.0))
            leak = float(rng.uniform(0, 0.08 if stress else 0.03))
            demand = float(rng.uniform(0.75, 0.90) if stress else rng.uniform(0.45, 0.65))
            demand = demand * capacity * efficiency - leak * 0.5
            scenario = Scenario(
                "pump",
                {
                    "dt": 0.1,
                    "horizon": 200,
                    "min_level": 0.05,
                    "max_level": 0.95,
                    "initial_level": float(rng.uniform(0.3, 0.7)),
                    "target_level": float(rng.uniform(0.4, 0.6)),
                    "pump_capacity": capacity,
                    "efficiency": efficiency,
                    "demand": demand,
                    "leak": leak,
                },
                seed=100000 + index * 10000 + j,
                environment_version="sirfex-pump:1",
                provenance={"partition": name, "generator": "pump-partitions:1"},
            )
            if analyze_feasibility(scenario)["classification"] != "target_feasible":
                raise ValueError("Protocol generated a non-target-feasible scenario")
            group.append(scenario)
        result[name] = group
    if not leakage_check(result)["passed"]:
        raise ValueError("Generated partitions overlap")
    return result


def competence(summary, gate=None):
    gate = gate or GATE
    checks = {
        "evidence": summary.get("episodes", 0) > 0,
        "execution": summary.get("execution_failures") == gate["maximum_execution_failures"],
        "physical": summary.get("constraint_violation_frequency") is not None
        and summary["constraint_violation_frequency"] <= gate["maximum_violation_frequency"],
        "tracking": summary.get("mean_tracking_error") is not None
        and summary["mean_tracking_error"] <= gate["maximum_tracking_error_mean"],
        "horizons": summary.get("full_horizon_frequency") == 1.0,
    }
    return {"passed": all(checks.values()), "checks": checks, "thresholds": gate}


def paired_interval(differences, seed=3101):
    values = np.asarray(differences, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite paired outcome")
    result = {
        "independent_training_seeds": len(values),
        "paired_differences": values.tolist(),
        "mean": float(values.mean()) if len(values) else None,
        "interval_95": None,
    }
    if len(values) >= 3:
        rng = np.random.default_rng(seed)
        means = rng.choice(values, (5000, len(values)), replace=True).mean(axis=1)
        result["interval_95"] = np.quantile(means, [0.025, 0.975]).tolist()
    result["method"] = "paired training-seed percentile bootstrap; <3 seeds: interval unavailable"
    return result


def compare_paired(pairs, tracking_tolerance=0.10):
    if not pairs:
        return {"classification": "inconclusive", "reason": "No matched continuations"}
    seeds = [pair.get("seed") for pair in pairs]
    if None in seeds or len(set(seeds)) != len(seeds):
        return {"classification": "inconclusive", "reason": "Missing or duplicate training seeds"}
    violations, magnitudes, guardrails = [], [], []
    for pair in pairs:
        normal, robust = pair["normal"], pair["robust"]
        for groups in (normal, robust):
            if any(
                v.get("episodes", 0) == 0 or v.get("execution_failures", 0) for v in groups.values()
            ):
                return {"classification": "inconclusive", "reason": "Incomplete execution evidence"}
        n, r = normal["nominal_holdout"], robust["nominal_holdout"]
        guardrails.append(
            r["mean_tracking_error"] <= n["mean_tracking_error"] * (1 + tracking_tolerance)
            and r["constraint_violation_frequency"] <= n["constraint_violation_frequency"]
        )
        n, r = normal["held_out_stress"], robust["held_out_stress"]
        violations.append(r["constraint_violation_frequency"] - n["constraint_violation_frequency"])
        magnitudes.append(r["maximum_violation_magnitude"] - n["maximum_violation_magnitude"])
    vi, mi = paired_interval(violations), paired_interval(magnitudes)
    better = any(x["interval_95"] is not None and x["interval_95"][1] < 0 for x in (vi, mi))
    worse = any(x["interval_95"] is not None and x["interval_95"][0] > 0 for x in (vi, mi))
    classification = "inconclusive"
    if len(pairs) >= 3:
        if not all(guardrails) or (worse and not better):
            classification = "regressed"
        elif better and not worse and max(violations) <= 0 and max(magnitudes) <= 0:
            classification = "improved"
    return {
        "classification": classification,
        "nominal_guardrails": guardrails,
        "held_out_frequency": vi,
        "held_out_magnitude": mi,
        "tracking_tolerance": tracking_tolerance,
        "preliminary": len(pairs) < 3,
    }


def matched_budget(first, second):
    fields = (
        "starting_sha256",
        "observation_schema",
        "reward_schema",
        "ppo_config",
        "additional_timesteps",
        "continuation_seed",
        "evaluation_hash",
    )
    mismatches = [key for key in fields if key not in first or first.get(key) != second.get(key)]
    if first.get("status") != "completed" or second.get("status") != "completed":
        mismatches.append("completion")
    return {"matched": not mismatches, "mismatches": mismatches}
