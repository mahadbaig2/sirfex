"""Exact single-environment curriculum exposure measured at physical transitions."""

from __future__ import annotations

import statistics

DEFAULT_GATE = {
    "stress_completed_episodes": 10,
    "stress_interactions": 1000,
    "distinct_discoveries": 3,
}


def exposure_adequacy(
    exposure: dict | None,
    pool_size: int,
    *,
    thresholds: dict | None = None,
    trial_complete: bool = True,
) -> dict:
    """Check a stress curriculum's actual, reconciled physical exposure."""
    limits = dict(DEFAULT_GATE if thresholds is None else thresholds)
    if pool_size < 0 or any(type(v) is not int or v < 0 for v in limits.values()):
        raise ValueError("Invalid exposure gate")
    required = (
        "actual_environment_interactions",
        "model_interactions",
        "nominal_interactions",
        "stress_interactions",
        "stress_completed_episodes",
        "distinct_discoveries",
        "reconciliation_valid",
        "invalid_assignments",
        "invalid_resets",
        "unassigned_interactions",
    )
    if (
        not trial_complete
        or not isinstance(exposure, dict)
        or any(key not in exposure for key in required)
        or not exposure["reconciliation_valid"]
        or any(
            exposure[key]
            for key in ("invalid_assignments", "invalid_resets", "unassigned_interactions")
        )
        or exposure["nominal_interactions"] + exposure["stress_interactions"]
        != exposure["actual_environment_interactions"]
        or exposure["model_interactions"] != exposure["actual_environment_interactions"]
    ):
        return {"status": "invalid_exposure", "checks": {}, "thresholds": limits}
    checks = {
        "stress_completed_episodes": exposure["stress_completed_episodes"]
        >= limits["stress_completed_episodes"],
        "stress_interactions": exposure["stress_interactions"] >= limits["stress_interactions"],
        "distinct_discoveries": pool_size < 3
        or exposure["distinct_discoveries"] >= limits["distinct_discoveries"],
    }
    return {
        "status": "adequate_exposure" if all(checks.values()) else "insufficient_exposure",
        "checks": checks,
        "thresholds": limits,
    }


class ExposureLedger:
    """Track episode starts, completed episodes and partial transition exposure."""

    def __init__(self, sampler_seed: int) -> None:
        self.sampler_seed = sampler_seed
        self.explicit_reset_seeds: list[int] = []
        self.starts = {"normal": 0, "stress": 0}
        self.completed = {"normal": 0, "stress": 0}
        self.interactions = {"normal": 0, "stress": 0}
        self.partial_interrupted = {"normal": 0, "stress": 0}
        self.by_scenario_starts: dict[str, int] = {}
        self.by_scenario_interactions: dict[str, int] = {}
        self.by_discovery_starts: dict[str, int] = {}
        self.by_discovery_interactions: dict[str, int] = {}
        self.stress_scenario_ids: set[str] = set()
        self.stress_lengths: list[int] = []
        self.nominal_lengths: list[int] = []
        self.invalid_assignments = 0
        self.invalid_resets = 0
        self.unassigned_interactions = 0
        self.active: dict | None = None

    def start(self, mode: str, scenario_id: str, discovery_id: str | None) -> None:
        if self.active is not None:
            self.invalid_resets += 1
            old = self.active
            if old["mode"] in self.partial_interrupted:
                self.partial_interrupted[old["mode"]] += old["length"]
        if mode not in self.starts:
            self.invalid_assignments += 1
        else:
            self.starts[mode] += 1
            self.by_scenario_starts[scenario_id] = self.by_scenario_starts.get(scenario_id, 0) + 1
            if mode == "stress":
                self.stress_scenario_ids.add(scenario_id)
                key = discovery_id or scenario_id
                self.by_discovery_starts[key] = self.by_discovery_starts.get(key, 0) + 1
        self.active = {
            "mode": mode,
            "scenario_id": scenario_id,
            "discovery_id": (discovery_id or scenario_id) if mode == "stress" else None,
            "length": 0,
        }

    def transition(self, terminated: bool, truncated: bool) -> None:
        if self.active is None:
            raise RuntimeError("Curriculum transition has no assigned episode")
        active = self.active
        active["length"] += 1
        mode = active["mode"]
        if mode in self.interactions:
            self.interactions[mode] += 1
            scenario_id = active["scenario_id"]
            self.by_scenario_interactions[scenario_id] = (
                self.by_scenario_interactions.get(scenario_id, 0) + 1
            )
            if mode == "stress":
                key = active["discovery_id"]
                self.by_discovery_interactions[key] = self.by_discovery_interactions.get(key, 0) + 1
        else:
            self.unassigned_interactions += 1
        if terminated or truncated:
            if mode in self.completed:
                self.completed[mode] += 1
                if mode == "stress":
                    self.stress_lengths.append(active["length"])
                else:
                    self.nominal_lengths.append(active["length"])
            self.active = None

    def snapshot(
        self, *, scheduled_interactions: int | None = None, model_interactions: int | None = None
    ) -> dict:
        total = sum(self.interactions.values()) + self.unassigned_interactions
        partial = dict(self.partial_interrupted)
        if self.active is not None and self.active["mode"] in partial:
            partial[self.active["mode"]] += self.active["length"]
        starts = sum(self.starts.values())
        valid = (
            not self.invalid_assignments
            and not self.invalid_resets
            and not self.unassigned_interactions
            and sum(self.interactions.values()) == total
            and (model_interactions is None or model_interactions == total)
            and (scheduled_interactions is None or total <= scheduled_interactions)
            and sum(self.stress_lengths) + partial["stress"] == self.interactions["stress"]
            and sum(self.nominal_lengths) + partial["normal"] == self.interactions["normal"]
            and sum(self.by_discovery_interactions.values()) == self.interactions["stress"]
            and sum(self.by_scenario_interactions.values()) == total
        )
        return {
            "schema_version": "pump-curriculum-exposure:1",
            "sampler_seed": self.sampler_seed,
            "explicit_reset_seeds": list(self.explicit_reset_seeds),
            "scheduled_interactions": scheduled_interactions,
            "model_interactions": model_interactions,
            "actual_environment_interactions": total,
            "episode_starts": starts,
            "nominal_episode_starts": self.starts["normal"],
            "stress_episode_starts": self.starts["stress"],
            "nominal_completed_episodes": self.completed["normal"],
            "stress_completed_episodes": self.completed["stress"],
            "partial_nominal_interactions": partial["normal"],
            "partial_stress_interactions": partial["stress"],
            "nominal_interactions": self.interactions["normal"],
            "stress_interactions": self.interactions["stress"],
            "stress_interaction_fraction": self.interactions["stress"] / total if total else 0.0,
            "stress_episode_start_fraction": self.starts["stress"] / starts if starts else 0.0,
            "per_scenario_episode_starts": dict(self.by_scenario_starts),
            "per_scenario_interactions": dict(self.by_scenario_interactions),
            "per_discovery_episode_starts": dict(self.by_discovery_starts),
            "per_discovery_interactions": dict(self.by_discovery_interactions),
            "distinct_stress_scenarios": len(self.stress_scenario_ids),
            "distinct_discoveries": len(self.by_discovery_starts),
            "completed_stress_episode_lengths": list(self.stress_lengths),
            "completed_nominal_episode_lengths": list(self.nominal_lengths),
            "mean_completed_stress_episode_length": (
                statistics.mean(self.stress_lengths) if self.stress_lengths else None
            ),
            "minimum_completed_stress_episode_length": (
                min(self.stress_lengths) if self.stress_lengths else None
            ),
            "maximum_completed_stress_episode_length": (
                max(self.stress_lengths) if self.stress_lengths else None
            ),
            "active_episode": dict(self.active) if self.active is not None else None,
            "invalid_assignments": self.invalid_assignments,
            "invalid_resets": self.invalid_resets,
            "unassigned_interactions": self.unassigned_interactions,
            "reconciliation_valid": valid,
        }
