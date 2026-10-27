"""Gymnasium training adapter over the existing pump dynamics."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from sirfex.core.interfaces import Scenario
from sirfex.environments.pump_observations import MINIMAL, PumpObservationEnvironment
from sirfex.training.actions import DIRECT, policy_action_space, to_physical
from sirfex.training.diagnostics import reward_components
from sirfex.training.exposure import ExposureLedger
from sirfex.training.protocol import scenario_record
from sirfex.training.rewards import LEGACY, TERMINATION, components, reward_spec


class PumpGymEnv(gym.Env[np.ndarray, np.ndarray]):
    """Expose PumpEnvironment through Gymnasium without duplicating physics."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenarios: list[Scenario] | None = None,
        *,
        seed: int = 0,
        violation_weight: float = 10.0,
        observation_schema: str = MINIMAL,
        stress_scenarios: list[Scenario] | None = None,
        stress_ratio: float = 0.0,
        policy_action_schema: str = DIRECT,
        reward_schema: str = LEGACY,
        termination_schema: str = TERMINATION,
        discovery_ids: dict[str, str] | None = None,
        stress_target_fraction: float | None = None,
    ) -> None:
        super().__init__()
        self._adapter = PumpObservationEnvironment(observation_schema)
        self.observation_space = self._adapter.observation_space
        self.policy_action_schema = policy_action_schema
        self.action_space = policy_action_space(policy_action_schema)
        self.reward_schema, self.termination_schema = reward_schema, termination_schema
        reward_spec(reward_schema)
        if termination_schema != TERMINATION:
            raise ValueError("Post-violation continuation is not a supported physical contract")
        if reward_schema != LEGACY and violation_weight != 10.0:
            raise ValueError("Versioned revised reward has fixed coefficients")
        self.scenarios = list([Scenario("pump")] if scenarios is None else scenarios)
        if not self.scenarios:
            raise ValueError("At least one pump scenario is required")
        if any(item.environment_id != "pump" for item in self.scenarios):
            raise ValueError("PumpGymEnv accepts pump scenarios only")
        self._rng = np.random.default_rng(seed)
        self.violation_weight = float(violation_weight)
        self._scenario = self.scenarios[0]
        self.stress_scenarios = list(stress_scenarios or [])
        if not 0 <= stress_ratio <= 1 or (stress_ratio > 0 and not self.stress_scenarios):
            raise ValueError("Positive stress ratio requires stress scenarios and ratio <= 1")
        if stress_target_fraction is not None:
            if not 0 <= stress_target_fraction <= 1 or (
                stress_target_fraction > 0 and not self.stress_scenarios
            ):
                raise ValueError("Exposure target must be in [0,1] with a stress pool")
            if stress_ratio != 0:
                raise ValueError("Use episode probability or interaction target, not both")
        self.stress_ratio = stress_ratio
        self.stress_target_fraction = stress_target_fraction
        self._normal_scenario_ids = {scenario_record(s)["scenario_id"] for s in self.scenarios}
        self._stress_scenario_ids = {
            scenario_record(s)["scenario_id"] for s in self.stress_scenarios
        }
        self.discovery_ids = dict(discovery_ids or {})
        if set(self.discovery_ids) - self._stress_scenario_ids:
            raise ValueError("Discovery mapping contains a scenario outside the stress pool")
        self._exposure = ExposureLedger(seed)
        self.sample_counts = {"normal": 0, "stress": 0}
        self.sample_counts_by_scenario_id: dict[str, int] = {}
        for scenario in self.scenarios + self.stress_scenarios:
            self._adapter.validate_scenario(scenario)
        reward_components(0, 0, 0, 0, self.violation_weight)

    @property
    def current_scenario(self) -> Scenario:
        return self._scenario

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._exposure.explicit_reset_seeds.append(seed)
        if options and "scenario" in options:
            candidate = options["scenario"]
            self._scenario = (
                candidate if isinstance(candidate, Scenario) else Scenario.from_dict(candidate)
            )
            key = scenario_record(self._scenario)["scenario_id"]
            mode = (
                "stress"
                if key in self._stress_scenario_ids and key not in self._normal_scenario_ids
                else "normal"
                if key in self._normal_scenario_ids and key not in self._stress_scenario_ids
                else "unassigned"
            )
        else:
            if self.stress_target_fraction is None:
                stress = self.stress_ratio > 0 and self._rng.random() < self.stress_ratio
            else:
                total = sum(self._exposure.interactions.values())
                stress = (
                    self.stress_target_fraction > 0
                    and total > 0
                    and self._exposure.interactions["stress"] < self.stress_target_fraction * total
                )
            pool = self.stress_scenarios if stress else self.scenarios
            self._scenario = pool[int(self._rng.integers(len(pool)))]
            mode = "stress" if stress else "normal"
            key = scenario_record(self._scenario)["scenario_id"]
        observation, info = self._adapter.reset(self._scenario)
        if not (options and "scenario" in options):
            self.sample_counts[mode] += 1
            self.sample_counts_by_scenario_id[key] = (
                self.sample_counts_by_scenario_id.get(key, 0) + 1
            )
        self._exposure.start(mode, key, self.discovery_ids.get(key))
        return observation, {**info, "scenario": self._scenario.to_dict()}

    def step(self, action: np.ndarray):
        physical = to_physical(action, self.policy_action_schema)
        observation, reward, terminated, truncated, info = self._adapter.step(physical)
        self._exposure.transition(terminated, truncated)
        violation = float(info.get("violation_magnitude", 0.0))
        shaped_reward = float(reward - self.violation_weight * violation)
        parts = reward_components(
            info["level"],
            self._adapter._active_parameters.target_level,
            float(physical[0]),
            violation,
            self.violation_weight,
        )
        if self.reward_schema != LEGACY:
            parts = components(
                self.reward_schema,
                info["level"],
                self._adapter._active_parameters.target_level,
                float(physical[0]),
                violation,
                terminated,
            )
            shaped_reward = parts["total"]
        return (
            observation,
            shaped_reward,
            terminated,
            truncated,
            {
                **info,
                "base_reward": reward,
                "reward_components": parts,
                "policy_action": np.asarray(action).tolist(),
                "physical_action": physical.tolist(),
                "policy_action_schema": self.policy_action_schema,
                "termination_schema": self.termination_schema,
            },
        )

    def close(self) -> None:
        self._adapter.close()

    def exposure_summary(
        self,
        *,
        scheduled_interactions: int | None = None,
        model_interactions: int | None = None,
    ) -> dict:
        summary = self._exposure.snapshot(
            scheduled_interactions=scheduled_interactions,
            model_interactions=model_interactions,
        )
        summary["sampling_mode"] = (
            "interaction-debt:1"
            if self.stress_target_fraction is not None
            else "episode-probability:1"
        )
        summary["configured_stress_probability"] = (
            self.stress_ratio if self.stress_target_fraction is None else None
        )
        summary["target_stress_interaction_fraction"] = self.stress_target_fraction
        summary["sampler_episode_starts"] = dict(self.sample_counts)
        summary["sampler_start_reconciles"] = (
            summary["nominal_episode_starts"] == self.sample_counts["normal"]
            and summary["stress_episode_starts"] == self.sample_counts["stress"]
        )
        summary["reconciliation_valid"] = (
            summary["reconciliation_valid"] and summary["sampler_start_reconciles"]
        )
        return summary
