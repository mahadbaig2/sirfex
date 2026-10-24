"""Search orchestration over the existing evaluation and experiment layers."""

from __future__ import annotations

import importlib.metadata
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sirfex import __version__
from sirfex.core import EvaluationError, ScenarioValidationError
from sirfex.experiments.factories import make_environment
from sirfex.experiments.replay import replay_experiment
from sirfex.experiments.service import run_and_save_experiment
from sirfex.experiments.storage import ExperimentStore
from sirfex.search.objectives import RiskObjective, RiskScore
from sirfex.search.persistence import SearchStore
from sirfex.search.ranking import rank_discoveries, similarity_key
from sirfex.search.records import DiscoveryRecord, TrialRecord, aggregate_scores
from sirfex.search.space import SearchSpace
from sirfex.search.strategies import RandomSearchStrategy, SearchStrategy, TPESearchStrategy


def _dependencies() -> dict[str, str]:
    versions = {}
    for package in ("sirfex", "numpy", "gymnasium", "optuna"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


class SearchRunner:
    def __init__(
        self,
        space: SearchSpace,
        objective: RiskObjective,
        controller_id: str,
        *,
        experiment_store: ExperimentStore | None = None,
        search_store: SearchStore | None = None,
        max_steps: int = 1_000,
    ) -> None:
        space.validate()
        self.space = space
        self.objective = objective
        self.controller_id = controller_id
        self.experiment_store = experiment_store or ExperimentStore()
        self.search_store = search_store or SearchStore()
        self.max_steps = max_steps

    def _aggregate(self, scores: list[RiskScore]) -> dict[str, Any]:
        return aggregate_scores(scores)

    def run(
        self,
        strategy_name: str,
        *,
        trials: int,
        strategy_seed: int = 0,
        scenario_seeds: list[int] | None = None,
        held_out_seeds: list[int] | None = None,
    ) -> dict[str, Any]:
        if trials < 1:
            raise ValueError("trials must be positive")
        seeds = scenario_seeds or [0]
        if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise ValueError("scenario_seeds must contain integers")
        strategy: SearchStrategy
        if strategy_name == "random":
            strategy = RandomSearchStrategy(strategy_seed)
        elif strategy_name == "tpe":
            strategy = TPESearchStrategy(strategy_seed, self.objective.direction)
        else:
            raise ValueError(f"Unsupported search strategy: {strategy_name}")
        environment = make_environment(self.space.environment_id)
        search_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:10]}"
        started = time.perf_counter()
        trial_records: list[TrialRecord] = []
        discoveries: list[DiscoveryRecord] = []
        best_history: list[dict[str, Any]] = []
        best_value: float | None = None
        actual_evaluations = 0
        try:
            for trial_index in range(trials):
                trial_started = time.perf_counter()
                parameters = strategy.propose(self.space)
                trial = TrialRecord(trial_index, "pending", parameters, list(seeds))
                try:
                    template = self.space.scenario(parameters, seeds[0])
                    environment.validate_scenario(template)
                except (ScenarioValidationError, ValueError) as exc:
                    trial.status = "invalid"
                    trial.error = {"type": type(exc).__name__, "message": str(exc)}
                    strategy.observe(None, failed=True)
                    trial.elapsed_seconds = time.perf_counter() - trial_started
                    trial_records.append(trial)
                    continue

                scores: list[RiskScore] = []
                execution_failed = False
                for seed in seeds:
                    scenario = self.space.scenario(parameters, seed)
                    try:
                        manifest = run_and_save_experiment(
                            self.experiment_store,
                            scenario,
                            self.controller_id,
                            max_steps=self.max_steps,
                        )
                        trial.experiment_ids.append(manifest.experiment_id)
                        loaded = self.experiment_store.load(manifest.experiment_id)
                        from sirfex.core.results import EpisodeResult

                        result = EpisodeResult(
                            environment_id=loaded.result["environment_id"],
                            controller_id=loaded.result["controller_id"],
                            scenario=loaded.result["scenario"],
                            seed=loaded.result["seed"],
                            episode_reward=loaded.result["episode_reward"],
                            episode_length=loaded.result["episode_length"],
                            terminated=loaded.result["terminated"],
                            truncated=loaded.result["truncated"],
                            outcome=loaded.result["outcome"],
                            metrics=loaded.result.get("metrics", {}),
                            trace=loaded.trajectory,
                            observed_outcome=loaded.result.get("observed_outcome", "unknown"),
                        )
                        score = self.objective.evaluate(result)
                        if not isinstance(score.value, (int, float)) or not float(
                            score.value
                        ) == float(score.value):
                            raise ValueError("Objective produced a non-finite value")
                        scores.append(score)
                        actual_evaluations += 1
                    except Exception as exc:
                        execution_failed = True
                        trial.status = "execution_failed"
                        trial.error = {"type": type(exc).__name__, "message": str(exc)}
                        if isinstance(exc, EvaluationError) and exc.experiment_id:
                            trial.experiment_ids.append(exc.experiment_id)
                        break
                trial.elapsed_seconds = time.perf_counter() - trial_started
                if execution_failed or not scores:
                    strategy.observe(None, failed=True)
                    trial_records.append(trial)
                    continue
                trial.status = "completed"
                trial.risk_scores = [score.to_dict() for score in scores]
                trial.aggregate = self._aggregate(scores)
                strategy.observe(trial.aggregate["mean"])
                trial_records.append(trial)
                value = trial.aggregate["mean"]
                if (
                    best_value is None
                    or (self.objective.direction == "maximize" and value > best_value)
                    or (self.objective.direction == "minimize" and value < best_value)
                ):
                    best_value = value
                best_history.append({"trial_index": trial_index, "value": best_value})
                if any(score.is_failure for score in scores):
                    discovery = DiscoveryRecord(
                        discovery_id=f"disc_{search_id[-10:]}_{len(discoveries):04d}",
                        search_id=search_id,
                        environment_id=self.space.environment_id,
                        controller_id=self.controller_id,
                        scenario=self.space.scenario(parameters, seeds[0]).to_dict(),
                        objective={**scores[0].to_dict(), **trial.aggregate},
                        observed_failure=any(
                            score.evidence_status == "observed" and score.is_failure
                            for score in scores
                        ),
                        evidence_status="observed"
                        if any(score.evidence_status == "observed" for score in scores)
                        else "inferred",
                        experiment_ids=list(trial.experiment_ids),
                        discovered_at_utc=datetime.now(UTC).isoformat(),
                        similarity_key=similarity_key(parameters),
                    )
                    discoveries.append(discovery)
        finally:
            environment.close()

        ranked = rank_discoveries(discoveries, self.objective.direction)
        held_out = self._held_out(ranked[:5], held_out_seeds or [])
        for discovery in ranked:
            discovery.held_out = held_out.get(discovery.discovery_id, {})
            statuses = [
                replay_experiment(self.experiment_store, experiment_id).status
                for experiment_id in discovery.experiment_ids
            ]
            discovery.replay_status = (
                "exact_match"
                if statuses and all(status == "exact_match" for status in statuses)
                else (statuses[0] if statuses else "unavailable")
            )

        elapsed = time.perf_counter() - started
        summary = {
            "search_id": search_id,
            "strategy": strategy_name,
            "objective_id": self.objective.identifier,
            "direction": self.objective.direction,
            "trial_proposals": len(trial_records),
            "valid_trials": sum(trial.status == "completed" for trial in trial_records),
            "actual_simulation_evaluations": actual_evaluations,
            "execution_failures": sum(
                trial.status == "execution_failed" for trial in trial_records
            ),
            "invalid_trials": sum(trial.status == "invalid" for trial in trial_records),
            "wall_clock_seconds": elapsed,
            "best_objective": best_value,
            "best_so_far": best_history,
            "unique_discoveries": len(ranked),
            "held_out_evaluations": sum(item.get("evaluations", 0) for item in held_out.values()),
        }
        manifest = {
            "schema_version": "1",
            "search_id": search_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "status": "completed",
            "sirfex_version": __version__,
            "environment_id": self.space.environment_id,
            "environment_version": self.space.environment_version,
            "controller_id": self.controller_id,
            "objective_id": self.objective.identifier,
            "strategy": strategy_name,
            "strategy_seed": strategy_seed,
            "trial_budget": trials,
            "scenario_seeds": seeds,
            "held_out_seeds": held_out_seeds or [],
            "dependencies": _dependencies(),
        }
        self.search_store.save(
            search_id,
            manifest,
            self.space.to_dict(),
            [trial.to_dict() for trial in trial_records],
            [discovery.to_dict() for discovery in ranked],
            summary,
        )
        return {
            "manifest": manifest,
            "summary": summary,
            "discoveries": [item.to_dict() for item in ranked],
        }

    def _held_out(
        self, discoveries: list[DiscoveryRecord], seeds: list[int]
    ) -> dict[str, dict[str, Any]]:
        if not seeds:
            return {}
        results: dict[str, dict[str, Any]] = {}
        for discovery in discoveries:
            scores = []
            experiment_ids = []
            for seed in seeds:
                scenario = self.space.scenario(discovery.scenario["parameters"], seed)
                try:
                    manifest = run_and_save_experiment(
                        self.experiment_store,
                        scenario,
                        self.controller_id,
                        max_steps=self.max_steps,
                    )
                    experiment_ids.append(manifest.experiment_id)
                    loaded = self.experiment_store.load(manifest.experiment_id)
                    from sirfex.core.results import EpisodeResult

                    result = EpisodeResult(
                        environment_id=loaded.result["environment_id"],
                        controller_id=loaded.result["controller_id"],
                        scenario=loaded.result["scenario"],
                        seed=loaded.result["seed"],
                        episode_reward=loaded.result["episode_reward"],
                        episode_length=loaded.result["episode_length"],
                        terminated=loaded.result["terminated"],
                        truncated=loaded.result["truncated"],
                        outcome=loaded.result["outcome"],
                        metrics=loaded.result.get("metrics", {}),
                        trace=loaded.trajectory,
                        observed_outcome=loaded.result.get("observed_outcome", "unknown"),
                    )
                    scores.append(self.objective.evaluate(result))
                except Exception:
                    continue
            results[discovery.discovery_id] = {
                "seeds": seeds,
                "evaluations": len(scores),
                "experiment_ids": experiment_ids,
                "mean_risk": sum(score.value for score in scores) / len(scores) if scores else None,
                "failure_frequency": sum(score.is_failure for score in scores) / len(scores)
                if scores
                else None,
            }
        return results
