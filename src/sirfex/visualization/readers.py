"""Read-only access to persisted SIRFEX evidence."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sirfex.experiments.replay import ReplayReport, replay_experiment
from sirfex.experiments.storage import ExperimentStore, StorageError, StoredExperiment
from sirfex.search.persistence import SearchStorageError, SearchStore
from sirfex.training.persistence import TrainingStorageError, TrainingStore


class EvidenceReadError(RuntimeError):
    """An evidence artifact cannot be safely read."""


@dataclass(frozen=True)
class ExperimentSummary:
    experiment_id: str
    created_at_utc: str
    environment_id: str
    controller_id: str
    execution_status: str
    outcome: str
    failure_count: int


class EvidenceReader:
    """Small query facade over the existing experiment and search stores."""

    def __init__(
        self,
        experiment_root: Path | str = "artifacts/experiments",
        search_root: Path | str = "artifacts/searches",
    ) -> None:
        self.experiments = ExperimentStore(experiment_root)
        self.searches = SearchStore(search_root)
        self.training = TrainingStore(Path(search_root).parent / "training")

    def list_experiments(self) -> list[ExperimentSummary]:
        return [
            ExperimentSummary(
                item.experiment_id,
                item.created_at_utc,
                item.environment_id,
                item.controller_id,
                item.execution_status,
                item.outcome,
                len(item.failures),
            )
            for item in self.experiments.list()
        ]

    def load_experiment(self, experiment_id: str) -> StoredExperiment:
        try:
            return self.experiments.load(experiment_id)
        except (StorageError, OSError, ValueError) as exc:
            raise EvidenceReadError(str(exc)) from exc

    def iter_trajectory(self, experiment_id: str) -> Iterator[dict[str, Any]]:
        """Yield a selected experiment's bounded-by-consumer JSONL trajectory."""
        stored = self.load_experiment(experiment_id)
        yield from stored.trajectory

    def list_searches(self) -> list[dict[str, Any]]:
        return self.searches.list()

    def load_search(self, search_id: str) -> dict[str, Any]:
        try:
            return self.searches.load(search_id)
        except (SearchStorageError, OSError, ValueError) as exc:
            raise EvidenceReadError(str(exc)) from exc

    def load_verification(self, search_id: str, discovery_id: str) -> dict[str, Any] | None:
        try:
            return self.searches.load_verification(search_id, discovery_id)
        except (SearchStorageError, OSError, ValueError) as exc:
            raise EvidenceReadError(str(exc)) from exc

    def supporting_experiments(self, experiment_ids: list[str]) -> list[StoredExperiment]:
        return [self.load_experiment(experiment_id) for experiment_id in experiment_ids]

    def replay(self, experiment_id: str, tolerance: float = 1e-9) -> ReplayReport:
        try:
            return replay_experiment(self.experiments, experiment_id, tolerance)
        except Exception as exc:  # the UI must render unavailable evidence, not crash
            return ReplayReport("unavailable", experiment_id, str(exc))

    def list_training_runs(self) -> list[dict[str, Any]]:
        return self.training.list()

    def load_training_run(self, training_run_id: str) -> dict[str, Any]:
        try:
            return self.training.load(training_run_id)
        except (TrainingStorageError, OSError, ValueError) as exc:
            raise EvidenceReadError(str(exc)) from exc

    def overview(self) -> dict[str, int]:
        experiments = self.list_experiments()
        searches = self.list_searches()
        discoveries = 0
        observed = 0
        proxy = 0
        replay_verified = 0
        for manifest in searches:
            try:
                loaded = self.load_search(manifest["search_id"])
            except EvidenceReadError:
                continue
            for discovery in loaded["discoveries"]:
                discoveries += 1
                if discovery.get("evidence_status") == "observed":
                    observed += 1
                if discovery.get("evidence_status") == "inferred":
                    proxy += 1
                if discovery.get("replay_status") in {"exact_match", "tolerance_match"}:
                    replay_verified += 1
        return {
            "experiments": len(experiments),
            "search_runs": len(searches),
            "completed_trials": sum(
                self.load_search(item["search_id"])["summary"].get("valid_trials", 0)
                for item in searches
                if self._search_available(item["search_id"])
            ),
            "discoveries": discoveries,
            "observed_discoveries": observed,
            "proxy_discoveries": proxy,
            "replay_verified_discoveries": replay_verified,
        }

    def _search_available(self, search_id: str) -> bool:
        try:
            self.load_search(search_id)
        except EvidenceReadError:
            return False
        return True
