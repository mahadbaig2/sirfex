import pytest

from sirfex.core import Scenario
from sirfex.experiments.service import run_and_save_experiment
from sirfex.experiments.storage import ExperimentStore, StorageError


def test_storage_detects_corrupt_manifest(tmp_path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    manifest = run_and_save_experiment(store, Scenario("pump"), "pump-rule", max_steps=2)
    (tmp_path / "experiments" / manifest.experiment_id / "manifest.json").write_text(
        "{", encoding="utf-8"
    )
    with pytest.raises(StorageError, match="Malformed"):
        store.load(manifest.experiment_id)


def test_storage_rejects_path_traversal(tmp_path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    with pytest.raises(StorageError):
        store.load("..\\outside")
