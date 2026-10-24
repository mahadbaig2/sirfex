import json

from sirfex.core import Scenario
from sirfex.experiments.comparison import compare_experiments
from sirfex.experiments.replay import replay_experiment
from sirfex.experiments.service import run_and_save_experiment
from sirfex.experiments.storage import ExperimentStore


def test_pump_experiment_save_reload_replay_and_compare(tmp_path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    scenario = Scenario("pump", seed=12, description="test")
    first = run_and_save_experiment(store, scenario, "pump-rule", max_steps=8)
    second = run_and_save_experiment(store, Scenario("pump", seed=13), "pump-rule", max_steps=8)

    loaded = store.load(first.experiment_id)
    json.dumps(loaded.result)
    assert loaded.manifest.experiment_id == first.experiment_id
    assert len(loaded.trajectory) == 8

    replay = replay_experiment(store, first.experiment_id)
    assert replay.status == "exact_match"

    comparison = compare_experiments(store, first.experiment_id, second.experiment_id)
    assert comparison.comparable is True
    assert comparison.differences["seed"]["first"] != comparison.differences["seed"]["second"]
