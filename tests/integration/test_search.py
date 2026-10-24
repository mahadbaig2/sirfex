from sirfex.experiments.storage import ExperimentStore
from sirfex.search.objectives import PumpConstraintObjective
from sirfex.search.persistence import SearchStore
from sirfex.search.runner import SearchRunner
from sirfex.search.space import SearchSpace


def _guaranteed_failure_space() -> SearchSpace:
    return SearchSpace(
        "pump",
        "sirfex-pump:1",
        fixed_parameters={
            "dt": 0.1,
            "horizon": 200,
            "demand": 0.9,
            "pump_capacity": 0.2,
            "efficiency": 0.4,
            "initial_level": 0.2,
            "target_level": 0.5,
            "leak": 0.0,
            "min_level": 0.05,
            "max_level": 0.95,
        },
    )


def test_search_finds_and_persists_observed_pump_violation(tmp_path) -> None:
    experiment_store = ExperimentStore(tmp_path / "experiments")
    search_store = SearchStore(tmp_path / "searches")
    result = SearchRunner(
        _guaranteed_failure_space(),
        PumpConstraintObjective(),
        "pump-rule",
        experiment_store=experiment_store,
        search_store=search_store,
        max_steps=200,
    ).run("random", trials=2, scenario_seeds=[0], held_out_seeds=[7])
    assert result["summary"]["actual_simulation_evaluations"] == 2
    assert result["summary"]["unique_discoveries"] == 1
    discovery = result["discoveries"][0]
    assert discovery["observed_failure"] is True
    assert discovery["replay_status"] == "exact_match"
    loaded = search_store.load(result["summary"]["search_id"])
    assert len(loaded["trials"]) == 2
    verification_dir = search_store.root / result["summary"]["search_id"] / "verification"
    assert any(verification_dir.iterdir())
