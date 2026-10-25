import pytest

from sirfex.experiments.replay import ReplayReport
from sirfex.experiments.storage import ExperimentStore
from sirfex.search.avoidable import run_avoidable, validate_avoidable_evidence
from sirfex.search.persistence import SearchStore
from sirfex.training.persistence import TrainingStore


def config(**parameters):
    return {
        "environment_id": "pump",
        "environment_version": "sirfex-pump:1",
        "objective": "pump.constraint_violation",
        "controller_id": "pump-fixed",
        "controller_config": {"command": 0.0},
        "trials": 1,
        "wall_seconds": 30,
        "fixed_parameters": {
            "dt": 0.1,
            "horizon": 3,
            "initial_level": 0.1,
            "target_level": 0.5,
            "min_level": 0.05,
            "max_level": 0.95,
            "demand": 0.9,
            "pump_capacity": 0.8,
            "efficiency": 1.0,
            "leak": 0.0,
            **parameters,
        },
    }


def stores(tmp_path):
    return (
        ExperimentStore(tmp_path / "experiments"),
        SearchStore(tmp_path / "search"),
        TrainingStore(tmp_path / "training"),
    )


def test_finite_horizon_witness_and_search_accounting(tmp_path):
    experiments, searches, training = stores(tmp_path)
    result = run_avoidable(config(), experiments, searches, training)
    summary = result["summary"]
    assert summary["feasible_candidates"] == 1
    assert summary["actual_simulation_evaluations"] == 1
    assert summary["reference_evaluations"] == 2
    assert summary["avoidable_discoveries"] == 1
    assert summary["observed_controller_violations"] == 1
    rows = searches.load(result["search_id"])["avoidable"]["records"]
    assert all(r["avoidable"] for r in rows)
    assert rows[0]["feasibility"]["classification"] == "bounded_operation_feasible"
    assert len(rows[0]["witness"]["commands"]) == 3
    assert all(r["status"] == "exact_match" for r in rows[0]["replay"])


def test_infeasible_proposal_is_not_simulated(tmp_path):
    experiments, searches, training = stores(tmp_path)
    result = run_avoidable(config(pump_capacity=0.1), experiments, searches, training)
    assert result["summary"]["rejected_candidates"] == 1
    assert result["summary"]["invalid_trials"] == 0
    assert result["summary"]["actual_simulation_evaluations"] == 0
    assert result["summary"]["avoidable_discoveries"] == 0


def test_success_and_invalid_proposals_have_no_discovery(tmp_path):
    experiments, searches, training = stores(tmp_path)
    result = run_avoidable(config(initial_level=0.5, demand=0.1), experiments, searches, training)
    assert result["summary"]["avoidable_discoveries"] == 0
    result = run_avoidable(config(initial_level=0.01), experiments, searches, training)
    assert result["summary"]["invalid_trials"] == 1


def test_replay_mismatch_and_missing_reference_cannot_claim_avoidability(tmp_path, monkeypatch):
    experiments, searches, training = stores(tmp_path)
    result = run_avoidable(config(), experiments, searches, training)
    row = searches.load(result["search_id"])["avoidable"]["records"][0]
    failing, reference = row["failure_experiment_id"], row["reference_experiment_id"]
    assert not validate_avoidable_evidence(experiments, failing, "missing_123")["avoidable"]
    monkeypatch.setattr(
        "sirfex.search.avoidable.replay_experiment",
        lambda *args: ReplayReport("mismatch", failing, "fixture"),
    )
    assert not validate_avoidable_evidence(experiments, failing, reference)["avoidable"]


def test_tpe_remains_available_with_candidate_filter(tmp_path):
    pytest.importorskip("optuna")
    experiments, searches, training = stores(tmp_path)
    result = run_avoidable({**config(), "strategy": "tpe"}, experiments, searches, training)
    assert result["summary"]["avoidable_discoveries"] == 1
