"""Feasibility denominators use the analyzer's actual classification contract."""

from sirfex.robustness.study_report import _grid_feasibility_counts, _paired, build_study_report


def test_equal_missing_scenarios_do_not_make_a_pair_complete():
    same = {
        "one": {
            "tracking_mae": 0.1,
            "margin": {
                "post_initial_minimum_margin": 0.2,
                "constraint_violated": False,
            },
        }
    }
    result = _paired(same, same, ["one", "two"])
    assert result["status"] == "incomplete"
    assert result["missing_x"] == result["missing_y"] == ["two"]


def test_viable_infeasible_unknown_and_missing_remain_separate():
    points = [
        {"domain_status": "valid", "feasibility": "target_feasible", "failure": False},
        {"domain_status": "valid", "feasibility": "bounded_operation_feasible", "failure": True},
        {"domain_status": "valid", "feasibility": "infeasible", "failure": True},
        {"domain_status": "valid", "feasibility": "undetermined", "failure": True},
        {"domain_status": "valid", "feasibility": "undetermined", "failure": False},
        {"domain_status": "valid", "feasibility": "target_feasible", "failure": None},
        {"domain_status": "valid", "feasibility": None, "failure": True},
        {"domain_status": "censored", "feasibility": None, "failure": None},
    ]
    summary = _grid_feasibility_counts(points)
    groups = summary["groups"]
    assert groups["viable"] == {
        "scenarios": 3,
        "evaluated": 2,
        "failures": 1,
        "survivals": 1,
        "missing_evidence": 1,
    }
    assert groups["infeasible"] == {
        "scenarios": 1,
        "evaluated": 1,
        "failures": 1,
        "survivals": 0,
        "missing_evidence": 0,
    }
    assert groups["unknown"] == {
        "scenarios": 2,
        "evaluated": 2,
        "failures": 1,
        "survivals": 1,
        "missing_evidence": 0,
    }
    assert groups["invalid"]["evaluated"] == 1
    assert groups["invalid"]["failures"] == 1
    assert summary["valid_evaluated"] == 5
    assert summary["censored"] == 1
    assert sum(groups[key]["evaluated"] for key in ("viable", "infeasible", "unknown")) == 5


def test_synthetic_report_never_calls_unknown_failure_avoidable():
    points = [
        {"domain_status": "valid", "feasibility": "target_feasible", "failure": True},
        {"domain_status": "valid", "feasibility": "infeasible", "failure": True},
        {"domain_status": "valid", "feasibility": "undetermined", "failure": True},
        {"domain_status": "valid", "feasibility": "undetermined", "failure": False},
        {"domain_status": "valid", "feasibility": "target_feasible", "failure": None},
    ]
    item = {
        "stage": "grid",
        "work_id": "grid",
        "controller": "synthetic",
        "anchor_id": "anchor",
        "direction": ["demand"],
    }
    cell = {
        "anchor_id": "anchor",
        "direction": ["demand"],
        "grid_outcomes": points,
        "delta_first": 0.0,
        "delta_infeasible_grid": 0.1,
        "relative_failure_threshold": 0.0,
        "status": "failed",
        "nonmonotone": True,
        "requested_grid_points": 5,
        "valid_grid_points": 5,
    }

    class FakeStore:
        plan = {
            "protocol_sha256": "protocol",
            "scenario_manifest_sha256": "manifest",
            "work_items": [item],
            "work_counts": {"grid": 1},
            "controllers": ["synthetic"],
            "pairs": [],
            "profile": "synthetic_test_only",
            "semantic_sha256": "plan",
        }

        def status(self):
            return {
                "stages": {
                    "grid": {
                        "pending": 0,
                        "running": 0,
                        "failed": 0,
                        "invalid": 0,
                        "completed": 1,
                        "censored": 0,
                    }
                }
            }

        def read(self, work_id):
            assert work_id == "grid"
            return {"status": "completed", "result": {"cell": cell}}

    protocol = {
        "semantic_sha256": "protocol",
        "controller_inventory": [],
        "search": {"search_seed": 1, "scenario_seed_start": 2},
    }
    report = build_study_report(FakeStore(), protocol, {"semantic_sha256": "manifest"}, {})
    row = report["controllers"][0]
    assert row["physically_viable_points"] == 1
    assert row["physically_infeasible_points"] == 1
    assert row["physically_unknown_points"] == 2
    assert row["avoidable_grid_violations"] == 1
    assert row["unknown_feasibility_grid_violations"] == 1
    assert row["grid_missing_evidence"] == 1
    assert row["valid_evaluated_grid_points"] == 4
    assert report["execution_status"] == "partial_incomplete"
