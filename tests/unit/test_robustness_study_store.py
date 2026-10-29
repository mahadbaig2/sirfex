import json

import pytest

from sirfex.robustness.identity import semantic_hash
from sirfex.robustness.store import RobustnessStorageError
from sirfex.robustness.study_store import InvocationBudget, StudyBudgetExhausted, StudyStore


def fixture_plan(*, wall=10, ppo=10, storage=100_000):
    item = {"work_id": "a" * 64, "stage": "fixed_stress", "controller": "screen_3_normal"}
    plan = {
        "schema_version": "pump-robustness-study-plan:1",
        "profile": "synthetic_test_only",
        "work_items": [item],
        "work_counts": {"fixed_stress": 1},
        "resource_limits": {
            "wall_seconds_cap": wall,
            "per_ppo_seconds_cap": ppo,
            "artifact_bytes_cap": storage,
        },
    }
    plan["semantic_sha256"] = semantic_hash(plan)
    return plan, item


def test_ledger_resume_immutability_and_corruption(tmp_path):
    plan, item = fixture_plan()
    store = StudyStore(tmp_path, plan)
    store.open()
    assert store.status()["stages"]["fixed_stress"]["pending"] == 1
    store.write(item, "running")
    store.write(item, "completed", {"metric": 0.2})
    assert store.status()["status"] == "complete"
    with pytest.raises(RobustnessStorageError, match="immutable"):
        store.write(item, "completed", {"metric": 0.3})
    reopened = StudyStore(tmp_path, plan)
    reopened.open()
    assert reopened.read(item["work_id"])["result"] == {"metric": 0.2}
    path = reopened._path(item["work_id"])
    path.write_text(path.read_text().replace("0.2", "0.3"))
    with pytest.raises(RobustnessStorageError, match="hash changed"):
        reopened.read(item["work_id"])


def test_explicit_resume_and_incompatible_plan(tmp_path):
    plan, item = fixture_plan()
    store = StudyStore(tmp_path, plan)
    store.open()
    store.write(item, "failed", {"error": "synthetic"})
    with pytest.raises(RobustnessStorageError, match="explicit resume"):
        store.write(item, "running")
    store.write(item, "running", resume=True)
    store.write(item, "completed", {"ok": True})
    changed = {**plan, "resource_limits": {**plan["resource_limits"], "wall_seconds_cap": 11}}
    changed["semantic_sha256"] = semantic_hash(
        {k: v for k, v in changed.items() if k != "semantic_sha256"}
    )
    other = StudyStore(tmp_path, changed)
    other.run_id = store.run_id
    other.directory = store.directory
    other.items_dir = store.items_dir
    with pytest.raises(RobustnessStorageError, match="incompatible"):
        other.open()


def test_cumulative_wall_controller_and_storage_budgets(tmp_path):
    plan, _ = fixture_plan(wall=5, ppo=2)
    store = StudyStore(tmp_path, plan)
    store.open()
    ticks = iter((0.0, 1.0, 1.0))
    budget = InvocationBudget(store, clock=lambda: next(ticks))
    budget.charge({"screen_3_normal": 2.1})
    with pytest.raises(StudyBudgetExhausted, match="controller_budget"):
        budget.check(("screen_3_normal",))
    with pytest.raises(StudyBudgetExhausted, match="storage_budget"):
        store.check_budget(reserve_bytes=plan["resource_limits"]["artifact_bytes_cap"])
    store.charge(4.0, {})
    with pytest.raises(StudyBudgetExhausted, match="wall_budget"):
        store.check_budget()
    path = store.directory / "budget.json"
    value = json.loads(path.read_text())
    value["cumulative_wall_seconds"] = 0
    path.write_text(json.dumps(value))
    with pytest.raises(RobustnessStorageError, match="hash changed"):
        store.budget()


def test_elapsed_wall_budget_is_durable_before_rejection(tmp_path):
    plan, _ = fixture_plan(wall=2)
    store = StudyStore(tmp_path, plan)
    store.open()
    ticks = iter((10.0, 12.5))
    budget = InvocationBudget(store, clock=lambda: next(ticks))
    with pytest.raises(StudyBudgetExhausted, match="wall_budget"):
        budget.check()
    assert store.budget()["cumulative_wall_seconds"] == 2.5
    reopened = StudyStore(tmp_path, plan)
    reopened.open()
    with pytest.raises(StudyBudgetExhausted, match="wall_budget"):
        reopened.check_budget()
