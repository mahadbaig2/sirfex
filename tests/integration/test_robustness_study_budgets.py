"""Approved limits remain binding across synthetic study restarts."""

import pytest

from sirfex.robustness.identity import semantic_hash
from sirfex.robustness.study import build_plan
from sirfex.robustness.study_execute import StudyEngine
from sirfex.robustness.study_store import StudyBudgetExhausted
from tests.integration.test_robustness_study_execution import synthetic_package


def _limited_engine(tmp_path, *, wall=300, ppo=30, storage=10_000_000):
    protocol, manifest = synthetic_package()
    protocol["budget"].update(
        wall_seconds_cap=wall,
        per_ppo_seconds_cap=ppo,
        artifact_bytes_cap=storage,
    )
    protocol["semantic_sha256"] = semantic_hash(
        {key: value for key, value in protocol.items() if key != "semantic_sha256"}
    )
    plan = build_plan(protocol, manifest, tmp_path, synthetic=True)
    return StudyEngine(protocol, manifest, plan, tmp_path, tmp_path, synthetic=True)


def test_wall_budget_stops_and_restart_cannot_reset_it(tmp_path):
    engine = _limited_engine(tmp_path, wall=1)
    try:
        engine.store.charge(1.0, {})
        with pytest.raises(StudyBudgetExhausted, match="wall_budget"):
            engine.execute_fixed_and_grid()
        assert engine.store.status()["status"] == "incomplete"
    finally:
        engine.close()
    resumed = _limited_engine(tmp_path, wall=1)
    try:
        with pytest.raises(StudyBudgetExhausted, match="wall_budget"):
            resumed.execute_fixed_and_grid(resume=True)
        assert resumed.store.budget()["cumulative_wall_seconds"] >= 1
    finally:
        resumed.close()


def test_ppo_budget_stops_scheduling_controller_and_preserves_completed_cases(tmp_path):
    engine = _limited_engine(tmp_path, ppo=0.5)
    try:
        # Exercise the same adapter scheduling key used for production PPO items.
        engine.store.charge(0, {"screen_3_normal": 0.5})
        with pytest.raises(StudyBudgetExhausted, match="controller_budget"):
            engine.store.check_budget(("screen_3_normal",))
        assert engine.store.status()["status"] == "incomplete"
    finally:
        engine.close()


def test_storage_budget_stops_before_overwriting_evidence(tmp_path):
    engine = _limited_engine(tmp_path, storage=20_000)
    try:
        item = next(row for row in engine.plan["work_items"] if row["stage"] == "fixed_stress")
        engine.store.write(item, "completed", {"ok": True})
        before = engine.store.read(item["work_id"])
        with pytest.raises(StudyBudgetExhausted, match="storage_budget"):
            engine.store.check_budget(reserve_bytes=20_000)
        assert engine.store.read(item["work_id"]) == before
    finally:
        engine.close()
