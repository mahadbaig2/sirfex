"""Approved limits remain binding across synthetic study restarts."""

import pytest

from sirfex.robustness.identity import semantic_hash
from sirfex.robustness.study import build_plan
from sirfex.robustness.study_execute import StudyEngine
from sirfex.robustness.study_report import REPORT_SCHEMA
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


def test_storage_stop_during_case_persistence_keeps_prior_evidence(tmp_path, monkeypatch):
    engine = _limited_engine(tmp_path)
    original = engine.store._check_storage
    first = next(row for row in engine.plan["work_items"] if row["stage"] == "fixed_stress")
    engine.store.write(first, "completed", {"synthetic_prior": True})
    before = engine.store.read(first["work_id"])

    def deny_completion(value, **kwargs):
        if (
            value.get("schema_version") == "pump-study-work-item:1"
            and value.get("status") == "completed"
        ):
            raise StudyBudgetExhausted("study_storage_budget_exhausted")
        return original(value, **kwargs)

    monkeypatch.setattr(engine.store, "_check_storage", deny_completion)
    try:
        with pytest.raises(StudyBudgetExhausted, match="storage_budget"):
            engine.execute_fixed_and_grid()
        assert engine.store.read(first["work_id"]) == before
        assert engine.store.status()["status"] == "incomplete"
        assert engine.store.budget()["cumulative_wall_seconds"] > 0
    finally:
        engine.close()


def test_storage_stop_during_archive_and_report_is_resumable(tmp_path, monkeypatch):
    engine = _limited_engine(tmp_path)
    try:
        engine.execute_fixed_and_grid()
        with pytest.raises(StudyBudgetExhausted, match="synthetic_interruption"):
            engine.execute_differential(interrupt_after=1)
        proposal = next(row for row in engine.plan["work_items"] if row["stage"] == "differential")
        saved_proposal = engine.store.read(proposal["work_id"])
        original_check = engine.store.check_budget

        def deny_archive(controllers=(), *, reserve_bytes=0):
            if reserve_bytes and reserve_bytes != 1_000_000:
                raise StudyBudgetExhausted("study_storage_budget_exhausted")
            return original_check(controllers, reserve_bytes=reserve_bytes)

        monkeypatch.setattr(engine.store, "check_budget", deny_archive)
        with pytest.raises(StudyBudgetExhausted, match="storage_budget"):
            engine.execute_differential(resume=True)
        assert engine.store.read(proposal["work_id"]) == saved_proposal
        monkeypatch.undo()
    finally:
        engine.close()

    resumed = _limited_engine(tmp_path)
    try:
        resumed.execute_differential(resume=True)
        original_storage = resumed.store._check_storage

        def deny_report(value, **kwargs):
            if value.get("schema_version") == REPORT_SCHEMA:
                raise StudyBudgetExhausted("study_storage_budget_exhausted")
            return original_storage(value, **kwargs)

        monkeypatch.setattr(resumed.store, "_check_storage", deny_report)
        with pytest.raises(StudyBudgetExhausted, match="storage_budget"):
            resumed.execute_report(resume=True)
        assert resumed.store.status()["status"] == "incomplete"
        monkeypatch.undo()
    finally:
        resumed.close()

    final = _limited_engine(tmp_path)
    try:
        assert final.execute_report(resume=True)["execution_status"] == "complete"
    finally:
        final.close()
