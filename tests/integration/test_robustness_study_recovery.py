"""Failure injection at each durable boundary of the synthetic study."""

import json

import pytest

from sirfex.robustness.store import RobustnessStorageError
from sirfex.robustness.study import build_plan
from sirfex.robustness.study_execute import StudyEngine
from tests.integration.test_robustness_study_execution import synthetic_package


def _engine(tmp_path):
    protocol, manifest = synthetic_package()
    plan = build_plan(protocol, manifest, tmp_path, synthetic=True)
    return StudyEngine(protocol, manifest, plan, tmp_path, tmp_path, synthetic=True)


def _finish(engine, *, resume=False):
    engine.execute_fixed_and_grid(resume=resume)
    engine.execute_differential(resume=resume)
    return engine.execute_report(resume=resume)


@pytest.mark.parametrize("stage", ["grid", "proposal", "archive", "report"])
def test_interrupted_stage_resumes_with_identical_scientific_report(tmp_path, monkeypatch, stage):
    baseline = _engine(tmp_path / "baseline")
    try:
        expected = _finish(baseline)["semantic_sha256"]
    finally:
        baseline.close()

    interrupted = _engine(tmp_path / "interrupted")
    try:
        if stage == "grid":
            target = interrupted._evaluate_grid_item
        elif stage == "proposal":
            interrupted.execute_fixed_and_grid()
            target = interrupted._evaluate_proposal
        elif stage == "archive":
            interrupted.execute_fixed_and_grid()
            import sirfex.robustness.study_execute as module

            target = module.archive_selected_findings
        else:
            interrupted.execute_fixed_and_grid()
            interrupted.execute_differential()
            import sirfex.robustness.study_execute as module

            target = module._atomic_json

        tripped = False

        def fail_once(*args, **kwargs):
            nonlocal tripped
            if stage == "report":
                if args[0].name != "report.json" or tripped:
                    return target(*args, **kwargs)
            elif tripped:
                return target(*args, **kwargs)
            tripped = True
            if stage == "archive":
                target(*args, **kwargs)  # Durable evidence exists before the crash.
            raise RuntimeError("injected interruption")

        if stage == "grid":
            monkeypatch.setattr(interrupted, "_evaluate_grid_item", fail_once)
            run = interrupted.execute_fixed_and_grid
        elif stage == "proposal":
            monkeypatch.setattr(interrupted, "_evaluate_proposal", fail_once)
            run = interrupted.execute_differential
        elif stage == "archive":
            monkeypatch.setattr(module, "archive_selected_findings", fail_once)
            run = interrupted.execute_differential
        else:
            monkeypatch.setattr(module, "_atomic_json", fail_once)
            run = interrupted.execute_report
        with pytest.raises(RuntimeError, match="injected interruption"):
            run()
        assert tripped
        assert interrupted.store.status()["status"] == "incomplete"
        monkeypatch.undo()
    finally:
        interrupted.close()

    resumed = _engine(tmp_path / "interrupted")
    try:
        assert _finish(resumed, resume=True)["semantic_sha256"] == expected
        assert resumed.store.status()["status"] == "complete"
    finally:
        resumed.close()


def test_corrupt_completed_case_fails_closed_on_resume(tmp_path):
    engine = _engine(tmp_path)
    try:
        engine.execute_fixed_and_grid()
        item = next(i for i in engine.plan["work_items"] if i["stage"] == "grid")
        path = engine.store._path(item["work_id"])
        value = json.loads(path.read_text())
        value["status"] = "failed"
        path.write_text(json.dumps(value))
    finally:
        engine.close()
    resumed = _engine(tmp_path)
    try:
        with pytest.raises(RobustnessStorageError, match="hash changed"):
            resumed.execute_fixed_and_grid(resume=True)
    finally:
        resumed.close()
