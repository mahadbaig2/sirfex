"""Calibration resume must report the current invocation separately from first timing."""

import json
from pathlib import Path

from sirfex.robustness.service import run_differential, run_tolerance


def test_tolerance_resume_preserves_result_hash_and_reports_zero_new_episodes(tmp_path):
    config = json.loads(Path("configs/robustness/pump_calibration.json").read_text())
    config["directions"] = config["directions"][:1]
    config["grid"] = config["grid"][:2]
    config["controllers"] = config["controllers"][:1]
    first = run_tolerance(config, tmp_path)
    resumed = run_tolerance(config, tmp_path)
    assert first["report_sha256"] == resumed["report_sha256"]
    assert first["calibration"]["measured_episodes_this_invocation"] > 0
    assert resumed["calibration"]["measured_episodes_this_invocation"] == 0
    assert (
        resumed["calibration"]["first_measurement"]["measured_episodes_this_invocation"]
        == first["calibration"]["measured_episodes_this_invocation"]
    )
    assert resumed["calibration"]["counts"]["reused_completed"] == 1


def test_differential_resume_reports_zero_new_episodes(tmp_path):
    config = json.loads(Path("configs/robustness/pump_differential_calibration.json").read_text())
    config["proposals"] = 1
    first = run_differential(config, tmp_path)
    resumed = run_differential(config, tmp_path)
    assert first["report_sha256"] == resumed["report_sha256"]
    assert first["calibration"]["measured_episodes_this_invocation"] > 0
    assert resumed["calibration"]["measured_episodes_this_invocation"] == 0
    assert (
        resumed["calibration"]["first_measurement"]["measured_episodes_this_invocation"]
        == first["calibration"]["measured_episodes_this_invocation"]
    )
