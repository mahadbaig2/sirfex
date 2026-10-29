"""Work-plan identity tests; no controller episodes or sealed partitions."""

import json
from pathlib import Path

import pytest

from sirfex.robustness.identity import semantic_hash
from sirfex.robustness.scientific import StudyGateError
from sirfex.robustness.study import build_plan

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def candidate():
    protocol = json.loads(
        (ROOT / "configs/robustness/phase37_checkpoint2_protocol.candidate.json").read_text()
    )
    manifest = json.loads(
        (ROOT / "docs/evidence/phase_3_7_development_scenario_manifest.json").read_text()
    )
    return protocol, manifest


def test_candidate_plan_is_complete_and_deterministic(candidate):
    protocol, manifest = candidate
    first = build_plan(protocol, manifest, ROOT)
    second = build_plan(protocol, manifest, ROOT)
    assert first["semantic_sha256"] == second["semantic_sha256"]
    assert first["work_counts"] == {
        "archive": 7,
        "differential": 1400,
        "fixed_stress": 408,
        "grid": 3264,
        "report": 1,
    }
    assert len({row["work_id"] for row in first["work_items"]}) == len(first["work_items"])
    assert first["protocol_sha256"] == protocol["semantic_sha256"]
    assert first["scenario_manifest_sha256"] == manifest["semantic_sha256"]
    assert first["grid"] == [round(i * 0.025, 3) for i in range(25)]


def test_pair_identity_change_requires_new_hash(candidate):
    original, manifest = candidate
    protocol = json.loads(json.dumps(original))
    protocol["differential_pairs"][0]["y"] = "missing_controller"
    protocol["semantic_sha256"] = semantic_hash(
        {key: value for key, value in protocol.items() if key != "semantic_sha256"}
    )
    with pytest.raises(StudyGateError, match="unavailable controller"):
        build_plan(protocol, manifest, ROOT)
