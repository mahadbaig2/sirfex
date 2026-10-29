"""Checkpoint 2A development-only identity and execution-gate tests."""

import json
from pathlib import Path

import pytest

from sirfex.core import Scenario
from sirfex.robustness.identity import semantic_hash
from sirfex.robustness.scientific import StudyGateError, dry_run, scientific_execute
from sirfex.robustness.store import RobustnessStorageError, RobustnessStore, run_items

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def package():
    protocol = json.loads(
        (ROOT / "configs/robustness/phase37_checkpoint2_protocol.candidate.json").read_text()
    )
    manifest = json.loads(
        (ROOT / "docs/evidence/phase_3_7_development_scenario_manifest.json").read_text()
    )
    return protocol, manifest


def rehash(value):
    value["semantic_sha256"] = semantic_hash(
        {k: v for k, v in value.items() if k != "semantic_sha256"}
    )
    return value


def test_candidate_dry_run_and_exact_design(package):
    protocol, manifest = package
    if any(
        not (ROOT / row["source_artifact"]).is_file()
        for row in protocol["controller_inventory"]
        if row["group"] != "reference"
    ):
        pytest.skip("The local PPO checkpoint artifacts are not installed")
    result = dry_run(protocol, manifest, ROOT)
    assert result["controllers"] == 17
    assert result["anchors"] == 12
    assert result["pairs"] == 7
    assert result["episode_upper"] == 88168
    assert [p["id"] for p in protocol["differential_pairs"]] == [
        "primary_3_expanded_5",
        "primary_3_expanded_10",
        "primary_17_expanded_5",
        "primary_17_expanded_10",
        "secondary_3_robust30",
        "secondary_17_robust30",
        "secondary_41_robust30",
    ]


def test_candidate_execution_always_refused(package):
    protocol, manifest = package
    for token in (None, protocol["semantic_sha256"]):
        with pytest.raises(StudyGateError, match="Draft/candidate"):
            scientific_execute(
                protocol,
                manifest,
                ROOT,
                protocol_file=ROOT
                / "configs/robustness/phase37_checkpoint2_protocol.candidate.json",
                approval_digest=token,
            )


def test_missing_and_mismatched_approval_refused(package):
    original, manifest = package
    protocol = rehash(
        {**original, "status": "FROZEN — AUTHORIZED FOR EXECUTION", "executable": True}
    )
    with pytest.raises(StudyGateError, match="Missing explicit"):
        scientific_execute(
            protocol, manifest, ROOT, protocol_file=ROOT / "x.json", approval_digest=None
        )
    with pytest.raises(StudyGateError, match="Approval digest"):
        scientific_execute(
            protocol, manifest, ROOT, protocol_file=ROOT / "x.json", approval_digest="0" * 64
        )


def test_manifest_tampering_and_sealed_partition_refused(package):
    protocol, original = package
    manifest = json.loads(json.dumps(original))
    manifest["scenarios"][0]["scenario"]["seed"] = 1
    with pytest.raises(StudyGateError, match="SHA-256 mismatch"):
        dry_run(protocol, manifest, ROOT)
    manifest = json.loads(json.dumps(original))
    manifest["scenarios"][0]["partition"] = "final_stress"
    rehash(manifest)
    mutated_protocol = rehash({**protocol, "scenario_manifest_sha256": manifest["semantic_sha256"]})
    with pytest.raises(StudyGateError, match="Unsupported scenario partition"):
        dry_run(mutated_protocol, manifest, ROOT)


def test_missing_checkpoint_and_budget_refused(package, tmp_path):
    original, manifest = package
    protocol = json.loads(json.dumps(original))
    protocol["controller_inventory"][0]["source_artifact"] = "missing.zip"
    rehash(protocol)
    with pytest.raises(StudyGateError, match="Missing checkpoint"):
        dry_run(protocol, manifest, ROOT)
    protocol = json.loads(json.dumps(original))
    protocol["budget"]["artifact_bytes_cap"] = 3 * 1024**3
    rehash(protocol)
    with pytest.raises(StudyGateError, match="budget ceilings"):
        dry_run(protocol, manifest, ROOT)


def test_incompatible_frozen_commit_refused(package):
    original, manifest = package
    protocol = rehash(
        {**original, "status": "FROZEN — AUTHORIZED FOR EXECUTION", "executable": True}
    )
    with pytest.raises(StudyGateError, match="Frozen protocol is not committed"):
        scientific_execute(
            protocol,
            manifest,
            ROOT,
            protocol_file=ROOT / "never-committed.json",
            approval_digest=protocol["semantic_sha256"],
        )


def test_synthetic_immutable_resume_and_report_identity(tmp_path):
    """Synthetic case execution exercises persistence without controller rollout."""
    store = RobustnessStore(tmp_path)
    protocol = {"schema_version": "pump-robustness-protocol:1", "profile": "synthetic_test_only"}
    run_id = store.open_run(protocol)
    scenario = Scenario(
        "pump",
        {
            "dt": 0.1,
            "horizon": 2,
            "demand": 0.4,
            "pump_capacity": 0.8,
            "efficiency": 0.9,
            "leak": 0.01,
        },
        seed=987654321,
        environment_version="sirfex-pump:1",
        provenance={"partition": "synthetic_test_only"},
    )
    controllers = [{"id": "pump-rule", "config": {"test_arm": i}} for i in (0, 1)]
    items = [(controller, scenario) for controller in controllers]
    attempted = []

    def interrupted(controller, _scenario):
        attempted.append(controller["config"]["test_arm"])
        if len(attempted) == 2:
            raise RuntimeError("synthetic interruption")
        return {"synthetic_value": controller["config"]["test_arm"]}

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_items(store, run_id, items, interrupted)
    counts = run_items(
        store,
        run_id,
        items,
        lambda controller, _scenario: {"synthetic_value": controller["config"]["test_arm"]},
    )
    assert counts["completed"] == 2
    digest = counts["semantic_results_sha256"]
    reused = run_items(store, run_id, items, lambda *_: pytest.fail("completed case reevaluated"))
    assert reused["reused_completed"] == 2
    assert reused["semantic_results_sha256"] == digest
    other = store.open_run({**protocol, "revision": 2}, run_id="synthetic_other")
    assert other != run_id
    with pytest.raises(RobustnessStorageError, match="incompatible protocol"):
        store.open_run({**protocol, "revision": 2}, run_id=run_id)
    case_id = store.case_id(run_id, controllers[0], scenario)
    path = store.directory(run_id) / "cases" / f"{case_id}.json"
    path.write_text(path.read_text().replace("synthetic_value", "tampered_value"))
    with pytest.raises(RobustnessStorageError, match="hash changed"):
        store.read_case(run_id, case_id)
