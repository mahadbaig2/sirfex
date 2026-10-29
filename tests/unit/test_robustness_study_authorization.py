"""Isolated authorization fixtures never execute the real development study."""

import json
import subprocess

import pytest

from sirfex.robustness.identity import semantic_hash
from sirfex.robustness.scientific import (
    StudyGateError,
    scientific_execute,
    validate_authorization_record,
)


def _authorization():
    protocol = {
        "status": "FROZEN — AUTHORIZED FOR EXECUTION",
        "executable": True,
        "controller_inventory": [{"controller_id": "synthetic"}],
        "budget": {"wall_seconds_cap": 5},
    }
    protocol["semantic_sha256"] = semantic_hash(protocol)
    manifest = {"semantic_sha256": "manifest"}
    plan = {"semantic_sha256": "plan"}
    record = {
        "schema_version": "sirfex-study-authorization:1",
        "approval_status": "HUMAN_APPROVED",
        "approved_by": "synthetic-test-operator",
        "approved_at_utc": "2026-01-01T00:00:00+00:00",
        "protocol_sha256": protocol["semantic_sha256"],
        "frozen_commit": "a" * 40,
        "scenario_manifest_sha256": manifest["semantic_sha256"],
        "controller_inventory_sha256": semantic_hash(protocol["controller_inventory"]),
        "resource_limits_sha256": semantic_hash(protocol["budget"]),
        "work_plan_sha256": plan["semantic_sha256"],
    }
    record["semantic_sha256"] = semantic_hash(record)
    return protocol, manifest, plan, record


def test_matching_synthetic_authorization_identity_and_mutations():
    protocol, manifest, plan, record = _authorization()
    assert (
        validate_authorization_record(record, protocol, manifest, plan, frozen_commit="a" * 40)
        == record["semantic_sha256"]
    )
    for field, changed in (
        ("protocol_sha256", "wrong"),
        ("controller_inventory_sha256", "wrong"),
        ("resource_limits_sha256", "wrong"),
        ("work_plan_sha256", "wrong"),
        ("frozen_commit", "b" * 40),
    ):
        modified = {**record, field: changed}
        modified["semantic_sha256"] = semantic_hash(
            {key: value for key, value in modified.items() if key != "semantic_sha256"}
        )
        with pytest.raises(StudyGateError, match="does not match"):
            validate_authorization_record(
                modified, protocol, manifest, plan, frozen_commit="a" * 40
            )
    with pytest.raises(StudyGateError, match="SHA-256 mismatch"):
        validate_authorization_record(
            {**record, "approved_by": "changed"},
            protocol,
            manifest,
            plan,
            frozen_commit="a" * 40,
        )


def test_committed_bytes_and_missing_record_fail_before_synthetic_execution(tmp_path):
    protocol, manifest, _, _ = _authorization()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    subprocess.run(["git", "add", "synthetic.json"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=CI Test",
            "-c",
            "user.email=ci@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    kwargs = {
        "protocol_file": path,
        "approval_digest": protocol["semantic_sha256"],
        "frozen_commit": commit,
        "output_dir": tmp_path / "output",
    }
    with pytest.raises(StudyGateError, match="Missing independent authorization record"):
        scientific_execute(protocol, manifest, tmp_path, **kwargs)
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    with pytest.raises(StudyGateError, match="Working protocol differs"):
        scientific_execute(protocol, manifest, tmp_path, **kwargs)
    with pytest.raises(StudyGateError, match="not committed"):
        scientific_execute(protocol, manifest, tmp_path, **{**kwargs, "frozen_commit": "b" * 40})
