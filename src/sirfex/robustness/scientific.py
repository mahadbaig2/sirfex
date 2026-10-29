"""Checkpoint 2 study validation and deliberately gated execution entry point."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sirfex.core import Scenario
from sirfex.robustness.identity import scenario_identity, semantic_hash
from sirfex.training.checkpoints import sha256_file, verify_checkpoint


class StudyGateError(ValueError):
    """A proposed or unapproved scientific study must not run."""


def _checked_hash(value: dict) -> str:
    payload = {key: item for key, item in value.items() if key != "semantic_sha256"}
    digest = semantic_hash(payload)
    if value.get("semantic_sha256") != digest:
        raise StudyGateError("Protocol semantic SHA-256 mismatch")
    return digest


def validate_authorization_record(
    record: dict, protocol: dict, manifest: dict, plan: dict, *, frozen_commit: str
) -> str:
    """Check a separate operator record; its hash is integrity, not a signature."""
    if not isinstance(record, dict):
        raise StudyGateError("Malformed independent authorization record")
    digest = _checked_hash(record)
    try:
        approved_at = datetime.fromisoformat(record["approved_at_utc"])
        valid = (
            record["schema_version"] == "sirfex-study-authorization:1"
            and record["approval_status"] == "HUMAN_APPROVED"
            and isinstance(record["approved_by"], str)
            and len(record["approved_by"].strip()) >= 3
            and approved_at.tzinfo is not None
            and record["protocol_sha256"] == protocol["semantic_sha256"]
            and record["frozen_commit"] == frozen_commit
            and record["scenario_manifest_sha256"] == manifest["semantic_sha256"]
            and record["controller_inventory_sha256"]
            == semantic_hash(protocol["controller_inventory"])
            and record["resource_limits_sha256"] == semantic_hash(protocol["budget"])
            and record["work_plan_sha256"] == plan["semantic_sha256"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StudyGateError("Malformed independent authorization record") from exc
    if not valid:
        raise StudyGateError("Authorization record does not match frozen study identity")
    return digest


def validate_manifest(value: dict, protocol: dict) -> list[Scenario]:
    if (
        value.get("scope") != "development_only"
        or value.get("schema_version") != "phase37-development-manifest:1"
    ):
        raise StudyGateError("Unsupported or final scenario partition")
    if _checked_hash(value) != protocol.get("scenario_manifest_sha256"):
        raise StudyGateError("Scenario manifest identity mismatch")
    rows = value.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise StudyGateError("Incomplete development scenario manifest")
    if {row.get("partition") for row in rows} - {
        "training",
        "validation",
        "screening_stress",
        "historical_discoveries",
    }:
        raise StudyGateError("Unsupported scenario partition")
    counts = {
        name: sum(row["partition"] == name for row in rows)
        for name in ("training", "validation", "screening_stress", "historical_discoveries")
    }
    if counts != {
        "training": 12,
        "validation": 12,
        "screening_stress": 24,
        "historical_discoveries": 7,
    }:
        raise StudyGateError("Incomplete development scenario manifest")
    if manifest_leakage := value.get("leakage"):
        if manifest_leakage.get("passed") is not True:
            raise StudyGateError("Development partition leakage check failed")
    else:
        raise StudyGateError("Missing development partition leakage check")
    anchors = []
    for row in rows:
        scenario = Scenario.from_dict(row["scenario"])
        if scenario.environment_id != "pump" or scenario.environment_version != "sirfex-pump:1":
            raise StudyGateError("Unsupported scenario environment")
        declared = scenario.provenance.get("partition")
        if declared is not None and declared != row["partition"]:
            raise StudyGateError("Scenario partition provenance mismatch")
        labels = str(scenario.provenance).lower() + str(scenario.tags).lower()
        if any(word in labels for word in ("final", "holdout", "held_out")):
            raise StudyGateError("Sealed scenario provenance is unsupported")
        if scenario_identity(scenario) != row["scenario_id"]:
            raise StudyGateError("Scenario semantic identity mismatch")
        if row["partition"] == "validation":
            anchors.append(scenario)
    if [scenario_identity(s) for s in anchors] != protocol.get("anchors"):
        raise StudyGateError("Anchor order or identities changed")
    if len(anchors) != 12:
        raise StudyGateError("The proposed study requires exactly 12 validation anchors")
    return anchors


def dry_run(protocol: dict, manifest: dict, root: Path | str) -> dict:
    """Read-only input validation; no controller rollout or run ID."""
    root = Path(root)
    digest = _checked_hash(protocol)
    anchors = validate_manifest(manifest, protocol)
    if (
        protocol.get("scope") != "development_only"
        or protocol.get("environment_version") != "sirfex-pump:1"
    ):
        raise StudyGateError("Unsupported study scope or environment")
    inventory = protocol.get("controller_inventory", [])
    if len(inventory) != 17:
        raise StudyGateError("Incomplete controller inventory")
    grid = protocol.get("disturbance", {}).get("grid", [])
    if grid != [round(i * 0.025, 3) for i in range(25)]:
        raise StudyGateError("Proposed disturbance grid changed")
    pairs = protocol.get("differential_pairs", [])
    if len(pairs) != 7 or len({p["id"] for p in pairs}) != 7:
        raise StudyGateError("Differential pair design changed")
    search = protocol.get("search", {})
    if search.get("space") != "configs/search/pump_avoidable.json" or sha256_file(
        root / search["space"]
    ) != search.get("space_file_sha256"):
        raise StudyGateError("Differential search-space identity changed")
    budget = protocol.get("budget", {})
    if (
        budget.get("grid_episodes_upper") != 17 * 12 * 16 * 25
        or budget.get("development_stress_context_episodes") != 17 * 24
        or budget.get("differential_pairs") != 7
        or budget.get("total_episode_upper") != 88168
        or budget.get("maximum_transitions") != 88168 * 200
    ):
        raise StudyGateError("Study workload does not match exact proposed design")
    if (
        budget.get("total_episode_upper", 0) > 100000
        or budget.get("maximum_transitions", 0) > 20_000_000
        or budget.get("artifact_bytes_cap", 0) > 2 * 1024**3
        or budget.get("wall_seconds_cap", 0) > 8 * 3600
    ):
        raise StudyGateError("Study exceeds approved preflight budget ceilings")
    for row in inventory:
        if row["group"] == "reference":
            continue
        path = root / row["source_artifact"]
        if not path.is_file():
            raise StudyGateError(f"Missing checkpoint: {row['source_key']}")
        verify_checkpoint(path, row["sha256"])
    from sirfex.robustness.study import build_plan

    plan = build_plan(protocol, manifest, root, inputs_verified=True)
    return {
        "status": "dry_run_passed_no_scientific_execution",
        "protocol_sha256": digest,
        "anchors": len(anchors),
        "controllers": len(inventory),
        "pairs": len(pairs),
        "episode_upper": budget["total_episode_upper"],
        "work_plan_sha256": plan["semantic_sha256"],
        "work_counts": plan["work_counts"],
    }


def scientific_execute(
    protocol: dict,
    manifest: dict,
    root: Path | str,
    *,
    protocol_file: Path | str,
    approval_digest: str | None,
    authorization_record: Path | str | None = None,
    frozen_commit: str | None = None,
    manifest_file: Path | str | None = None,
    output_dir: Path | str | None = None,
    resume: bool = False,
) -> dict:
    """Refuse candidate and uncommitted protocols before any scientific side effect.

    A future frozen protocol can enter the execution adapter only after a separately
    supplied digest and committed-byte check. This milestone issues no such protocol.
    """
    digest = _checked_hash(protocol)
    if (
        protocol.get("status") != "FROZEN — AUTHORIZED FOR EXECUTION"
        or protocol.get("executable") is not True
    ):
        raise StudyGateError("Draft/candidate protocol cannot execute")
    if approval_digest is None:
        raise StudyGateError("Missing explicit operator approval digest")
    if approval_digest != digest:
        raise StudyGateError("Approval digest does not match protocol")
    if frozen_commit is None or not isinstance(frozen_commit, str) or len(frozen_commit) != 40:
        raise StudyGateError("Frozen protocol is not committed at a declared commit")
    root = Path(root).resolve()
    file = Path(protocol_file).resolve()
    try:
        relative = file.relative_to(root).as_posix()
        committed = subprocess.check_output(
            ["git", "show", f"{frozen_commit}:{relative}"],
            cwd=root,
            timeout=5,
        )
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", frozen_commit, "HEAD"],
            cwd=root,
            timeout=5,
            check=True,
            capture_output=True,
        )
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        raise StudyGateError("Frozen protocol is not committed at HEAD") from exc
    if file.read_bytes() != committed or json.loads(committed) != protocol:
        raise StudyGateError("Working protocol differs from committed frozen protocol")
    if authorization_record is None:
        raise StudyGateError("Missing independent authorization record")
    record_file = Path(authorization_record).resolve()
    if record_file == file:
        raise StudyGateError("Authorization record must be separate from protocol")
    try:
        record = json.loads(record_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StudyGateError("Authorization record is missing or invalid") from exc
    dry_run(protocol, manifest, root)
    if output_dir is None:
        raise StudyGateError("Scientific execution requires an explicit output directory")
    from sirfex.robustness.study import build_plan
    from sirfex.robustness.study_execute import StudyEngine
    from sirfex.robustness.study_store import StudyBudgetExhausted

    plan = build_plan(protocol, manifest, root, inputs_verified=True)
    record_digest = validate_authorization_record(
        record, protocol, manifest, plan, frozen_commit=frozen_commit
    )
    engine = StudyEngine(protocol, manifest, plan, root, output_dir)
    try:
        from sirfex.robustness.store import RobustnessStorageError, _atomic_json

        saved_path = engine.store.directory / "authorization.json"
        if saved_path.exists():
            if json.loads(saved_path.read_text(encoding="utf-8")) != record:
                raise RobustnessStorageError("Stored study authorization changed")
        else:
            engine.store._check_storage(record)
            _atomic_json(saved_path, record)
        request = {
            "schema_version": "sirfex-study-invocation-request:1",
            "mode": "resume" if resume else "scientific-execute",
            "protocol_file": str(file),
            "manifest_file": str(Path(manifest_file).resolve()) if manifest_file else None,
            "authorization_file": str(record_file),
            "output_dir": str(Path(output_dir).resolve()),
            "approval_digest": approval_digest,
            "frozen_commit": frozen_commit,
            "authorization_record_sha256": record_digest,
            "work_plan_sha256": plan["semantic_sha256"],
        }
        request["semantic_sha256"] = semantic_hash(request)
        request_path = engine.store.directory / f"invocation_request_{uuid4().hex}.json"
        engine.store._check_storage(request)
        _atomic_json(request_path, request)
        try:
            engine.execute_fixed_and_grid(resume=resume)
            engine.execute_differential(resume=resume)
            report = engine.execute_report(resume=resume)
            result = {
                "status": report["execution_status"],
                "run_id": engine.store.run_id,
                "report_sha256": report["semantic_sha256"],
            }
        except StudyBudgetExhausted as exc:
            partial = engine.regenerate_report(persist=False)
            result = {
                "status": str(exc),
                "run_id": engine.store.run_id,
                "report_sha256": partial["semantic_sha256"],
            }
    finally:
        engine.close()
    result["work_status"] = engine.store.status()
    return result


def regenerate_study_report(
    protocol: dict, manifest: dict, root: Path | str, output_dir: Path | str
) -> dict:
    """Read verified artifacts and regenerate a partial or complete report."""
    from sirfex.robustness.study import build_plan
    from sirfex.robustness.study_execute import StudyEngine

    plan = build_plan(protocol, manifest, root)
    run_name = f"study_{plan['semantic_sha256'][:20]}"
    if not (Path(output_dir).resolve() / run_name / "plan.json").is_file():
        raise StudyGateError("No stored scientific study exists for this plan")
    engine = StudyEngine(protocol, manifest, plan, root, output_dir)
    try:
        report = engine.regenerate_report(persist=True)
        return {
            "status": report["execution_status"],
            "run_id": engine.store.run_id,
            "report_sha256": report["semantic_sha256"],
        }
    finally:
        engine.close()
