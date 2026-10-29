"""Archive selected differential episodes in the existing ExperimentStore format."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sirfex import __version__
from sirfex.core import EpisodeResult, Scenario
from sirfex.experiments.manifests import ExperimentManifest
from sirfex.experiments.replay import compare_results, replay_experiment
from sirfex.experiments.storage import ExperimentStore, StorageError
from sirfex.robustness.differential import classify_pair
from sirfex.robustness.identity import scenario_identity, semantic_hash
from sirfex.robustness.margins import evaluate_margin
from sirfex.robustness.store import controller_identity
from sirfex.training.feasibility import analyze_feasibility

MATCHES = {"exact_match", "tolerance_match"}


class EvidenceArchiveError(RuntimeError):
    """Saved evidence is missing, corrupt, or incompatible with its identity."""


def _validate_stored(
    store: ExperimentStore,
    experiment_id: str,
    scenario_id: str,
    controller_id: str,
    expected_hash: str | None = None,
    expected_controller_identity: str | None = None,
    expected_protocol_sha256: str | None = None,
) -> dict:
    try:
        stored = store.load(experiment_id)
        manifest = stored.manifest
        if (
            scenario_identity(stored.scenario) != scenario_id
            or scenario_identity(manifest.scenario) != scenario_id
            or scenario_identity(stored.result["scenario"]) != scenario_id
            or manifest.environment_version != stored.scenario.environment_version
            or manifest.controller_id != controller_id
        ):
            raise EvidenceArchiveError("Archived scenario or controller identity changed")
        recorded_identity = controller_identity(
            {"id": manifest.controller_id, "config": manifest.controller_config}
        )
        if (
            recorded_identity != manifest.controller_metadata.get("identity_sha256")
            or (
                expected_controller_identity is not None
                and recorded_identity != expected_controller_identity
            )
            or (
                expected_protocol_sha256 is not None
                and manifest.artifact_refs.get("protocol_sha256") != expected_protocol_sha256
            )
        ):
            raise EvidenceArchiveError("Archived checkpoint or protocol identity changed")
        digest = semantic_hash(stored.result)
        if digest != manifest.artifact_refs.get("episode_sha256") or (
            expected_hash is not None and digest != expected_hash
        ):
            raise EvidenceArchiveError("Archived trajectory/result hash changed")
        margin = evaluate_margin(stored.result)
        if margin.evidence_status != "valid":
            raise EvidenceArchiveError("Archived episode is incomplete")
        return {
            "experiment_id": experiment_id,
            "episode_sha256": digest,
            "margin": margin.to_dict(),
        }
    except (StorageError, KeyError, ValueError, TypeError) as exc:
        raise EvidenceArchiveError("Archived experiment is missing or corrupt") from exc


def _save_episode(
    store: ExperimentStore,
    protocol_sha256: str,
    spec: dict,
    scenario: Scenario,
    role: str,
    result: EpisodeResult,
) -> dict:
    scenario_id = scenario_identity(scenario)
    if scenario_identity(result.scenario) != scenario_id or result.controller_id != spec["id"]:
        raise EvidenceArchiveError("Evaluation changed scenario or controller")
    margin = evaluate_margin(result)
    if margin.evidence_status != "valid":
        raise EvidenceArchiveError("Cannot archive incomplete differential evidence")
    experiment_id = (
        "robust_"
        + semantic_hash(
            {
                "protocol_sha256": protocol_sha256,
                "scenario_id": scenario_id,
                "controller_identity": controller_identity(spec),
                "role": role,
            }
        )[:24]
    )
    digest = semantic_hash(result.to_dict())
    manifest = ExperimentManifest(
        experiment_id=experiment_id,
        created_at_utc=datetime.now(UTC).isoformat(),
        sirfex_version=__version__,
        environment_id="pump",
        environment_version=scenario.environment_version,
        controller_id=spec["id"],
        controller_metadata={"identity_sha256": controller_identity(spec)},
        controller_config=spec["config"],
        scenario=scenario.to_dict(),
        evaluation_config={
            "max_steps": scenario.parameters["horizon"],
            "deterministic": True,
            "record_trace": True,
        },
        dependencies={},
        execution_status=result.execution_status.value,
        outcome=result.outcome,
        artifact_refs={"episode_sha256": digest, "protocol_sha256": protocol_sha256},
    )
    try:
        store.save(manifest, scenario, result)
    except StorageError as exc:
        if "already exists" not in str(exc):
            raise EvidenceArchiveError("Could not archive episode") from exc
    return _validate_stored(
        store,
        experiment_id,
        scenario_id,
        spec["id"],
        digest,
        controller_identity(spec),
        protocol_sha256,
    )


def archive_selected_findings(
    result: dict,
    controllers: list[dict],
    evaluator,
    directory: Path,
    protocol_sha256: str,
    *,
    label_specs: dict[str, dict] | None = None,
) -> dict:
    """Archive only ranked headlines; downgrade B unless repeats and replay verify."""
    store = ExperimentStore(directory / "experiments")
    specs = label_specs if label_specs is not None else {item["id"]: item for item in controllers}
    records = {row["scenario_id"]: row for row in result["records"] if row["status"] == "evaluated"}
    promoted = []
    downgraded = []
    for classification in ("class_b_unverified", "class_b", "class_a"):
        for finding in result["findings"][classification]:
            row = records[finding["scenario_id"]]
            scenario = Scenario.from_dict(row["scenario"])
            if scenario_identity(scenario) != finding["scenario_id"]:
                raise EvidenceArchiveError("Candidate scenario identity changed")
            feasibility = analyze_feasibility(scenario)
            if feasibility["finite_horizon_viable"] is not True:
                raise EvidenceArchiveError(
                    "Headline scenario lacks proven finite-horizon viability"
                )
            episodes = {}
            for label, controller_key, spec in (
                ("x", result["controller_x"], specs[result["controller_x"]]),
                ("y", result["controller_y"], specs[result["controller_y"]]),
            ):
                for role in ("original", "repeat"):
                    episode = evaluator(controller_key, scenario)
                    episodes[f"{label}_{role}"] = (
                        episode,
                        _save_episode(
                            store, protocol_sha256, spec, scenario, f"{label}_{role}", episode
                        ),
                    )
            repeat_statuses = {
                label: compare_results(
                    episodes[f"{label}_original"][0].to_dict(),
                    episodes[f"{label}_repeat"][0].to_dict(),
                ).status
                for label in ("x", "y")
            }
            regeneration = {
                role: replay_experiment(store, saved["experiment_id"]).status
                for role, (_, saved) in episodes.items()
            }
            candidate_matches = all(
                episodes[f"{label}_original"][1]["margin"] == row[f"margin_{label}"]
                for label in ("x", "y")
            )
            independent_ok = all(status in MATCHES for status in regeneration.values())
            reconfirmed = classify_pair(
                evaluate_margin(episodes["x_original"][0]),
                evaluate_margin(episodes["y_original"][0]),
                finite_horizon_viable=True,
                near_miss_delta=result.get("near_miss_delta", 0.05),
                replay_statuses=(repeat_statuses["x"], repeat_statuses["y"]),
                independent_replay_verified=independent_ok,
            )
            verified = (
                candidate_matches
                and reconfirmed.classification
                == ("class_a" if classification == "class_a" else "class_b")
                and all(status in MATCHES for status in repeat_statuses.values())
                and independent_ok
            )
            evidence = {
                "schema_version": "pump-differential-archive:1",
                "scenario_id": finding["scenario_id"],
                "controller_x_identity": controller_identity(specs[result["controller_x"]]),
                "controller_y_identity": controller_identity(specs[result["controller_y"]]),
                "experiments": {role: saved for role, (_, saved) in episodes.items()},
                "persisted_repeat_statuses": repeat_statuses,
                "independent_regeneration_statuses": regeneration,
                "candidate_margin_match": candidate_matches,
                "independent_classification": reconfirmed.classification,
                "feasibility": feasibility["classification"],
                "verification_status": "independently_replayed" if verified else "unverified",
                "replay_command": (
                    "sirfex experiments --store <run>/experiments replay <experiment-id>"
                ),
            }
            finding["durable_evidence"] = evidence
            row["finding"] = finding
            if classification == "class_b_unverified" and verified:
                finding["classification"] = "class_b"
                finding["reason"] = "Persisted repeats and independent regeneration verified"
                promoted.append(finding)
            elif classification == "class_b" and not verified:
                finding["classification"] = "class_b_unverified"
                finding["reason"] = "Durable independent replay did not verify the candidate"
                downgraded.append(finding)
            elif classification == "class_b_unverified" and not verified:
                finding["reason"] = "Durable independent replay did not verify the candidate"
    if promoted:
        result["findings"]["class_b_unverified"] = [
            finding
            for finding in result["findings"]["class_b_unverified"]
            if finding not in promoted
        ]
        result["findings"]["class_b"].extend(promoted)
    if downgraded:
        result["findings"]["class_b"] = [
            finding for finding in result["findings"]["class_b"] if finding not in downgraded
        ]
        result["findings"]["class_b_unverified"].extend(downgraded)
    result["durable_headline_selection"] = {
        "schema_version": "pump-differential-headlines:1",
        "selected_class_a": len(result["findings"]["class_a"]),
        "verified_class_b": len(result["findings"]["class_b"]),
        "unverified_class_b": len(result["findings"]["class_b_unverified"]),
    }
    return result
