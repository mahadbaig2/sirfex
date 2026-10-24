"""Versioned JSON experiment manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

MANIFEST_SCHEMA_VERSION = "1"


@dataclass
class ExperimentManifest:
    experiment_id: str
    created_at_utc: str
    sirfex_version: str
    environment_id: str
    environment_version: str
    controller_id: str
    controller_metadata: dict[str, Any]
    controller_config: dict[str, Any]
    scenario: dict[str, Any]
    evaluation_config: dict[str, Any]
    dependencies: dict[str, str]
    execution_status: str
    outcome: str
    failures: list[dict[str, Any]] = field(default_factory=list)
    artifact_refs: dict[str, str] = field(default_factory=dict)
    git_commit: str | None = None
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExperimentManifest:
        if not isinstance(value, dict):
            raise ValueError("Experiment manifest must be a JSON object")
        if value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported manifest schema version: {value.get('schema_version')!r}"
            )
        required = {
            "experiment_id",
            "created_at_utc",
            "sirfex_version",
            "environment_id",
            "environment_version",
            "controller_id",
            "controller_metadata",
            "controller_config",
            "scenario",
            "evaluation_config",
            "dependencies",
            "execution_status",
            "outcome",
        }
        missing = required - value.keys()
        if missing:
            raise ValueError(f"Manifest is missing required fields: {sorted(missing)}")
        return cls(
            **{key: value[key] for key in required}
            | {
                "failures": value.get("failures", []),
                "artifact_refs": value.get("artifact_refs", {}),
                "git_commit": value.get("git_commit"),
                "schema_version": value["schema_version"],
            }
        )
