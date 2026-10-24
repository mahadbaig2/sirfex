"""Stable failure and execution semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class ExecutionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class FailureCategory(StrEnum):
    NOMINAL = "nominal"
    CONSTRAINT_VIOLATION = "constraint_violation"
    OBJECTIVE_FAILURE = "objective_failure"
    INVALID_ACTION = "invalid_action"
    INVALID_OBSERVATION = "invalid_observation"
    NUMERICAL_INSTABILITY = "numerical_instability"
    CONTROLLER_EXCEPTION = "controller_exception"
    ENVIRONMENT_EXCEPTION = "environment_exception"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class EvidenceStatus(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FailureRecord:
    category: FailureCategory
    explanation: str
    step: int | None = None
    measurements: dict[str, Any] | None = None
    threshold: str | None = None
    evidence_source: str = "environment"
    evidence_status: EvidenceStatus = EvidenceStatus.OBSERVED

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["category"] = self.category.value
        value["evidence_status"] = self.evidence_status.value
        return value
