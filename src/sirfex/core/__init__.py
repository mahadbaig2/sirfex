"""Core protocols, execution, and result types."""

from sirfex.core.failures import (
    EvaluationError,
    EvidenceStatus,
    ExecutionStatus,
    FailureCategory,
    FailureRecord,
)
from sirfex.core.interfaces import ControllerAdapter, EnvironmentAdapter, Scenario
from sirfex.core.results import EpisodeResult
from sirfex.core.runner import EvaluationRunner, managed_environment
from sirfex.core.scenarios import ParameterSpec, ScenarioValidationError

__all__ = [
    "ControllerAdapter",
    "EnvironmentAdapter",
    "EpisodeResult",
    "EvaluationRunner",
    "EvaluationError",
    "managed_environment",
    "ExecutionStatus",
    "EvidenceStatus",
    "FailureCategory",
    "FailureRecord",
    "ParameterSpec",
    "Scenario",
    "ScenarioValidationError",
]
