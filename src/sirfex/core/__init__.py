"""Core protocols, execution, and result types."""

from sirfex.core.failures import EvidenceStatus, ExecutionStatus, FailureCategory, FailureRecord
from sirfex.core.interfaces import ControllerAdapter, EnvironmentAdapter, Scenario
from sirfex.core.results import EpisodeResult
from sirfex.core.runner import EvaluationRunner
from sirfex.core.scenarios import ParameterSpec, ScenarioValidationError

__all__ = [
    "ControllerAdapter",
    "EnvironmentAdapter",
    "EpisodeResult",
    "EvaluationRunner",
    "ExecutionStatus",
    "EvidenceStatus",
    "FailureCategory",
    "FailureRecord",
    "ParameterSpec",
    "Scenario",
    "ScenarioValidationError",
]
