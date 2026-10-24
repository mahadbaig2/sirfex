"""Small reusable scenario specification and validation helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from sirfex.core.interfaces import Scenario


class ScenarioValidationError(ValueError):
    """Raised when a scenario cannot be safely executed."""


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    type_name: str
    default: int | float | bool
    unit: str
    description: str
    minimum: float | None = None
    maximum: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type_name,
            "default": self.default,
            "unit": self.unit,
            "description": self.description,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def ensure_jsonable(value: Any, field_name: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ScenarioValidationError(f"{field_name} must be JSON-serializable") from exc


def validate_scenario_basics(scenario: Scenario, environment_id: str, version: str) -> None:
    if scenario.environment_id != environment_id:
        raise ScenarioValidationError(
            f"Scenario targets {scenario.environment_id!r}, expected {environment_id!r}"
        )
    if scenario.environment_version not in (None, version):
        raise ScenarioValidationError(
            f"Unsupported environment version {scenario.environment_version!r}; "
            f"expected {version!r}"
        )
    if isinstance(scenario.seed, bool) or not isinstance(scenario.seed, int):
        raise ScenarioValidationError("seed must be an integer, not a boolean")
    if not isinstance(scenario.schema_version, str) or not scenario.schema_version:
        raise ScenarioValidationError("schema_version must be a non-empty string")
    if not isinstance(scenario.parameters, dict):
        raise ScenarioValidationError("parameters must be an object")
    ensure_jsonable(scenario.to_dict(), "scenario")


def validate_number(value: Any, name: str, *, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        expected = "integer" if integer else "number"
        raise ScenarioValidationError(f"{name} must be a finite {expected}")
    if integer and not isinstance(value, int):
        raise ScenarioValidationError(f"{name} must be an integer")
    if not math.isfinite(float(value)):
        raise ScenarioValidationError(f"{name} must be finite")


def validate_parameter_specs(
    parameters: dict[str, Any], specs: tuple[ParameterSpec, ...]
) -> dict[str, Any]:
    spec_by_name = {spec.name: spec for spec in specs}
    unknown = sorted(set(parameters) - set(spec_by_name))
    if unknown:
        raise ScenarioValidationError(f"Unknown scenario parameter(s): {', '.join(unknown)}")
    resolved = {spec.name: spec.default for spec in specs}
    for name, value in parameters.items():
        spec = spec_by_name[name]
        validate_number(value, name, integer=spec.type_name == "integer")
        if spec.minimum is not None and value < spec.minimum:
            raise ScenarioValidationError(f"{name} must be >= {spec.minimum}")
        if spec.maximum is not None and value > spec.maximum:
            raise ScenarioValidationError(f"{name} must be <= {spec.maximum}")
        resolved[name] = value
    return resolved
