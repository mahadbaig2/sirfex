"""Validated, bounded scenario search spaces."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sirfex.core import Scenario, ScenarioValidationError

PARAMETER_TYPES = {"float", "integer", "categorical", "fixed"}


@dataclass(frozen=True)
class ParameterDefinition:
    name: str
    type_name: str
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[str | int | float | bool, ...] = ()
    fixed: str | int | float | bool | None = None
    unit: str = ""
    description: str = ""

    def validate(self) -> None:
        if not self.name or self.type_name not in PARAMETER_TYPES:
            raise ScenarioValidationError(f"Invalid search parameter definition: {self.name!r}")
        if self.type_name in {"float", "integer"}:
            if self.low is None or self.high is None:
                raise ScenarioValidationError(f"{self.name} requires low and high bounds")
            if isinstance(self.low, bool) or isinstance(self.high, bool):
                raise ScenarioValidationError(f"{self.name} bounds cannot be booleans")
            if not math.isfinite(float(self.low)) or not math.isfinite(float(self.high)):
                raise ScenarioValidationError(f"{self.name} bounds must be finite")
            if self.low > self.high:
                raise ScenarioValidationError(f"{self.name} low bound must not exceed high bound")
            if self.type_name == "integer" and (
                not isinstance(self.low, int) or not isinstance(self.high, int)
            ):
                raise ScenarioValidationError(f"{self.name} integer bounds must be integers")
        elif self.type_name == "categorical":
            if not self.choices or len(set(self.choices)) != len(self.choices):
                raise ScenarioValidationError(f"{self.name} choices must be non-empty and unique")
        elif self.type_name == "fixed":
            if self.fixed is None or isinstance(self.fixed, (dict, list)):
                raise ScenarioValidationError(
                    f"{self.name} fixed parameter requires a scalar value"
                )

    def sample(self, rng: np.random.Generator) -> Any:
        self.validate()
        if self.type_name == "float":
            return float(rng.uniform(float(self.low), float(self.high)))
        if self.type_name == "integer":
            return int(rng.integers(int(self.low), int(self.high) + 1))
        if self.type_name == "categorical":
            return self.choices[int(rng.integers(0, len(self.choices)))]
        return self.fixed

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "type": self.type_name,
            "unit": self.unit,
            "description": self.description,
        }
        if self.type_name in {"float", "integer"}:
            value.update({"low": self.low, "high": self.high})
        elif self.type_name == "categorical":
            value["choices"] = list(self.choices)
        else:
            value["value"] = self.fixed
        return value


@dataclass(frozen=True)
class SearchSpace:
    environment_id: str
    environment_version: str
    parameters: tuple[ParameterDefinition, ...] = ()
    schema_version: str = "1"
    fixed_parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def validate(self) -> None:
        if not self.environment_id or not self.environment_version:
            raise ScenarioValidationError(
                "Search space requires environment_id and environment_version"
            )
        if self.schema_version != "1":
            raise ScenarioValidationError(
                f"Unsupported search-space schema version: {self.schema_version}"
            )
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ScenarioValidationError("Search-space parameter names must be unique")
        if set(names) & set(self.fixed_parameters):
            raise ScenarioValidationError("A parameter cannot be both searched and fixed")
        for parameter in self.parameters:
            parameter.validate()
        try:
            json.dumps(self.to_dict(), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ScenarioValidationError("Search space must be JSON-serializable") from exc

    def sample_parameters(self, rng: np.random.Generator) -> dict[str, Any]:
        self.validate()
        return {
            **self.fixed_parameters,
            **{parameter.name: parameter.sample(rng) for parameter in self.parameters},
        }

    def scenario(
        self, parameters: dict[str, Any], seed: int, *, description: str | None = None
    ) -> Scenario:
        self.validate()
        expected = {parameter.name for parameter in self.parameters} | set(self.fixed_parameters)
        if set(parameters) != expected:
            raise ScenarioValidationError(
                "Candidate parameters do not match the declared search space"
            )
        for name, value in self.fixed_parameters.items():
            if parameters[name] != value:
                raise ScenarioValidationError(f"Candidate changed fixed parameter: {name}")
        for definition in self.parameters:
            value = parameters[definition.name]
            if definition.type_name in {"float", "integer"}:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not definition.low <= value <= definition.high
                    or (definition.type_name == "integer" and not isinstance(value, int))
                ):
                    raise ScenarioValidationError(
                        f"Candidate outside search bounds: {definition.name}"
                    )
            elif definition.type_name == "categorical" and value not in definition.choices:
                raise ScenarioValidationError(
                    f"Candidate outside search choices: {definition.name}"
                )
            elif definition.type_name == "fixed" and value != definition.fixed:
                raise ScenarioValidationError(
                    f"Candidate changed fixed parameter: {definition.name}"
                )
        return Scenario(
            environment_id=self.environment_id,
            environment_version=self.environment_version,
            parameters=dict(parameters),
            seed=seed,
            description=description or self.description,
            tags=("search",),
            provenance={"search_space_schema": self.schema_version},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "environment_version": self.environment_version,
            "schema_version": self.schema_version,
            "description": self.description,
            "parameters": {parameter.name: parameter.to_dict() for parameter in self.parameters},
            "fixed_parameters": dict(self.fixed_parameters),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SearchSpace:
        if not isinstance(value, dict):
            raise ScenarioValidationError("Search space must be a JSON object")
        parameters = []
        for name, definition in value.get("parameters", {}).items():
            if not isinstance(definition, dict):
                raise ScenarioValidationError(f"Parameter {name} definition must be an object")
            type_name = definition.get("type")
            parameters.append(
                ParameterDefinition(
                    name=name,
                    type_name=type_name,
                    low=definition.get("low"),
                    high=definition.get("high"),
                    choices=tuple(definition.get("choices", [])),
                    fixed=definition.get("value"),
                    unit=definition.get("unit", ""),
                    description=definition.get("description", ""),
                )
            )
        space = cls(
            environment_id=value.get("environment_id", ""),
            environment_version=value.get("environment_version", ""),
            parameters=tuple(parameters),
            schema_version=value.get("schema_version", "1"),
            fixed_parameters=dict(value.get("fixed_parameters", {})),
            description=value.get("description", ""),
        )
        space.validate()
        return space

    @classmethod
    def from_file(cls, path: str) -> SearchSpace:
        with open(path, encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))
