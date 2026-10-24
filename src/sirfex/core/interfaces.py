"""Small, environment-agnostic contracts used by the evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Scenario:
    """A versioned, serializable scenario configuration for one episode."""

    environment_id: str
    parameters: dict[str, float | int | bool | str] = field(default_factory=dict)
    seed: int = 0
    environment_version: str | None = None
    schema_version: str = "1"
    description: str | None = None
    tags: tuple[str, ...] = ()
    provenance: dict[str, str] = field(default_factory=dict)

    @property
    def version(self) -> str:
        """Backward-compatible alias for the scenario schema version."""
        return self.schema_version

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "parameters": dict(self.parameters),
            "seed": self.seed,
            "environment_version": self.environment_version,
            "schema_version": self.schema_version,
            "description": self.description,
            "tags": list(self.tags),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Scenario:
        if not isinstance(value, dict):
            raise ValueError("Scenario must be a JSON object")
        required = {"environment_id", "parameters", "seed"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"Scenario is missing required fields: {sorted(missing)}")
        tags = value.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError("Scenario tags must be a list of strings")
        return cls(
            environment_id=value["environment_id"],
            parameters=dict(value["parameters"]),
            seed=value["seed"],
            environment_version=value.get("environment_version"),
            schema_version=value.get("schema_version", value.get("version", "1")),
            description=value.get("description"),
            tags=tuple(tags),
            provenance=dict(value.get("provenance", {})),
        )


class ControllerAdapter(Protocol):
    """Controller contract consumed by :class:`EvaluationRunner`."""

    @property
    def id(self) -> str: ...

    def reset(self) -> None: ...

    def act(self, observation: Any, deterministic: bool = True) -> Any: ...

    def metadata(self) -> dict[str, Any]: ...


class EnvironmentAdapter(Protocol):
    """Environment contract consumed by :class:`EvaluationRunner`."""

    @property
    def id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def observation_space(self) -> Any: ...

    @property
    def action_space(self) -> Any: ...

    def reset(self, scenario: Scenario) -> tuple[Any, dict[str, Any]]: ...

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]: ...

    def metadata(self) -> dict[str, Any]: ...

    def evaluate_failure(
        self, total_reward: float, terminated: bool, info: dict[str, Any]
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...
