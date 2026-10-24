"""Small, environment-agnostic contracts used by the evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Scenario:
    """A serializable scenario configuration for one episode."""

    environment_id: str
    parameters: dict[str, float | int | bool | str] = field(default_factory=dict)
    seed: int = 0
    version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "parameters": dict(self.parameters),
            "seed": self.seed,
            "version": self.version,
        }


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
