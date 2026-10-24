"""Deterministic random and Optuna TPE candidate strategies."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from sirfex.search.space import SearchSpace


class SearchStrategy(Protocol):
    identifier: str

    def propose(self, space: SearchSpace) -> dict[str, Any]: ...

    def observe(self, value: float | None, *, failed: bool = False) -> None: ...


class RandomSearchStrategy:
    identifier = "random"

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def propose(self, space: SearchSpace) -> dict[str, Any]:
        return space.sample_parameters(self.rng)

    def observe(self, value: float | None, *, failed: bool = False) -> None:
        del value, failed


class TPESearchStrategy:
    identifier = "tpe"

    def __init__(self, seed: int, direction: str) -> None:
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError(
                "Optuna is required for the tpe strategy; install sirfex[search]"
            ) from exc
        if direction not in {"maximize", "minimize"}:
            raise ValueError(f"Unsupported optimization direction: {direction}")
        self.seed = seed
        self._optuna = optuna
        self.study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        self._active_trial = None

    def propose(self, space: SearchSpace) -> dict[str, Any]:
        self._active_trial = self.study.ask()
        values: dict[str, Any] = dict(space.fixed_parameters)
        for parameter in space.parameters:
            if parameter.type_name == "float":
                values[parameter.name] = self._active_trial.suggest_float(
                    parameter.name, float(parameter.low), float(parameter.high)
                )
            elif parameter.type_name == "integer":
                values[parameter.name] = self._active_trial.suggest_int(
                    parameter.name, int(parameter.low), int(parameter.high)
                )
            elif parameter.type_name == "categorical":
                values[parameter.name] = self._active_trial.suggest_categorical(
                    parameter.name, list(parameter.choices)
                )
            else:
                values[parameter.name] = parameter.fixed
        return values

    def observe(self, value: float | None, *, failed: bool = False) -> None:
        if self._active_trial is None:
            return
        if failed or value is None:
            self.study.tell(self._active_trial, state=self._optuna.trial.TrialState.FAIL)
        else:
            self.study.tell(self._active_trial, float(value))
        self._active_trial = None
