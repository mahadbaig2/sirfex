"""Stable semantic identities, independent of runtime and artifact paths."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sirfex.core import Scenario
from sirfex.environments.pump import PumpEnvironment


def semantic_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def scenario_identity(value: Scenario | dict) -> str:
    scenario = value if isinstance(value, Scenario) else Scenario.from_dict(value)
    parameters = PumpEnvironment().validate_scenario(scenario)
    return semantic_hash(
        {
            "environment_id": scenario.environment_id,
            "environment_version": PumpEnvironment.version,
            "parameters": parameters,
            "seed": scenario.seed,
        }
    )
