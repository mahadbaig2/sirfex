from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from sirfex.controllers import PumpRuleController, RandomController
from sirfex.core import EvaluationRunner, Scenario
from sirfex.environments import LunarLanderEnvironment, PumpEnvironment


def _system_info() -> dict[str, Any]:
    result: dict[str, Any] = {
        "os": platform.platform(),
        "python": sys.version,
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    try:
        import psutil

        result.update(
            {
                "logical_cpus": psutil.cpu_count(logical=True),
                "physical_cpus": psutil.cpu_count(logical=False),
                "memory_total_bytes": psutil.virtual_memory().total,
                "memory_available_bytes": psutil.virtual_memory().available,
            }
        )
    except ImportError:
        result["psutil"] = "unavailable"
    return result


def _run_environment(name: str, episodes: int) -> dict[str, Any]:
    environment = LunarLanderEnvironment() if name == "lunarlander" else PumpEnvironment()
    controller = (
        RandomController(environment.action_space)
        if name == "lunarlander"
        else PumpRuleController()
    )
    runner = EvaluationRunner()
    start = time.perf_counter()
    results = []
    try:
        for seed in range(episodes):
            result = runner.run_episode(
                environment, controller, Scenario(name, seed=seed), record_trace=False
            )
            results.append(result)
    finally:
        environment.close()
    elapsed = time.perf_counter() - start
    steps = sum(result.episode_length for result in results)
    return {
        "episodes": episodes,
        "elapsed_seconds": elapsed,
        "episodes_per_second": episodes / elapsed if elapsed else 0.0,
        "steps_per_second": steps / elapsed if elapsed else 0.0,
        "mean_reward": sum(result.episode_reward for result in results) / episodes,
        "outcomes": {
            outcome: sum(result.outcome == outcome for result in results)
            for outcome in {r.outcome for r in results}
        },
    }


def run_benchmark(profile: str = "quick", output_dir: Path | None = None) -> dict[str, Any]:
    episodes = 5 if profile == "quick" else 25
    report = {
        "profile": profile,
        "episodes_per_environment": episodes,
        "system": _system_info(),
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "gymnasium")
            if importlib.util.find_spec(package) is not None
        },
        "environments": {
            "pump": _run_environment("pump", episodes),
            "lunarlander": _run_environment("lunarlander", episodes),
        },
        "ppo_smoke_test": "not run by default; use scripts/ppo_smoke_test.py",
    }
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
