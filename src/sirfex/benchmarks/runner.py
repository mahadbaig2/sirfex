from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from sirfex.controllers import PumpRuleController, RandomController
from sirfex.core import EvaluationRunner, Scenario
from sirfex.environments import LunarLanderEnvironment, PumpEnvironment
from sirfex.experiments.replay import replay_experiment
from sirfex.experiments.service import run_and_save_experiment
from sirfex.experiments.storage import ExperimentStore


def _rss_bytes() -> int | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except ImportError:
        return None


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
    memory_before = _rss_bytes()
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
    memory_after = _rss_bytes()
    steps = sum(result.episode_length for result in results)
    return {
        "episodes": episodes,
        "elapsed_seconds": elapsed,
        "episodes_per_second": episodes / elapsed if elapsed else 0.0,
        "steps_per_second": steps / elapsed if elapsed else 0.0,
        "memory_before_bytes": memory_before,
        "memory_after_bytes": memory_after,
        "memory_delta_bytes": (
            memory_after - memory_before
            if memory_before is not None and memory_after is not None
            else None
        ),
        "mean_reward": sum(result.episode_reward for result in results) / episodes,
        "outcomes": {
            outcome: sum(result.outcome == outcome for result in results)
            for outcome in {r.outcome for r in results}
        },
    }


def _persistence_benchmark() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sirfex-benchmark-") as directory:
        store = ExperimentStore(Path(directory) / "experiments")
        started = time.perf_counter()
        manifest = run_and_save_experiment(
            store, Scenario("pump", seed=0), "pump-rule", max_steps=25
        )
        persist_elapsed = time.perf_counter() - started
        experiment_path = store.root / manifest.experiment_id
        artifact_size = sum(
            path.stat().st_size for path in experiment_path.iterdir() if path.is_file()
        )
        replay_started = time.perf_counter()
        replay = replay_experiment(store, manifest.experiment_id)
        replay_elapsed = time.perf_counter() - replay_started
        return {
            "persist_elapsed_seconds": persist_elapsed,
            "replay_elapsed_seconds": replay_elapsed,
            "artifact_size_bytes": artifact_size,
            "replay_status": replay.status,
        }


def run_benchmark(profile: str = "quick", output_dir: Path | None = None) -> dict[str, Any]:
    profiles = {"quick": 5, "standard": 25, "stress": 100}
    if profile not in profiles:
        raise ValueError(f"Unknown benchmark profile: {profile}")
    episodes = profiles[profile]
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
        "persistence": _persistence_benchmark(),
        "ppo_smoke_test": "not run by default; use scripts/ppo_smoke_test.py",
    }
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        markdown = [
            f"# SIRFEX {profile.title()} Benchmark",
            "",
            f"Episodes per environment: {episodes}",
            "",
            "| Environment | Episodes/s | Steps/s | RSS delta (bytes) |",
            "|---|---:|---:|---:|",
        ]
        for name, values in report["environments"].items():
            markdown.append(
                f"| {name} | {values['episodes_per_second']:.2f} | "
                f"{values['steps_per_second']:.2f} | {values['memory_delta_bytes']} |"
            )
        markdown.extend(["", f"Replay status: `{report['persistence']['replay_status']}`"])
        (output_dir / "benchmark.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report
