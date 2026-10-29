"""Work-item execution for a gated Phase 3.7 pump study."""

from __future__ import annotations

import json
import time
from pathlib import Path

from sirfex.core import Scenario
from sirfex.robustness.disturbance import evaluate_direction, prepare_grid
from sirfex.robustness.identity import scenario_identity, semantic_hash
from sirfex.robustness.margins import evaluate_margin
from sirfex.robustness.service import _Session
from sirfex.robustness.store import RobustnessStorageError, _atomic_json
from sirfex.robustness.study_store import InvocationBudget, StudyBudgetExhausted, StudyStore
from sirfex.training.checkpoints import verify_checkpoint


def _specs(protocol: dict, root: Path) -> dict[str, dict]:
    result = {}
    for row in protocol["controller_inventory"]:
        key = row.get("source_key") or row.get("controller_id") or row["id"]
        if "source_artifact" in row:
            path = (root / row["source_artifact"]).resolve()
            verify_checkpoint(path, row["sha256"])
            metadata_path = path.with_suffix(".metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata["checkpoint_id"] != key
                or metadata["sha256"] != row["sha256"]
                or metadata["training_run_id"] != row["source_run"]
            ):
                raise RobustnessStorageError("Checkpoint metadata identity mismatch")
            metadata["path"] = str(path)
            result[key] = {"id": "ppo", "config": {"checkpoint": metadata}}
        else:
            result[key] = {
                "id": row.get("controller_id", row.get("id")),
                "config": row.get("config", {}),
            }
    return result


class ControllerPool:
    def __init__(self, specs: dict[str, dict], *, max_open: int = 2):
        self.specs = specs
        self.max_open = max_open
        self.sessions: dict[str, _Session] = {}

    def get(self, key: str) -> _Session:
        if key not in self.specs:
            raise RobustnessStorageError("Controller is outside the study inventory")
        if key not in self.sessions:
            if len(self.sessions) >= self.max_open:
                oldest = next(iter(self.sessions))
                self.sessions.pop(oldest).close()
            self.sessions[key] = _Session(self.specs[key])
        return self.sessions[key]

    def evaluate(self, key: str, scenario: Scenario):
        return self.get(key).evaluate(scenario)

    def seconds(self, key: str) -> float:
        return self.sessions[key].seconds if key in self.sessions else 0.0

    def close(self) -> None:
        for session in self.sessions.values():
            session.close()
        self.sessions.clear()


def compact_episode(result) -> dict:
    margin = evaluate_margin(result)
    if margin.evidence_status != "valid":
        raise ValueError("Incomplete episode cannot enter a scientific comparison")
    scenario = Scenario.from_dict(result.scenario)
    target = float(scenario.parameters.get("target_level", 0.5))
    levels = [float(row["info"]["level"]) for row in result.trace]
    if len(levels) != result.episode_length or not levels:
        raise ValueError("Tracking evidence is missing")
    return {
        "scenario_id": scenario_identity(scenario),
        "controller_id": result.controller_id,
        "episode_sha256": semantic_hash(result.to_dict()),
        "tracking_mae": sum(abs(level - target) for level in levels) / len(levels),
        "margin": margin.to_dict(),
        "episode_length": result.episode_length,
    }


class StudyEngine:
    def __init__(
        self,
        protocol: dict,
        manifest: dict,
        plan: dict,
        root: Path | str,
        output: Path | str,
        *,
        synthetic: bool = False,
        clock=time.monotonic,
    ):
        if synthetic != (plan["profile"] == "synthetic_test_only"):
            raise ValueError("Synthetic execution profile mismatch")
        if (
            plan["protocol_sha256"] != protocol["semantic_sha256"]
            or plan["scenario_manifest_sha256"] != manifest["semantic_sha256"]
        ):
            raise RobustnessStorageError("Execution plan input identity changed")
        self.protocol = protocol
        self.manifest = manifest
        self.plan = plan
        self.root = Path(root).resolve()
        self.store = StudyStore(output, plan)
        self.store.open()
        self.budget = InvocationBudget(self.store, clock=clock)
        self.specs = _specs(protocol, self.root)
        self.pool = ControllerPool(self.specs)
        self.scenarios = {
            row["scenario_id"]: Scenario.from_dict(row["scenario"]) for row in manifest["scenarios"]
        }
        self.prepared = {}

    def close(self) -> None:
        self.pool.close()

    def prepare(self) -> None:
        """Recompute shared physics and bind its ordered hashes before evaluation."""
        started = self.budget.clock()
        grid = self.plan["grid"]
        for anchor_id in self.plan["nominal_scenarios"]:
            anchor = self.scenarios[anchor_id]
            for direction in self.plan["directions"]:
                self.budget.check()
                key = (anchor_id, tuple(direction))
                self.prepared[key] = prepare_grid(
                    anchor,
                    tuple(direction),
                    grid,
                    protocol_sha256=self.plan["protocol_sha256"],
                )
        record = {
            "schema_version": "pump-study-prepared-grids:1",
            "plan_sha256": self.plan["semantic_sha256"],
            "grids": [
                {
                    "anchor_id": anchor_id,
                    "direction": list(direction),
                    "semantic_sha256": grid.semantic_sha256,
                }
                for (anchor_id, direction), grid in self.prepared.items()
            ],
        }
        record["semantic_sha256"] = semantic_hash(record)
        path = self.store.directory / "prepared_grids.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved != record:
                raise RobustnessStorageError("Prepared physical grids changed on resume")
        else:
            self.store._check_storage(record)
            _atomic_json(path, record)
        self.store.charge(self.budget.clock() - started, {})
        self.budget.charged_until = self.budget.clock()

    def _run_item(self, item: dict, evaluate, *, controllers: tuple[str, ...], resume: bool):
        previous = self.store.read(item["work_id"])
        if previous and previous["status"] in {"completed", "censored"}:
            return previous
        if previous and not resume:
            raise RobustnessStorageError("Incomplete work item requires explicit resume")
        self.budget.check(controllers)
        self.store.write(item, "running", resume=resume)
        before = {key: self.pool.seconds(key) for key in controllers}
        try:
            result = evaluate()
            status = "censored" if result.get("censored", False) else "completed"
            saved = self.store.write(item, status, result, resume=resume)
        except Exception as exc:
            try:
                self.store.write(
                    item,
                    "invalid" if hasattr(exc, "evidence") else "failed",
                    {"error_type": type(exc).__name__, "message": str(exc)},
                    resume=resume,
                )
            finally:
                self.budget.charge(
                    {key: max(0.0, self.pool.seconds(key) - before[key]) for key in controllers}
                )
            raise
        self.budget.charge(
            {key: max(0.0, self.pool.seconds(key) - before[key]) for key in controllers}
        )
        return saved

    def execute_fixed_and_grid(self, *, resume: bool = False, interrupt_after: int | None = None):
        self.prepare()
        completed_this_invocation = 0
        try:
            for item in self.plan["work_items"]:
                if item["stage"] not in {"fixed_stress", "grid"}:
                    continue
                if interrupt_after is not None and completed_this_invocation >= interrupt_after:
                    raise StudyBudgetExhausted("synthetic_interruption")
                controller = item["controller"]
                if item["stage"] == "fixed_stress":
                    scenario = self.scenarios[item["scenario_id"]]

                    def evaluate_fixed(controller=controller, scenario=scenario):
                        return compact_episode(self.pool.evaluate(controller, scenario))

                    self._run_item(item, evaluate_fixed, controllers=(controller,), resume=resume)
                else:
                    self._run_item(
                        item,
                        lambda item=item: self._evaluate_grid_item(item),
                        controllers=(controller,),
                        resume=resume,
                    )
                completed_this_invocation += 1
        finally:
            self.pool.close()
        return self.store.status()

    def _evaluate_grid_item(self, item: dict) -> dict:
        controller = item["controller"]
        anchor_id = item["anchor_id"]
        anchor = self.scenarios[anchor_id]
        direction = tuple(item["direction"])
        prepared = self.prepared[(anchor_id, direction)]
        nominal = None

        def evaluator(scenario):
            nonlocal nominal
            result = self.pool.evaluate(controller, scenario)
            if (
                direction == tuple(self.plan["directions"][0])
                and scenario_identity(scenario) == anchor_id
            ):
                nominal = compact_episode(result)
            return result

        cell = evaluate_direction(
            anchor,
            direction,
            self.plan["grid"],
            self.specs[controller]["id"],
            evaluator,
            prepared=prepared,
            refine_feasibility=False,
            protocol_sha256=self.plan["protocol_sha256"],
        ).to_dict()
        return {
            "cell": cell,
            "nominal": nominal,
            "prepared_grid_sha256": prepared.semantic_sha256,
            "censored": cell["status"].startswith("censored") or cell["status"] == "undefined_axis",
        }
