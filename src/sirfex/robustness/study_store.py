"""Write-once, integrity-checked work ledger for a Phase 3.7 study plan."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from sirfex.robustness.identity import semantic_hash
from sirfex.robustness.store import RobustnessStorageError, _atomic_json, _exclusive_lock


class StudyBudgetExhausted(RuntimeError):
    """Execution stopped at a durable work boundary and can resume if budget allows."""


class StudyStore:
    def __init__(self, root: Path | str, plan: dict) -> None:
        self.root = Path(root).resolve()
        self.plan = plan
        expected = semantic_hash({k: v for k, v in plan.items() if k != "semantic_sha256"})
        if plan.get("semantic_sha256") != expected:
            raise RobustnessStorageError("Study plan identity mismatch")
        self.run_id = (
            "synthetic_" if plan["profile"] == "synthetic_test_only" else "study_"
        ) + expected[:20]
        self.directory = self.root / self.run_id
        self.items_dir = self.directory / "items"
        self._known = {item["work_id"]: item for item in plan["work_items"]}
        if len(self._known) != len(plan["work_items"]):
            raise RobustnessStorageError("Duplicate study work item")

    def open(self) -> None:
        self.items_dir.mkdir(parents=True, exist_ok=True)
        with _exclusive_lock(self.directory / ".plan.lock"):
            path = self.directory / "plan.json"
            if path.exists():
                try:
                    saved = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise RobustnessStorageError("Stored study plan is corrupt") from exc
                if saved != self.plan:
                    raise RobustnessStorageError("Stored study plan is incompatible")
            else:
                _atomic_json(path, self.plan)
            if not (self.directory / "budget.json").exists():
                budget = {
                    "schema_version": "pump-study-budget:1",
                    "plan_sha256": self.plan["semantic_sha256"],
                    "cumulative_wall_seconds": 0.0,
                    "controller_seconds": {},
                }
                budget["semantic_sha256"] = semantic_hash(budget)
                _atomic_json(
                    self.directory / "budget.json",
                    budget,
                )

    def _path(self, work_id: str) -> Path:
        if work_id not in self._known:
            raise RobustnessStorageError("Work item is outside the frozen plan")
        return self.items_dir / f"{work_id}.json"

    def read(self, work_id: str) -> dict | None:
        path = self._path(work_id)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            digest = value.pop("semantic_sha256")
            if digest != semantic_hash(value) or value.get("work_id") != work_id:
                raise RobustnessStorageError("Study work-item evidence hash changed")
            if value.get("stage") != self._known[work_id]["stage"]:
                raise RobustnessStorageError("Study work-item stage changed")
            if value.get("plan_sha256") != self.plan["semantic_sha256"]:
                raise RobustnessStorageError("Study work-item plan changed")
            value["semantic_sha256"] = digest
            return value
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RobustnessStorageError("Study work-item evidence is corrupt") from exc

    def _storage_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.directory.rglob("*") if path.is_file())

    def _check_storage(self, value: dict, *, replacing: Path | None = None) -> None:
        cap = self.plan["resource_limits"].get("artifact_bytes_cap", 0)
        serialized = len(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8"))
        old = replacing.stat().st_size if replacing and replacing.exists() else 0
        # Atomic replacement temporarily needs the old and new files simultaneously.
        if cap <= 0 or self._storage_bytes() + 2 * serialized > cap or old > cap:
            raise StudyBudgetExhausted("study_storage_budget_exhausted")

    def write(
        self, item: dict, status: str, result: dict | None = None, *, resume: bool = False
    ) -> dict:
        if item.get("work_id") not in self._known or self._known[item["work_id"]] != item:
            raise RobustnessStorageError("Work-item identity differs from plan")
        if status not in {"running", "completed", "failed", "invalid", "censored"}:
            raise RobustnessStorageError("Invalid study work-item status")
        work_id = item["work_id"]
        path = self._path(work_id)
        with _exclusive_lock(path.with_suffix(".lock")):
            previous = self.read(work_id)
            if previous is not None:
                if previous["status"] in {"completed", "censored"}:
                    raise RobustnessStorageError("Completed study evidence is immutable")
                if previous["status"] in {"failed", "invalid"} and not resume:
                    raise RobustnessStorageError("Failed study item requires explicit resume")
            value = {
                "schema_version": "pump-study-work-item:1",
                "plan_sha256": self.plan["semantic_sha256"],
                "work_id": work_id,
                "stage": item["stage"],
                "status": status,
                "result": result,
            }
            value["semantic_sha256"] = semantic_hash(value)
            self._check_storage(value, replacing=path)
            _atomic_json(path, value)
            return value

    def budget(self) -> dict:
        path = self.directory / "budget.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RobustnessStorageError("Study budget ledger is missing or corrupt") from exc
        if value.get("plan_sha256") != self.plan["semantic_sha256"]:
            raise RobustnessStorageError("Study budget ledger plan mismatch")
        digest = value.pop("semantic_sha256", None)
        if digest != semantic_hash(value):
            raise RobustnessStorageError("Study budget ledger hash changed")
        value["semantic_sha256"] = digest
        return value

    def charge(self, seconds: float, controller_seconds: dict[str, float]) -> None:
        if seconds < 0 or any(value < 0 for value in controller_seconds.values()):
            raise RobustnessStorageError("Negative study resource usage")
        path = self.directory / "budget.json"
        with _exclusive_lock(self.directory / ".budget.lock"):
            value = self.budget()
            value.pop("semantic_sha256")
            value["cumulative_wall_seconds"] += seconds
            for key, amount in controller_seconds.items():
                value["controller_seconds"][key] = value["controller_seconds"].get(key, 0) + amount
            value["semantic_sha256"] = semantic_hash(value)
            self._check_storage(value, replacing=path)
            _atomic_json(path, value)

    def check_budget(self, controllers: tuple[str, ...] = (), *, reserve_bytes: int = 0) -> None:
        limits = self.plan["resource_limits"]
        ledger = self.budget()
        if ledger["cumulative_wall_seconds"] >= limits["wall_seconds_cap"]:
            raise StudyBudgetExhausted("study_wall_budget_exhausted")
        for controller in controllers:
            if (
                controller.startswith("ppo_")
                or controller.startswith("screen_")
                or controller.startswith("seed_")
            ):
                if ledger["controller_seconds"].get(controller, 0) >= limits["per_ppo_seconds_cap"]:
                    raise StudyBudgetExhausted(f"controller_budget_exhausted:{controller}")
        if self._storage_bytes() + reserve_bytes >= limits["artifact_bytes_cap"]:
            raise StudyBudgetExhausted("study_storage_budget_exhausted")

    def status(self) -> dict:
        counts = {stage: Counter() for stage in self.plan["work_counts"]}
        for item in self.plan["work_items"]:
            record = self.read(item["work_id"])
            counts[item["stage"]][record["status"] if record else "pending"] += 1
        stages = {
            stage: {
                key: counter.get(key, 0)
                for key in ("pending", "running", "completed", "failed", "invalid", "censored")
            }
            for stage, counter in counts.items()
        }
        complete = all(
            sum(row[key] for key in ("completed", "censored")) == self.plan["work_counts"][stage]
            for stage, row in stages.items()
        )
        return {
            "schema_version": "pump-study-status:1",
            "plan_sha256": self.plan["semantic_sha256"],
            "status": "complete" if complete else "incomplete",
            "stages": stages,
            "resource_usage": self.budget(),
            "artifact_bytes": self._storage_bytes(),
        }


class InvocationBudget:
    """Monotonic per-invocation time with durable cumulative charges at item boundaries."""

    def __init__(self, store: StudyStore, clock=time.monotonic) -> None:
        self.store = store
        self.clock = clock
        self.started = clock()
        self.charged_until = self.started

    def check(self, controllers: tuple[str, ...] = (), reserve_bytes: int = 0) -> None:
        now = self.clock()
        pending = now - self.charged_until
        if (
            self.store.budget()["cumulative_wall_seconds"] + pending
            >= self.store.plan["resource_limits"]["wall_seconds_cap"]
        ):
            # Persist elapsed time before rejecting work so a restart cannot reset it.
            self.store.charge(pending, {})
            self.charged_until = now
            raise StudyBudgetExhausted("study_wall_budget_exhausted")
        self.store.check_budget(controllers, reserve_bytes=reserve_bytes)

    def charge(self, controller_seconds: dict[str, float]) -> None:
        now = self.clock()
        self.store.charge(now - self.charged_until, controller_seconds)
        self.charged_until = now
