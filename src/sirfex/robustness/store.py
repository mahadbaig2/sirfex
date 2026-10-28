"""Atomic, resumable controller-by-anchor robustness evidence."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

from sirfex.core import Scenario
from sirfex.robustness.identity import scenario_identity, semantic_hash

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{5,80}$")


class RobustnessStorageError(RuntimeError):
    """Stored protocol or case evidence is missing, unsafe, or incompatible."""


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def controller_identity(spec: dict) -> str:
    """Checkpoint paths are local locators; checkpoint hashes identify policies."""
    safe = json.loads(json.dumps(spec, allow_nan=False))
    checkpoint = safe.get("config", {}).get("checkpoint")
    if isinstance(checkpoint, dict):
        checkpoint.pop("path", None)
    return semantic_hash(safe)


class RobustnessStore:
    def __init__(self, root: Path | str = "artifacts/robustness") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, run_id: str) -> Path:
        if not _SAFE_ID.fullmatch(run_id):
            raise RobustnessStorageError("Invalid robustness run ID")
        path = (self.root / run_id).resolve()
        if path.parent != self.root:
            raise RobustnessStorageError("Unsafe robustness path")
        return path

    def open_run(self, protocol: dict, *, run_id: str | None = None) -> str:
        if protocol.get("schema_version") != "pump-robustness-protocol:1":
            raise RobustnessStorageError("Unsupported robustness protocol schema")
        digest = semantic_hash(protocol)
        chosen = run_id or f"robustness_{digest[:20]}"
        directory = self.directory(chosen)
        path = directory / "protocol.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if semantic_hash(saved) != digest:
                raise RobustnessStorageError("Run ID has an incompatible protocol/configuration")
        else:
            directory.mkdir(parents=True, exist_ok=True)
            _atomic_json(path, protocol)
        (directory / "cases").mkdir(exist_ok=True)
        return chosen

    def case_id(self, run_id: str, controller: dict, anchor: Scenario) -> str:
        protocol = self._protocol(run_id)
        return semantic_hash(
            {
                "protocol_sha256": semantic_hash(protocol),
                "controller_identity": controller_identity(controller),
                "anchor_identity": scenario_identity(anchor),
            }
        )

    def _protocol(self, run_id: str) -> dict:
        try:
            return json.loads((self.directory(run_id) / "protocol.json").read_text("utf-8"))
        except (OSError, ValueError) as exc:
            raise RobustnessStorageError("Robustness protocol is missing or invalid") from exc

    def _case_path(self, run_id: str, case_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", case_id):
            raise RobustnessStorageError("Invalid case ID")
        return self.directory(run_id) / "cases" / f"{case_id}.json"

    def read_case(self, run_id: str, case_id: str) -> dict | None:
        path = self._case_path(run_id, case_id)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            stored_hash = value.pop("semantic_hash")
            if semantic_hash(value) != stored_hash or value.get("case_id") != case_id:
                raise RobustnessStorageError("Stored robustness case hash changed")
            value["semantic_hash"] = stored_hash
            return value
        except (OSError, ValueError, KeyError) as exc:
            raise RobustnessStorageError("Stored robustness case is corrupt") from exc

    def write_case(self, run_id: str, case_id: str, case: dict) -> dict:
        if case.get("case_id") != case_id:
            raise RobustnessStorageError("Case identity mismatch")
        record = dict(case)
        record.pop("semantic_hash", None)
        record["semantic_hash"] = semantic_hash(record)
        _atomic_json(self._case_path(run_id, case_id), record)
        return record


def run_items(
    store: RobustnessStore,
    run_id: str,
    items: Iterable[tuple[dict, Scenario]],
    evaluate: Callable[[dict, Scenario], dict],
) -> dict:
    """Skip verified completed cases, preserve failures, and keep semantic order stable."""
    ordered = sorted(items, key=lambda item: store.case_id(run_id, item[0], item[1]))
    case_ids = [store.case_id(run_id, controller, anchor) for controller, anchor in ordered]
    if len(case_ids) != len(set(case_ids)):
        raise RobustnessStorageError("Duplicate controller-anchor work item")
    reused = 0
    known = {case_id: store.read_case(run_id, case_id) for case_id in case_ids}
    counts = {
        "completed": 0,
        "pending": 0,
        "invalid": 0,
        "failed": 0,
        "censored": 0,
        "reused_completed": 0,
    }
    for item in known.values():
        status = item["status"] if item is not None else "pending"
        if status not in ("completed", "pending", "invalid", "failed"):
            raise RobustnessStorageError("Unknown robustness case status")
        counts[status] += 1
        if status == "completed":
            counts["censored"] += bool(item.get("censored"))

    def snapshot() -> dict:
        result = dict(counts)
        result["reused_completed"] = reused
        result["semantic_results_sha256"] = (
            semantic_hash([known[case_id]["semantic_hash"] for case_id in case_ids])
            if result["completed"] == len(case_ids)
            else None
        )
        _atomic_json(
            store.directory(run_id) / "status.json",
            {
                "schema_version": "pump-robustness-status:1",
                "run_id": run_id,
                "planned_case_ids": case_ids,
                "counts": result,
            },
        )
        return result

    def replace_case(case_id: str, saved: dict) -> None:
        previous = known[case_id]
        previous_status = previous["status"] if previous is not None else "pending"
        counts[previous_status] -= 1
        if previous_status == "completed":
            counts["censored"] -= bool(previous.get("censored"))
        known[case_id] = saved
        counts[saved["status"]] += 1
        if saved["status"] == "completed":
            counts["censored"] += bool(saved.get("censored"))
        snapshot()

    snapshot()
    for controller, anchor in ordered:
        case_id = store.case_id(run_id, controller, anchor)
        existing = known[case_id]
        if existing is not None and existing["status"] == "completed":
            reused += 1
            continue
        try:
            result = evaluate(controller, anchor)
            case = {
                "schema_version": "pump-robustness-case:1",
                "case_id": case_id,
                "controller_identity": controller_identity(controller),
                "anchor_id": scenario_identity(anchor),
                "status": "completed",
                "censored": bool(result.get("censored", False)),
                "result": result,
            }
            saved = store.write_case(run_id, case_id, case)
            replace_case(case_id, saved)
        except Exception as exc:
            evidence = getattr(exc, "evidence", None)
            saved = store.write_case(
                run_id,
                case_id,
                {
                    "schema_version": "pump-robustness-case:1",
                    "case_id": case_id,
                    "controller_identity": controller_identity(controller),
                    "anchor_id": scenario_identity(anchor),
                    "status": "invalid" if evidence is not None else "failed",
                    "error_type": type(exc).__name__,
                    "diagnostic_evidence": evidence,
                },
            )
            replace_case(case_id, saved)
            raise
    return snapshot()
