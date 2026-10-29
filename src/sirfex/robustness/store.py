"""Atomic, resumable controller-by-anchor robustness evidence."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path

from sirfex.core import Scenario
from sirfex.robustness.identity import scenario_identity, semantic_hash

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{5,80}$")
_HELD_LOCKS = threading.local()


class RobustnessStorageError(RuntimeError):
    """Stored protocol or case evidence is missing, unsafe, or incompatible."""


@contextmanager
def _exclusive_lock(path: Path):
    """Reject simultaneous writers. A crash leaves a visible lock for manual recovery."""
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RobustnessStorageError(f"Concurrent or interrupted writer lock: {path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        held = getattr(_HELD_LOCKS, "paths", set())
        _HELD_LOCKS.paths = held | {path}
        yield
    finally:
        _HELD_LOCKS.paths = getattr(_HELD_LOCKS, "paths", set()) - {path}
        path.unlink(missing_ok=True)


@contextmanager
def _run_writer(path: Path):
    """A run worker may write its own cases; other threads/processes are refused."""
    if path in getattr(_HELD_LOCKS, "paths", set()):
        yield
    else:
        with _exclusive_lock(path):
            yield


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
        directory.mkdir(parents=True, exist_ok=True)
        with _exclusive_lock(directory / ".protocol.lock"):
            if path.exists():
                try:
                    saved = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise RobustnessStorageError("Stored protocol is corrupt") from exc
                if semantic_hash(saved) != digest:
                    raise RobustnessStorageError(
                        "Run ID has an incompatible protocol/configuration"
                    )
            else:
                _atomic_json(path, protocol)
            identity_path = directory / "protocol.sha256"
            if identity_path.exists():
                if identity_path.read_text(encoding="ascii").strip() != digest:
                    raise RobustnessStorageError("Stored protocol identity changed")
            else:
                atomic_text(identity_path, digest + "\n")
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
            directory = self.directory(run_id)
            protocol = json.loads((directory / "protocol.json").read_text("utf-8"))
            identity = directory / "protocol.sha256"
            if identity.exists() and identity.read_text(encoding="ascii").strip() != semantic_hash(
                protocol
            ):
                raise RobustnessStorageError("Stored protocol identity changed")
            return protocol
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

    def write_case(self, run_id: str, case_id: str, case: dict, *, recovery: bool = False) -> dict:
        """Write once; only an explicit recovery may replace failed/pending evidence."""
        if case.get("case_id") != case_id:
            raise RobustnessStorageError("Case identity mismatch")
        protocol = self._protocol(run_id)
        identity = case.get("controller_identity")
        anchor_id = case.get("anchor_id")
        if not isinstance(identity, str) or not isinstance(anchor_id, str):
            raise RobustnessStorageError("Case controller or anchor identity is missing")
        expected = semantic_hash(
            {
                "protocol_sha256": semantic_hash(protocol),
                "controller_identity": identity,
                "anchor_identity": anchor_id,
            }
        )
        if expected != case_id:
            raise RobustnessStorageError("Case protocol, controller, or anchor identity mismatch")
        if protocol.get("controllers") and identity not in {
            row.get("identity_sha256") for row in protocol["controllers"]
        }:
            raise RobustnessStorageError("Controller checkpoint identity is outside protocol")
        if protocol.get("anchors") and anchor_id not in protocol["anchors"]:
            raise RobustnessStorageError("Anchor identity is outside protocol")
        if case.get("status") not in {"completed", "failed", "invalid", "pending"}:
            raise RobustnessStorageError("Invalid case status")
        record = dict(case)
        record.pop("semantic_hash", None)
        record["semantic_hash"] = semantic_hash(record)
        path = self._case_path(run_id, case_id)
        with _run_writer(self.directory(run_id) / ".run.lock"):
            with _exclusive_lock(path.with_suffix(".lock")):
                existing = self.read_case(run_id, case_id)
                if existing is not None:
                    if existing["semantic_hash"] == record["semantic_hash"]:
                        return existing
                    if existing["status"] == "completed":
                        raise RobustnessStorageError("Completed case evidence is immutable")
                    if not recovery or existing["status"] not in {"failed", "invalid", "pending"}:
                        raise RobustnessStorageError(
                            "Failed/pending case requires explicit recovery"
                        )
                    if (existing["controller_identity"], existing["anchor_id"]) != (
                        identity,
                        anchor_id,
                    ):
                        raise RobustnessStorageError("Recovery identity mismatch")
                _atomic_json(path, record)
                return record


def run_items(
    store: RobustnessStore,
    run_id: str,
    items: Iterable[tuple[dict, Scenario]],
    evaluate: Callable[[dict, Scenario], dict],
) -> dict:
    """Skip verified completed cases, preserve failures, and keep semantic order stable."""
    with _exclusive_lock(store.directory(run_id) / ".run.lock"):
        return _run_items_locked(store, run_id, items, evaluate)


def _run_items_locked(
    store: RobustnessStore,
    run_id: str,
    items: Iterable[tuple[dict, Scenario]],
    evaluate: Callable[[dict, Scenario], dict],
) -> dict:
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
                recovery=existing is not None,
            )
            replace_case(case_id, saved)
            raise
        case = {
            "schema_version": "pump-robustness-case:1",
            "case_id": case_id,
            "controller_identity": controller_identity(controller),
            "anchor_id": scenario_identity(anchor),
            "status": "completed",
            "censored": bool(result.get("censored", False)),
            "result": result,
        }
        saved = store.write_case(run_id, case_id, case, recovery=existing is not None)
        replace_case(case_id, saved)
    return snapshot()
