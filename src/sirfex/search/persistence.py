"""Atomic local persistence for search runs."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_SEARCH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{5,80}$")
_DISCOVERY_ID = re.compile(r"^disc_[A-Za-z0-9_-]{5,80}$")


class SearchStorageError(RuntimeError):
    pass


class SearchStore:
    def __init__(self, root: Path | str = Path("artifacts/searches")) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _directory(self, search_id: str) -> Path:
        if not _SEARCH_ID.fullmatch(search_id):
            raise SearchStorageError("Invalid search ID")
        directory = (self.root / search_id).resolve()
        if directory.parent != self.root:
            raise SearchStorageError("Unsafe search path")
        return directory

    @staticmethod
    def _atomic_write(path: Path, value: str) -> None:
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

    def save(
        self,
        search_id: str,
        manifest: dict[str, Any],
        search_space: dict[str, Any],
        trials: list[dict[str, Any]],
        discoveries: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> Path:
        directory = self._directory(search_id)
        if directory.exists():
            raise SearchStorageError(f"Search already exists: {search_id}")
        directory.mkdir(parents=True)
        marker = directory / ".incomplete"
        marker.write_text("incomplete", encoding="utf-8")
        try:
            self._atomic_write(
                directory / "search_manifest.json", json.dumps(manifest, indent=2, allow_nan=False)
            )
            self._atomic_write(
                directory / "search_space.json", json.dumps(search_space, indent=2, allow_nan=False)
            )
            self._atomic_write(
                directory / "trials.jsonl",
                "".join(
                    json.dumps(trial, allow_nan=False, separators=(",", ":")) + "\n"
                    for trial in trials
                ),
            )
            self._atomic_write(
                directory / "discoveries.json", json.dumps(discoveries, indent=2, allow_nan=False)
            )
            self._atomic_write(
                directory / "summary.json", json.dumps(summary, indent=2, allow_nan=False)
            )
            marker.unlink()
        except Exception:
            raise
        return directory

    def load(self, search_id: str) -> dict[str, Any]:
        directory = self._directory(search_id)
        if not directory.is_dir():
            raise SearchStorageError(f"Search not found: {search_id}")
        if (directory / ".incomplete").exists():
            raise SearchStorageError(f"Search is incomplete: {search_id}")
        try:
            manifest = json.loads((directory / "search_manifest.json").read_text(encoding="utf-8"))
            space = json.loads((directory / "search_space.json").read_text(encoding="utf-8"))
            trials = [
                json.loads(line)
                for line in (directory / "trials.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            discoveries = json.loads((directory / "discoveries.json").read_text(encoding="utf-8"))
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SearchStorageError(f"Malformed search artifacts: {search_id}") from exc
        if manifest.get("search_id") != search_id:
            raise SearchStorageError("Search manifest ID does not match its directory")
        result = {
            "manifest": manifest,
            "space": space,
            "trials": trials,
            "discoveries": discoveries,
            "summary": summary,
        }
        if (directory / "avoidable.json").is_file():
            try:
                result["avoidable"] = json.loads(
                    (directory / "avoidable.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise SearchStorageError("Malformed avoidable-failure evidence") from exc
        return result

    def list(self) -> list[dict[str, Any]]:
        results = []
        for directory in sorted(self.root.iterdir()):
            if directory.is_dir() and _SEARCH_ID.fullmatch(directory.name):
                try:
                    results.append(self.load(directory.name)["manifest"])
                except SearchStorageError:
                    continue
        return results

    def find_discovery(self, discovery_id: str) -> tuple[str, dict[str, Any]]:
        for manifest in self.list():
            loaded = self.load(manifest["search_id"])
            for discovery in loaded["discoveries"]:
                if discovery.get("discovery_id") == discovery_id:
                    return manifest["search_id"], discovery
        raise SearchStorageError(f"Discovery not found: {discovery_id}")

    def save_verification(
        self, search_id: str, discovery_id: str, verification: dict[str, Any]
    ) -> Path:
        directory = self._directory(search_id)
        if not directory.is_dir() or (directory / ".incomplete").exists():
            raise SearchStorageError(f"Search not available: {search_id}")
        if not _DISCOVERY_ID.fullmatch(discovery_id):
            raise SearchStorageError("Invalid discovery ID")
        verification_dir = directory / "verification"
        verification_dir.mkdir(exist_ok=True)
        path = verification_dir / f"{discovery_id}.json"
        self._atomic_write(path, json.dumps(verification, indent=2, allow_nan=False))
        return path

    def load_verification(self, search_id: str, discovery_id: str) -> dict[str, Any] | None:
        directory = self._directory(search_id)
        if not directory.is_dir() or (directory / ".incomplete").exists():
            raise SearchStorageError(f"Search not available: {search_id}")
        if not _DISCOVERY_ID.fullmatch(discovery_id):
            raise SearchStorageError("Invalid discovery ID")
        path = directory / "verification" / f"{discovery_id}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SearchStorageError(f"Malformed verification artifact: {discovery_id}") from exc
