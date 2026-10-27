"""Small atomic local store for bounded research studies."""

from __future__ import annotations

import json
import re
from pathlib import Path

from sirfex.training.checkpoints import write_json

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{5,80}$")


class ResearchStore:
    def __init__(self, root: Path | str = "artifacts/research") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, run_id: str) -> Path:
        if not _ID.fullmatch(run_id):
            raise ValueError("Invalid research run ID")
        result = (self.root / run_id).resolve()
        if result.parent != self.root:
            raise ValueError("Unsafe research path")
        return result

    def create(self, run_id: str, files: dict[str, dict]) -> Path:
        directory = self.directory(run_id)
        directory.mkdir(exist_ok=False)
        for name, value in files.items():
            self.write(run_id, name, value)
        return directory

    def write(self, run_id: str, name: str, value: dict) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*\.json", name):
            raise ValueError("Unsafe research artifact name")
        directory = self.directory(run_id)
        if not directory.is_dir():
            raise ValueError("Research run not found")
        write_json(directory / name, value)

    def read(self, run_id: str, name: str) -> dict:
        if not re.fullmatch(r"[a-z][a-z0-9_]*\.json", name):
            raise ValueError("Unsafe research artifact name")
        path = self.directory(run_id) / name
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        result = []
        for child in sorted(self.root.iterdir()):
            if child.is_dir() and _ID.fullmatch(child.name):
                try:
                    result.append(self.read(child.name, "manifest.json"))
                except (OSError, json.JSONDecodeError):
                    continue
        return result
