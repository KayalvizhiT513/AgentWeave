"""Append-only record of every eval run. Rows are never edited; a rerun adds a row with a higher attempt."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parents[3] / "evals"
SCHEMA_VERSION = 1


def git_state(cwd: Path = EVALS_DIR.parent) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    sha = run("rev-parse", "--short", "HEAD")
    status = run("status", "--porcelain")
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


class Ledger:
    def __init__(self, directory: Path = EVALS_DIR) -> None:
        self.directory = directory
        self.path = directory / "ledger.jsonl"
        self.transcripts = directory / "transcripts"

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def latest_by_run(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self.rows():
            current = latest.get(row["run_id"])
            if current is None or row["attempt"] >= current["attempt"]:
                latest[row["run_id"]] = row
        return latest

    def next_attempt(self, run_id: str) -> int:
        return max((row["attempt"] for row in self.rows() if row["run_id"] == run_id), default=0) + 1

    def append(self, row: dict[str, Any], transcript: dict[str, Any] | None = None) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        row = {"schema": SCHEMA_VERSION, **row}
        if transcript is not None:
            self.transcripts.mkdir(parents=True, exist_ok=True)
            relative = f"transcripts/{row['run_id']}.a{row['attempt']}.json"
            (self.directory / relative).write_text(json.dumps(transcript, indent=2, default=str))
            row["transcript"] = relative
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
