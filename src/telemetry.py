"""JSONL telemetry logger. One record per sampled (VLM-analyzed) frame."""
from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Optional


class JsonlLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: Optional[IO] = None

    def __enter__(self) -> "JsonlLogger":
        self._fh = self.path.open("w", encoding="utf-8")
        return self

    def write(self, record: dict) -> None:
        assert self._fh is not None, "use as context manager"
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def __exit__(self, *exc) -> None:
        if self._fh:
            self._fh.close()


def read_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
