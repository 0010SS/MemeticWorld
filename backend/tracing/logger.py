"""Structured JSONL trace of everything that happens in a run.

Records produced inside parallel jobs are buffered per job scope and flushed in
sorted scope order at the end of each simulation phase, so the trace file is
byte-identical across replays regardless of thread scheduling.

Record ids are deterministic: "<scope>#<n>".
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from backend.llm.client import current_scope


class TraceLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.fh = open(self.run_dir / "trace.jsonl", "w")
        self._lock = threading.Lock()
        self._buffers: dict[str, list[dict]] = {}
        self._counters: dict[str, int] = {}
        self.tick = 0
        self.time = ""

    def log(self, type_: str, **fields) -> str:
        scope = current_scope()
        with self._lock:
            n = self._counters.get(scope, 0)
            self._counters[scope] = n + 1
            rid = f"{scope}#{n}"
            rec = {"id": rid, "type": type_, "tick": self.tick, "time": self.time, **fields}
            self._buffers.setdefault(scope, []).append(rec)
        return rid

    def flush(self):
        with self._lock:
            for scope in sorted(self._buffers):
                for rec in self._buffers[scope]:
                    self.fh.write(json.dumps(rec, default=str) + "\n")
            self._buffers.clear()
            self.fh.flush()

    def close(self):
        self.flush()
        self.fh.close()


def read_trace(run_dir: Path, types: set[str] | None = None):
    with open(Path(run_dir) / "trace.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if types is None or rec["type"] in types:
                yield rec
