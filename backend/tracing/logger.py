"""Structured JSONL trace of everything that happens in a run.

Records produced inside parallel jobs are buffered per job scope and flushed in
sorted scope order at the end of each simulation phase, so the trace file is
byte-identical across replays regardless of thread scheduling.

Record ids are deterministic: "<scope>#<n>".

v3 (docs/ONTOLOGY_V3.md §6.2): per-tick digests. Every trace line written while the logger's `tick` is t
feeds sha256 digest t; when `tick` moves on (the engine assigns `tracer.tick` at the start of each tick)
or the logger closes, the digest is appended to `digests.jsonl` as {"tick", "sha256", "n"}. A branch's
ticks < T must have the parent's digests (`compare_digests`, `prefix_digest`).

Record types and their fields are registered in `backend/tracing/schema.py` (docs/TRACE_SCHEMA.md). With
`check=True` or the environment variable MEMEWORLD_TRACE_CHECK=1, `log` validates every record against the
registry and warns (never fails) once per unregistered type or missing required field.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import warnings
from pathlib import Path

from backend.llm.client import current_scope

_EMPTY = hashlib.sha256().hexdigest()


class TraceLogger:
    def __init__(self, run_dir: Path, digests: bool = True, check: bool | None = None):
        self.run_dir = Path(run_dir)
        self.fh = open(self.run_dir / "trace.jsonl", "w")
        self._lock = threading.Lock()
        self._buffers: dict[str, list[dict]] = {}
        self._counters: dict[str, int] = {}
        self._tick = 0
        self.time = ""
        self._dfh = open(self.run_dir / "digests.jsonl", "w") if digests else None
        self._dh = hashlib.sha256()
        self._dn = 0
        self.digests: dict[int, str] = {}
        self.late_writes = 0
        self.check = os.environ.get("MEMEWORLD_TRACE_CHECK", "") not in ("", "0") if check is None else bool(check)
        self._warned: set[str] = set()

    # the engine sets `tracer.tick = t` at the start of every tick; moving on closes the previous digest
    @property
    def tick(self) -> int:
        return self._tick

    @tick.setter
    def tick(self, value: int):
        value = int(value)
        with self._lock:
            if value != self._tick:
                self._end_tick_locked()
                self._tick = value

    def _end_tick_locked(self):
        if self._dfh is None or self._tick in self.digests:
            return
        sha = self._dh.hexdigest()
        self.digests[self._tick] = sha
        self._dfh.write(json.dumps({"tick": self._tick, "sha256": sha, "n": self._dn}) + "\n")
        self._dfh.flush()
        self._dh, self._dn = hashlib.sha256(), 0

    def end_tick(self):
        """Explicitly close the current tick's digest (flushes pending records first)."""
        self.flush()
        with self._lock:
            self._end_tick_locked()

    def log(self, type_: str, **fields) -> str:
        scope = current_scope()
        with self._lock:
            n = self._counters.get(scope, 0)
            self._counters[scope] = n + 1
            rid = f"{scope}#{n}"
            rec = {"id": rid, "type": type_, "tick": self._tick, "time": self.time, **fields}
            self._buffers.setdefault(scope, []).append(rec)
            if self.check:
                self._check_locked(rec)
        return rid

    def _check_locked(self, rec: dict):
        """Debug mode: warn once per problem (unregistered type, missing required field); never raise."""
        from backend.tracing.schema import validate
        for problem in validate(rec):
            if problem not in self._warned:
                self._warned.add(problem)
                warnings.warn(f"trace record {rec.get('id')}: {problem}", RuntimeWarning, stacklevel=3)

    def flush(self):
        with self._lock:
            for scope in sorted(self._buffers):
                for rec in self._buffers[scope]:
                    line = json.dumps(rec, default=str) + "\n"
                    self.fh.write(line)
                    if self._dfh is None:
                        continue
                    if self._tick in self.digests:   # tick moved backwards: never rewrite a closed digest
                        self.late_writes += 1            # (trace_sha256 still covers the line)
                        continue
                    self._dh.update(line.encode())
                    self._dn += 1
            self._buffers.clear()
            self.fh.flush()

    def close(self, end_tick: bool = True):
        """Flush and close. `end_tick=False` (a paused/crashed run) leaves the unfinished tick without a
        digest, so digests.jsonl only covers complete ticks."""
        self.flush()
        with self._lock:
            if end_tick:
                self._end_tick_locked()
            if self._dfh is not None:
                self._dfh.close()
        self.fh.close()


def read_trace(run_dir: Path, types: set[str] | None = None):
    with open(Path(run_dir) / "trace.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if types is None or rec["type"] in types:
                yield rec


def read_digests(run_dir: Path) -> dict[int, str]:
    """tick -> sha256 of that tick's trace lines (digests.jsonl; {} if the run predates v3)."""
    p = Path(run_dir) / "digests.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            out[int(r["tick"])] = r["sha256"]
    return out


def prefix_digest(run_dir: Path, until_tick: int) -> str:
    """One sha256 over the per-tick digests of ticks < until_tick (manifest branch.prefix_digest).
    Ticks without a digest (no records) count as the empty digest; a missing digests file raises."""
    d = read_digests(run_dir)
    if not d and until_tick > 0:
        raise FileNotFoundError(f"no digests.jsonl in {run_dir}")
    h = hashlib.sha256()
    for t in range(int(until_tick)):
        h.update(f"{t}:{d.get(t, _EMPTY)}\n".encode())
    return h.hexdigest()


def compare_digests(parent_dir: Path, child_dir: Path, until_tick: int) -> list[int]:
    """Ticks < until_tick whose trace digests differ between parent and child (empty = identical prefix).
    A tick present in one run and missing from the other counts as a mismatch."""
    a, b = read_digests(parent_dir), read_digests(child_dir)
    return [t for t in range(int(until_tick)) if a.get(t) != b.get(t)]
