"""Small, versioned artifact primitives shared by the research workers."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from pathlib import Path


def digest(value) -> str:
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, p)


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, p)


def jsonl(path):
    p = Path(path)
    if not p.exists():
        return
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            # A live writer can have an unfinished last line. Never consume it.
            if not line.endswith("\n"):
                break
            if line.strip():
                yield json.loads(line)


def parse_object(text):
    """Accept JSON or a fenced JSON object; reject prose and empty failed responses."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("The judge must return a JSON object")
    return value


@contextlib.contextmanager
def lock(path):
    """Exclusive process lock. Recover only when the recorded local process is gone."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        prior = read_json(p, {})
        pid = prior.get("pid")
        from backend.experiment.process import alive
        if not pid or alive(pid):
            raise RuntimeError(f"Research worker already owns {p} (pid {pid})")
        p.unlink()
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"pid": os.getpid(), "started": time.time()}, fh)
        yield
    finally:
        p.unlink(missing_ok=True)
