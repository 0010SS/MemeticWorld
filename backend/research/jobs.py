"""Durable background jobs for the local research interface. No shell commands."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from backend.config import REPO_ROOT
from backend.experiment.process import alive
from backend.research.common import read_json, write_json


def launch(root, arguments, kind):
    root = Path(root).resolve()
    job_id = "job_" + uuid.uuid4().hex[:16]
    directory = root / ".research_jobs" / job_id
    spec = {"id": job_id, "kind": kind, "arguments": list(arguments), "status": "queued", "created": time.time()}
    write_json(directory / "job.json", spec)
    with (directory / "log.txt").open("a", encoding="utf-8") as log:
        subprocess.Popen([sys.executable, "-B", "-X", "utf8", "-m", "backend.research.jobs",
                          "--root", str(root), "--job", job_id], cwd=REPO_ROOT,
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "MEMEWORLD_RUNS_ROOT": str(root)},
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return spec


def status(directory):
    directory = Path(directory)
    result = read_json(directory / "job.json", {})
    if result.get("status") == "running" and not alive(result.get("pid")):
        result = {**result, "status": "interrupted", "error": "The job worker exited without recording completion. Check the log and rerun to recover completed work."}
    log = directory / "log.txt"
    if log.exists():
        with log.open("rb") as fh:
            fh.seek(max(0, log.stat().st_size - 16000))
            result["log"] = fh.read().decode("utf-8", errors="replace")
    return result


def worker(root, job_id):
    directory = Path(root) / ".research_jobs" / job_id
    spec = read_json(directory / "job.json")
    if not spec or spec["status"] != "queued":
        raise ValueError("No queued job")
    spec.update(status="running", pid=os.getpid(), started=time.time())
    write_json(directory / "job.json", spec)
    try:
        proc = subprocess.Popen([sys.executable, "-B", "-X", "utf8", "-m", "backend.cli", *spec["arguments"]],
                                cwd=REPO_ROOT, stdin=subprocess.DEVNULL,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        spec["child_pid"] = proc.pid
        write_json(directory / "job.json", spec)
        code = proc.wait()
        spec.update(status="complete" if code == 0 else "failed", returncode=code)
    except BaseException as exc:
        spec.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        spec["ended"] = time.time()
        write_json(directory / "job.json", spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    worker(args.root, args.job)
