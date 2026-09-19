"""Pause and verified replay continuation of an existing simulation."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from backend.tracing.logger import compare_digests


def request_pause(run_dir):
    run = Path(run_dir).resolve()
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if man.get("status") != "running":
        raise ValueError("Only a running simulation can be paused")
    (run / "pause.request").write_text("Requested through the experiment controller.\n", encoding="utf-8")
    return {"status": "pause_requested", "run_dir": str(run)}


from backend.tracing.replay import check_continuation


def continue_run(parent, out, days=None, *, progress=True):
    from backend.simulation.engine import Simulation, _code_version
    parent, out = Path(parent).resolve(), Path(out).resolve()
    if parent == out or parent in out.parents:
        raise ValueError("Continuation must use a separate run directory, outside its parent recording")
    man = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    if man.get("status") not in ("finished", "paused", "failed"):
        raise ValueError("Wait until the parent is finished or paused")
    if (parent / "digests.jsonl").exists():
        digests = [json.loads(l) for l in (parent / "digests.jsonl").read_text(encoding="utf-8").splitlines() if l]
    else:
        raise ValueError("Continuation needs the recorded per-tick digests")
    boundary = max((d["tick"] for d in digests), default=-1) + 1
    if boundary == 0:
        raise ValueError("The parent has no committed history to continue")
    old, current = man.get("code_version") or {}, _code_version()
    for key in ("git_sha", "prompt_hashes", "source_hash"):
        if old.get(key) is not None and old[key] != current.get(key):
            raise ValueError(f"Continuation requires unchanged simulation code: {key} differs")
    cfg = yaml.safe_load((parent / "config.resolved.yaml").read_text(encoding="utf-8"))
    if cfg.get("world", {}).get("mode") == "commons":
        raise ValueError("Verified continuation currently applies to the campus/co-op engine")
    tpd = int(man["ticks_per_day"])
    total = int(cfg["simulation_days"]) if days is None else int(days)
    if total * tpd <= boundary:
        raise ValueError("--days is the total horizon and must extend beyond the committed parent history")
    cfg["simulation_days"] = total
    cfg["llm"].update(replay_from=str(parent / "llm_calls.jsonl"), replay_until_tick=boundary)
    cfg["_continuation"] = {"parent": str(parent), "at_tick": boundary, "prefix_verified": False}
    sim = Simulation(cfg, out, progress=progress)
    sim.run()
    return out
