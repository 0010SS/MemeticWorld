"""Checkpoints (EXPERIMENT CONTROLLER layer; docs/ONTOLOGY_V3.md §5.1).

`write_checkpoint(sim, day)` is called by the engine's day-end hook (tick order 8b) and writes
`<run>/checkpoints/C{day}/`:

    agents/<id>/associative_memory/   GA format (nodes.json, kw_strength.json, embeddings.json), active agents
    memory_meta.json                  the D50 sidecar (node_id -> MemoryMeta), sorted keys
    agent_state.json                  role, cohort, arrival day, relationship table, scratch essentials,
                                      open matters, day plan, position, rng stream states
    binder.json                       all revisions, archived binders, receipts (world state)
    roster.json                       roster / turnover state
    world_state.json                  HIDDEN: tick, day, regime, mapping, job outcomes to date
    CHECKSUMS                         "<sha256>  <relative path>" per file, sorted by path

Serialisation is deterministic (sorted keys, sets as sorted lists, datetimes as ISO strings), so a branch
that replays the same prefix regenerates byte-identical pre-branch checkpoints. Files are write-once
(an existing `C{d}` is never overwritten) and made read-only.

Checkpoints are PROBE INPUTS ONLY (§6.2: no state-restore branching). Nothing in the simulation reads them.

World objects (binder, roster, workshop / CoopWorld) are found by duck typing on the Simulation. Any object
may provide `checkpoint_state() -> dict` (preferred), else `to_dict()`, `state_dict()` or `snapshot()`; a
dataclass is serialised with `asdict`. The engine can also pass explicit sections through
`write_checkpoint(sim, day, extra={"binder": ..., "roster": ..., "world_state": ...})`.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

import numpy as np

CHECKSUMS = "CHECKSUMS"


# ------------------------------------------------------------------------------------ serialisation
def canon(x: Any) -> Any:
    """JSON-ready, order-independent copy: sets -> sorted lists, tuples -> lists, numpy -> python,
    datetimes -> ISO, dataclasses -> dicts, dict keys -> str."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return canon(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): canon(v) for k, v in x.items()}
    if isinstance(x, (set, frozenset)):
        return sorted((canon(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True, default=str))
    if isinstance(x, (list, tuple)):
        return [canon(v) for v in x]
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.random.Generator):
        return canon(x.bit_generator.state)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def dumps(obj: Any) -> bytes:
    return (json.dumps(canon(obj), sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def object_state(obj: Any) -> Any:
    """The checkpointable state of a world object (see the module docstring)."""
    if obj is None:
        return None
    for meth in ("checkpoint_state", "to_dict", "state_dict", "snapshot"):
        f = getattr(obj, meth, None)
        if callable(f):
            return canon(f())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return canon(obj)
    if isinstance(obj, (dict, list)):
        return canon(obj)
    return canon({k: v for k, v in vars(obj).items() if not k.startswith("_") and not callable(v)})


def _find(sim, *names):
    """First attribute found on sim, sim.coop or sim.ctx (the integrator decides where world objects live)."""
    for holder in (sim, getattr(sim, "coop", None), getattr(sim, "ctx", None)):
        if holder is None:
            continue
        for n in names:
            v = getattr(holder, n, None)
            if v is not None:
                return v
    return None


# ------------------------------------------------------------------------------------------ config
def enabled_for(cfg: dict, day: int) -> bool:
    """checkpoints.enabled and day in checkpoints.days ('all' or a list of days)."""
    cc = cfg.get("checkpoints") or {}
    if not cc.get("enabled", False):
        return False
    days = cc.get("days", "all")
    return days in (None, "all") or int(day) in {int(d) for d in (days or [])}


def checkpoint_id(day: int) -> str:
    return f"C{int(day)}"


# ------------------------------------------------------------------------------------------ sections
def active_ids(sim) -> list[str]:
    """Active agents: roster.active_agents() / ctx.active if present, else every agent."""
    roster = _find(sim, "roster")
    f = getattr(roster, "active_agents", None) if roster is not None else None
    if callable(f):
        ids = f()
    else:
        ids = getattr(getattr(sim, "ctx", None), "active", None)
    if ids is None:
        ids = list(sim.agents)
    ids = [i if isinstance(i, str) else getattr(i, "id", str(i)) for i in ids]
    return sorted(i for i in ids if i in sim.agents)


def _rng_states(agent) -> dict:
    return {name: canon(g.bit_generator.state) for name, g in sorted(getattr(agent, "_streams", {}).items())}


def agent_state(sim, aid: str, roster_state: Any = None) -> dict:
    a = sim.agents[aid]
    prof = a.profile
    st = a.state
    sc = a.scratch
    info = {}
    if isinstance(roster_state, dict):
        members = roster_state.get("members") or roster_state.get("agents") or {}
        if isinstance(members, dict) and isinstance(members.get(aid), dict):
            info = members[aid]
    return {
        "id": aid,
        "name": a.name,
        "role": info.get("role", getattr(prof, "role", None) or (getattr(prof, "demographics", {}) or {}).get("role")),
        "cohort": info.get("cohort", getattr(prof, "cohort", None)),
        "arrival_day": info.get("arrival_day", getattr(prof, "arrival_day", None)),
        "active": info.get("active", getattr(st, "active", True)),
        "relationships": {k: canon(v) for k, v in sorted(getattr(prof, "relationships", {}).items())},
        "scratch": {
            "curr_time": canon(getattr(sc, "curr_time", None)),
            "importance_trigger_max": getattr(sc, "importance_trigger_max", None),
            "importance_trigger_curr": getattr(sc, "importance_trigger_curr", None),
            "importance_ele_n": getattr(sc, "importance_ele_n", None),
            "act_description": getattr(sc, "act_description", None),
            "currently": getattr(sc, "currently", None),
        },
        "state": canon({k: v for k, v in vars(st).items() if k != "path"}),
        "open_matters": canon(getattr(a, "open_matters", [])),
        "day_plan": canon(getattr(a, "day_plan", [])),
        "rng_streams": _rng_states(a),
    }


def active_regime(cfg: dict, day: int) -> str | None:
    """regimes.schedule: the regime of the last entry with entry.day <= day."""
    sched = sorted((cfg.get("regimes") or {}).get("schedule") or [], key=lambda e: int(e.get("day", 1)))
    reg = None
    for e in sched:
        if int(e.get("day", 1)) <= int(day):
            reg = e.get("regime")
    return reg


def mapping_of(cfg: dict, wseed) -> str | None:
    """regimes.mapping: auto -> M1 for an odd world seed, M2 for an even one (§1.4)."""
    m = (cfg.get("regimes") or {}).get("mapping", "auto")
    if m == "auto":
        return None if wseed is None else ("M1" if int(wseed) % 2 else "M2")
    return m


def _script_prefix_sha(sim, day: int, tick: int) -> str | None:
    p = Path(getattr(sim, "run_dir", ".")) / "world_script.jsonl"
    if not p.exists():
        return None
    h = hashlib.sha256()
    for line in sorted(p.read_text().splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        d = r.get("day")
        t = r.get("start_tick", r.get("tick"))
        if (d is not None and int(d) <= day) or (d is None and t is not None and int(t) <= tick):
            h.update(line.encode())
    return h.hexdigest()


def world_state(sim, day: int, tick: int) -> dict:
    """HIDDEN world position. Never shown to agents; the API strips it in demo mode."""
    coop = _find(sim, "coop", "workshop")
    out = {"day": int(day), "tick": int(tick), "time": sim.clock.time_of(tick).isoformat(),
           "world_seed": getattr(sim, "world_seed", None),
           # the script PREFIX up to this checkpoint: branches run longer horizons than their parent, so a
           # whole-file hash would differ although every record the checkpoint depends on is identical
           "world_script_prefix_sha256": _script_prefix_sha(sim, day, tick),
           "legacy_rng": canon(sim.rng) if getattr(sim, "rng", None) is not None else None}
    # regime / mapping from config (§1.4); the integrator's world object, if any, is saved alongside
    out["regime"] = active_regime(sim.cfg, day)
    out["mapping"] = mapping_of(sim.cfg, out["world_seed"])
    if coop is not None:
        out["coop"] = object_state(coop)
    return out


# ------------------------------------------------------------------------------------------- write
def _write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def _files(root: Path) -> list[Path]:
    return sorted((p for p in root.rglob("*") if p.is_file() and p.name != CHECKSUMS),
                  key=lambda p: p.relative_to(root).as_posix())


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def checksums(root: Path) -> dict[str, str]:
    root = Path(root)
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in _files(root)}


def _make_readonly(root: Path):
    for p in sorted(root.rglob("*"), reverse=True):
        mode = os.stat(p).st_mode
        if p.is_file():
            os.chmod(p, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    # directories stay writable by the owner so that run dirs can still be removed; the files are read-only


def write_checkpoint(sim, day: int, extra: dict | None = None, root: Path | None = None) -> dict:
    """Write checkpoints/C{day}/ for `sim` at the end of `day`. Returns the trace payload
    {id, tick, files: {path: sha256}} and logs a `checkpoint` trace record. Write-once: raises
    FileExistsError if C{day} already exists."""
    extra = extra or {}
    cid = checkpoint_id(day)
    base = Path(root) if root else Path(sim.run_dir) / "checkpoints"
    final = base / cid
    if final.exists():
        raise FileExistsError(f"checkpoint {final} already exists (checkpoints are write-once)")
    tmp = base / f".{cid}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    tick = int(day) * sim.clock.ticks_per_day - 1                 # last tick of the day

    roster_obj = extra.get("roster", _find(sim, "roster"))
    roster_state = roster_obj if isinstance(roster_obj, (dict, list)) or roster_obj is None else object_state(roster_obj)
    binder_obj = extra.get("binder", _find(sim, "binder"))
    binder_state = binder_obj if isinstance(binder_obj, (dict, list)) or binder_obj is None else object_state(binder_obj)

    ids = active_ids(sim)
    for aid in ids:
        sim.agents[aid].a_mem.save_ga(tmp / "agents" / aid / "associative_memory")
    meta = getattr(sim, "meta", None)
    meta_d = {nid: dataclasses.asdict(m) for nid, m in (getattr(meta, "meta", {}) or {}).items()}
    _write(tmp / "memory_meta.json", dumps(meta_d))
    _write(tmp / "agent_state.json", dumps({aid: agent_state(sim, aid, roster_state) for aid in ids}))
    _write(tmp / "binder.json", dumps(binder_state if binder_state is not None else {"enabled": False}))
    _write(tmp / "roster.json", dumps(roster_state if roster_state is not None
                                      else {"enabled": False, "active": ids}))
    ws = world_state(sim, day, tick)
    if "world_state" in extra:
        ws.update(canon(extra["world_state"]))
    _write(tmp / "world_state.json", dumps(ws))
    files = checksums(tmp)
    _write(tmp / CHECKSUMS, "".join(f"{h}  {p}\n" for p, h in sorted(files.items())).encode())
    _make_readonly(tmp)
    tmp.rename(final)
    rec = {"id": cid, "tick": tick, "files": files}
    tracer = getattr(sim, "tracer", None)
    if tracer is not None:
        from backend.llm.client import llm_scope
        with llm_scope(f"t{tick:04d}:99checkpoint"):
            tracer.log("checkpoint", **rec)
        tracer.flush()                        # into this tick's digest, not the next one's
    return rec


def maybe_checkpoint(sim, tick: int) -> dict | None:
    """Engine hook (tick order 8b): at the last tick of a day, write C{day} if checkpoints are on for it."""
    tpd = sim.clock.ticks_per_day
    if (tick + 1) % tpd:
        return None
    day = (tick + 1) // tpd
    if not enabled_for(sim.cfg, day):
        return None
    return write_checkpoint(sim, day)


# -------------------------------------------------------------------------------------------- read
class ChecksumMismatch(RuntimeError):
    pass


def read_checksums(path: Path) -> dict[str, str]:
    out = {}
    for line in (Path(path) / CHECKSUMS).read_text().splitlines():
        if line.strip():
            h, p = line.split("  ", 1)
            out[p] = h
    return out


def verify_checkpoint(path: Path) -> bool:
    path = Path(path)
    return (path / CHECKSUMS).exists() and read_checksums(path) == checksums(path)


def checkpoint_digest(path: Path) -> str:
    """One sha256 for the whole checkpoint (sha of CHECKSUMS): identical prefixes share probes (§5.2.8)."""
    return sha256_file(Path(path) / CHECKSUMS)


def load_checkpoint(path: Path, load_memories: bool = False) -> dict:
    """Read a checkpoint (verifying CHECKSUMS first). Returns {id, path, digest, memory_meta, agent_state,
    binder, roster, world_state, agents: [ids], memories: {id: MemoryStream} (if load_memories)}.
    Read-only: nothing here writes into the checkpoint."""
    path = Path(path)
    if not verify_checkpoint(path):
        raise ChecksumMismatch(f"{path}: CHECKSUMS do not match the files")
    rd = lambda n: json.loads((path / n).read_text())  # noqa: E731
    agents_dir = path / "agents"
    ids = sorted(p.name for p in agents_dir.iterdir()) if agents_dir.exists() else []
    out = {"id": path.name, "path": str(path), "digest": checkpoint_digest(path),
           "memory_meta": rd("memory_meta.json"), "agent_state": rd("agent_state.json"),
           "binder": rd("binder.json"), "roster": rd("roster.json"), "world_state": rd("world_state.json"),
           "agents": ids}
    if load_memories:
        from backend.memory.store import MemoryStream
        out["memories"] = {aid: MemoryStream.load_ga(aid, agents_dir / aid / "associative_memory") for aid in ids}
    return out


def list_checkpoints(run_dir: Path) -> list[str]:
    d = Path(run_dir) / "checkpoints"
    if not d.exists():
        return []
    return sorted((p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("C")),
                  key=lambda n: int(n[1:]) if n[1:].isdigit() else 10 ** 9)
