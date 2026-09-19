"""Runtime replay verification and random-stream continuation, independent of study design."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from backend.tracing.logger import compare_digests

class PrefixMismatch(RuntimeError):
    """A recorded prefix differs from its parent."""

def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, default=str)

def ticks_per_day(cfg: dict) -> int:
    from backend.simulation.world import Clock
    return Clock({**cfg, "simulation_days": 1}).ticks_per_day


def branch_tick(cfg: dict, at_day: int) -> int:
    """T(at_day): the first tick of day `at_day`."""
    return (int(at_day) - 1) * ticks_per_day(cfg)

def _read_json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def read_config(run_dir: Path) -> dict:
    return yaml.safe_load(open(Path(run_dir) / "config.resolved.yaml"))


def llm_error_count(run_dir: Path) -> int:
    """LLM_ERROR responses in a run's llm_calls.jsonl (§6.1: a valid parent has none)."""
    p = Path(run_dir) / "llm_calls.jsonl"
    if not p.exists():
        return 0
    n = 0
    with open(p) as fh:
        for line in fh:
            if "LLM_ERROR" in line:
                try:
                    if str(json.loads(line).get("response", "")).startswith("LLM_ERROR"):
                        n += 1
                except ValueError:
                    pass
    return n

def at_tick(cfg: dict) -> int | None:
    """T of a branch config, from branch.at_day (not llm.replay_until_tick, which a resume moves)."""
    at_day = (cfg.get("branch") or {}).get("at_day")
    return None if at_day is None else branch_tick(cfg, at_day)


def salt_of(cfg: dict):
    return (cfg.get("branch") or {}).get("salt")


def salted_parts(cfg: dict, tick: int, *parts) -> tuple:
    """Seed parts for a tick-keyed stream: unchanged before T, `+ ("salt", salt)` from T on (salt set)."""
    salt = salt_of(cfg)
    T = at_tick(cfg)
    if salt is None or T is None or int(tick) < int(T):
        return parts
    return (*parts, "salt", salt)


def apply_salt(agents, seed, salt, names=None) -> None:
    """Engine hook at tick T (salted branches only): re-derive every agent substream as
    seed_rng(seed, agent, name, "salt", salt). Also re-binds `agent.rng` (the legacy stream)."""
    from backend.simulation.rngs import seed_rng
    try:
        from backend.agents.agent import STREAMS
    except ImportError:
        STREAMS = ()
    items = agents.values() if isinstance(agents, dict) else agents
    for a in items:
        streams = getattr(a, "_streams", {})
        for name in sorted(set(streams) | set(STREAMS) | set(names or ())):
            streams[name] = seed_rng(seed, a.id, name, "salt", salt)
        a._streams = streams
        if "legacy" in streams:
            a.rng = streams["legacy"]


# ---------------------------------------------------------------------------------- identity checks
def read_digests(run_dir: Path) -> dict[int, str]:
    """digests.jsonl -> {tick: sha256}. Accepts {"tick", "sha256"|"digest"} records."""
    p = Path(run_dir) / "digests.jsonl"
    out: dict[int, str] = {}
    if not p.exists():
        return out
    with open(p) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[int(r["tick"])] = r.get("sha256") or r.get("digest")
    return out


_EMPTY = hashlib.sha256().hexdigest()


def prefix_digest(digests: dict[int, str], T: int) -> str:
    """One sha256 over the per-tick digests of ticks < T; a tick without a digest counts as the empty digest
    (the same definition as backend.tracing.logger.prefix_digest)."""
    h = hashlib.sha256()
    for t in range(int(T)):
        h.update(f"{t}:{digests.get(t, _EMPTY)}\n".encode())
    return h.hexdigest()


def _script_records(run_dir: Path, at_day: int, T: int) -> list[str]:
    p = Path(run_dir) / "world_script.jsonl"
    out = []
    if not p.exists():
        return out
    with open(p) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            day = r.get("day")
            tick = r.get("start_tick")
            if (day is not None and int(day) < at_day) or (day is None and tick is not None and int(tick) < T):
                out.append(_canon(r))
    return out


def check_prefix(parent_run: str | Path, branch_run: str | Path, at_day: int | None = None,
                 require_digests: bool = True, write: bool = True) -> dict:
    """Identity checks between a parent and a branch run for everything before T(at_day). Returns
    {ok, failures: [..], T, parent_prefix_digest, prefix_digest, n_ticks}. With `write`, the digests and the
    result are stored in the branch's branch.json. Works mid-run (on what the branch has flushed so far),
    so the engine can call it right before its first live tick."""
    parent_run, branch_run = Path(parent_run), Path(branch_run)
    bcfg = read_config(branch_run)
    at_day = int(at_day if at_day is not None else (bcfg.get("branch") or {}).get("at_day"))
    T = branch_tick(bcfg, at_day)
    failures = []
    pd, bd = read_digests(parent_run), read_digests(branch_run)
    ppre = {t: d for t, d in pd.items() if t < T}
    bpre = {t: d for t, d in bd.items() if t < T}
    if not ppre and require_digests:
        failures.append("parent has no per-tick digests before T")
    if ppre != bpre and (ppre or require_digests):
        diff = sorted(t for t in set(ppre) | set(bpre) if ppre.get(t) != bpre.get(t))
        failures.append(f"per-tick digests differ at {len(diff)} ticks (first {diff[:5]})")
    if _script_records(parent_run, at_day, T) != _script_records(branch_run, at_day, T):
        failures.append(f"world-script records of days < {at_day} differ")
    pm, bm = _read_json(parent_run / "manifest.json") or {}, _read_json(branch_run / "manifest.json") or {}
    pv, bv = pm.get("code_version") or {}, bm.get("code_version") or {}
    if bm and pv.get("git_sha") != bv.get("git_sha"):
        failures.append(f"git sha differs ({pv.get('git_sha')} vs {bv.get('git_sha')})")
    if bm and (pv.get("prompt_hashes") or {}) != (bv.get("prompt_hashes") or {}):
        failures.append("prompt hashes differ")
    if pm.get("status") != "finished":
        failures.append(f"parent status {pm.get('status')!r}")
    if llm_error_count(parent_run):
        failures.append("parent has LLM_ERROR records")
    res = {"ok": not failures, "failures": failures, "T": T, "at_day": at_day, "n_ticks": len(bpre),
           "parent_prefix_digest": prefix_digest(pd, T), "prefix_digest": prefix_digest(bd, T)}
    if write:
        bj = branch_run / "branch.json"
        rec = _read_json(bj) or {}
        rec.update(parent_prefix_digest=res["parent_prefix_digest"], prefix_digest=res["prefix_digest"],
                   prefix_check={"ok": res["ok"], "failures": failures})
        bj.write_text(json.dumps(rec, indent=1, sort_keys=True))
    return res


def assert_prefix(parent_run, branch_run, at_day=None, **kw) -> dict:
    res = check_prefix(parent_run, branch_run, at_day, **kw)
    if not res["ok"]:
        raise PrefixMismatch(f"{branch_run}: prefix identity failed: " + "; ".join(res["failures"]))
    return res


def manifest_block(run_dir: str | Path) -> dict | None:
    """What the engine puts under manifest['branch'] (branch.json, if this run is a branch)."""
    return _read_json(Path(run_dir) / "branch.json")


# ---------------------------------------------------------------------------------- engine hook
def on_branch_tick(sim, tick: int) -> dict | None:
    """Engine hook, called at the top of every tick (before phase 0) of a branch run: at tick T it flushes,
    runs the identity checks against the parent (aborting on failure) and applies the salt."""
    b = sim.cfg.get("branch") or {}
    T = at_tick(sim.cfg)
    if not b.get("parent") or T is None or int(tick) != int(T):
        return None
    sim.tracer.flush()
    res = None
    if int(T) > 0:
        res = assert_prefix(b["parent"], sim.run_dir, b.get("at_day"), write=True)
    if b.get("salt") is not None:
        apply_salt(sim.agents, sim.cfg["seed"], b["salt"])
    return res


def check_continuation(sim, tick):
    info = sim.cfg["_continuation"]
    if tick != info["at_tick"]:
        return
    sim.tracer.flush()
    mismatches = compare_digests(Path(info["parent"]), sim.run_dir, tick)
    if mismatches:
        raise RuntimeError(f"Continuation prefix diverged at ticks {mismatches[:10]}")
    info["prefix_verified"] = True
    sim.write_manifest("running")


