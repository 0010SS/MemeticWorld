"""Branches (EXPERIMENT CONTROLLER layer; docs/ONTOLOGY_V3.md §6.2).

A branch is `{parent, at_day, days, set: overlay, salt}`. Its config is the parent's resolved config plus the
overlay, with

    simulation_days       = at_day - 1 + days
    llm.replay_from       = <parent>/llm_calls.jsonl
    llm.replay_until_tick = T(at_day) = (at_day - 1) * ticks_per_day
    branch                = {parent, at_day, salt}

so days 1..at_day-1 replay from the parent's cache (prefix mode of the LLM client; a pre-T miss raises
PrefixDivergence) and everything from tick T on is live. There is NO state-restore branching: checkpoints are
probe inputs only.

Overlay whitelist (anything else is rejected): dated entries with day >= at_day of
`regimes.schedule`, `regimes.cues`, `records.transitions`, `turnover.waves`, `turnover.reassign`;
`records.history_from_day`, `records.consult_from_day` and `comm.**.*from_day` set to null or a day >= at_day;
plus `branch.salt`, `simulation_days` and `checkpoints`.

Salt (replicate branches): tick-keyed streams mix in the salt only for tick >= T (`salted_parts`), and agent
substreams are re-derived at T as seed_rng(seed, agent, name, "salt", salt) (`apply_salt`). Pre-T draws are
untouched, so the prefix still replays. The world script is never salted.

Identity checks (`check_prefix`): per-tick digests for ticks < T equal the parent's; world-script records of
days < at_day identical; git sha and prompt hashes equal; the parent finished with zero LLM_ERROR records.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from backend.config import deep_merge

# dated list keys: every entry has a `day`; the branch may only add/change entries with day >= at_day
DATED_LISTS = {("regimes", "schedule"), ("regimes", "cues"), ("records", "transitions"), ("turnover", "waves"),
               ("turnover", "reassign")}
FROM_DAY_KEYS = {("records", "history_from_day"), ("records", "consult_from_day")}
FREE_KEYS = {("branch", "salt"), ("simulation_days",), ("checkpoints",)}


class BranchError(ValueError):
    """The overlay or the parent run does not allow a branch."""


class PrefixMismatch(RuntimeError):
    """The branch's replayed prefix differs from its parent's."""


# ------------------------------------------------------------------------------------------ helpers
def _leaves(d: dict, prefix: tuple = ()) -> list[tuple[tuple, Any]]:
    out = []
    for k, v in d.items():
        path = prefix + (str(k),)
        if isinstance(v, dict) and v and not _is_whole(path):
            out += _leaves(v, path)
        else:
            out.append((path, v))
    return out


def _is_whole(path: tuple) -> bool:
    """Keys whose value is taken as a whole (not descended into)."""
    return path in FREE_KEYS or path in DATED_LISTS


def _get(cfg: dict, path: tuple, default=None):
    cur = cfg
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, default=str)


def sha256_json(x) -> str:
    return hashlib.sha256(_canon(x).encode()).hexdigest()


def ticks_per_day(cfg: dict) -> int:
    from backend.simulation.world import Clock
    return Clock({**cfg, "simulation_days": 1}).ticks_per_day


def branch_tick(cfg: dict, at_day: int) -> int:
    """T(at_day): the first tick of day `at_day`."""
    return (int(at_day) - 1) * ticks_per_day(cfg)


# --------------------------------------------------------------------------------------- whitelist
def check_overlay(overlay: dict, at_day: int, parent_cfg: dict | None = None) -> list[str]:
    """Validate a branch overlay against the whitelist (§6.2). Raises BranchError listing every violation;
    returns the dotted keys it sets. With `parent_cfg`, dated lists must keep the parent's entries with
    day < at_day unchanged, and *_from_day keys may not move a parent value that is already < at_day."""
    at_day = int(at_day)
    bad, keys = [], []
    for path, v in _leaves(overlay or {}):
        dotted = ".".join(path)
        keys.append(dotted)
        if path in FREE_KEYS or path[:1] == ("checkpoints",):
            continue
        if path in DATED_LISTS:
            if not isinstance(v, list) or not all(isinstance(e, dict) and "day" in e for e in v):
                bad.append(f"{dotted}: must be a list of entries with a 'day'")
                continue
            new_pre = [e for e in v if int(e["day"]) < at_day]
            if parent_cfg is None:
                # without the parent, entries before at_day are allowed only if they look unchanged later
                continue
            old_pre = [e for e in (_get(parent_cfg, path) or []) if int(e.get("day", 0)) < at_day]
            if _canon(new_pre) != _canon(old_pre):
                bad.append(f"{dotted}: entries with day < {at_day} differ from the parent's "
                           f"({_canon(old_pre)} -> {_canon(new_pre)})")
            continue
        is_from_day = path in FROM_DAY_KEYS or (path[0] == "comm" and path[-1].endswith("from_day"))
        if is_from_day:
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < at_day):
                bad.append(f"{dotted}: must be null or a day >= {at_day}, got {v!r}")
            elif parent_cfg is not None:
                old = _get(parent_cfg, path)
                if old is not None and int(old) < at_day and old != v:
                    bad.append(f"{dotted}: the parent's value {old} is already in effect before day {at_day}")
            continue
        bad.append(f"{dotted}: not a branch key (allowed: dated regimes.schedule/cues, records.transitions, "
                   f"turnover.waves/reassign, records/comm *_from_day, branch.salt, simulation_days, checkpoints)")
    if bad:
        raise BranchError("branch overlay rejected:\n  " + "\n  ".join(bad))
    return keys


# ------------------------------------------------------------------------------------- run reading
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


def code_version() -> dict:
    """The current checkout's git sha + prompt hashes (the engine's `_code_version`)."""
    from backend.simulation.engine import _code_version
    return _code_version()


def check_parent(parent_run: Path, freeze: bool = True, current: dict | None = None) -> dict:
    """Pre-flight: the parent finished, has zero LLM_ERROR records, and (with `freeze`) was produced by the
    same git sha and prompt hashes as the code that will run the branch. Returns the parent manifest."""
    parent_run = Path(parent_run)
    man = _read_json(parent_run / "manifest.json")
    if not man:
        raise BranchError(f"{parent_run}: no manifest.json (parent never ran)")
    if man.get("status") != "finished":
        raise BranchError(f"{parent_run}: parent status is {man.get('status')!r}, not 'finished'")
    if not (parent_run / "llm_calls.jsonl").exists():
        raise BranchError(f"{parent_run}: no llm_calls.jsonl to replay")
    n_err = llm_error_count(parent_run)
    if n_err:
        raise BranchError(f"{parent_run}: parent has {n_err} LLM_ERROR records")
    if freeze:
        cur = current if current is not None else code_version()
        pv = man.get("code_version") or {}
        if pv.get("git_sha") != cur.get("git_sha"):
            raise BranchError(f"{parent_run}: parent git sha {pv.get('git_sha')} != current {cur.get('git_sha')}")
        if (pv.get("prompt_hashes") or {}) != (cur.get("prompt_hashes") or {}):
            raise BranchError(f"{parent_run}: prompt hashes differ from the parent's")
    return man


# ----------------------------------------------------------------------------------------- config
def branch_config(parent_cfg: dict, parent_run: Path, at_day: int, overlay: dict | None = None,
                  salt=None, days: int | None = None, condition: dict | None = None) -> dict:
    """The branch's full config: parent config + overlay + prefix-mode llm keys + branch block."""
    overlay = copy.deepcopy(overlay or {})
    at_day = int(at_day)
    check_overlay(overlay, at_day, parent_cfg)
    if salt is None:
        salt = _get(overlay, ("branch", "salt"))
    cfg = deep_merge(parent_cfg, overlay)
    if days is not None:
        cfg["simulation_days"] = at_day - 1 + int(days)
    elif "simulation_days" not in overlay:
        cfg["simulation_days"] = max(int(parent_cfg["simulation_days"]), at_day)
    if int(cfg["simulation_days"]) < at_day:
        raise BranchError(f"simulation_days {cfg['simulation_days']} ends before at_day {at_day}")
    if at_day < 1 or at_day > int(parent_cfg["simulation_days"]) + 1:
        raise BranchError(f"at_day {at_day}: the parent has only {parent_cfg['simulation_days']} days to replay")
    T = branch_tick(cfg, at_day)
    cfg["llm"] = {**cfg.get("llm", {}), "replay_from": str(Path(parent_run) / "llm_calls.jsonl"),
                  "replay_until_tick": T}
    cfg["branch"] = {**(cfg.get("branch") or {}), "parent": str(parent_run), "at_day": at_day, "salt": salt}
    if condition is not None:
        cfg["_condition"] = condition
    return cfg


def branch_record(cfg: dict, parent_run: Path, overlay: dict | None, parent_manifest: dict | None = None) -> dict:
    """The manifest `branch` block (§6.2); the digests are filled in by check_prefix."""
    b = cfg.get("branch") or {}
    cv = (parent_manifest or {}).get("code_version") or {}
    return {"parent_run": str(parent_run), "at_day": b.get("at_day"),
            "at_tick": (cfg.get("llm") or {}).get("replay_until_tick"),
            "overlay_sha": sha256_json(overlay or {}), "overlay": overlay or {}, "salt": b.get("salt"),
            "parent_prefix_digest": None, "prefix_digest": None,
            "git_sha": cv.get("git_sha"), "prompt_hashes": cv.get("prompt_hashes")}


def make_branch(parent_run: str | Path, at_day: int, overlay: dict | None = None, salt=None,
                days: int | None = None, out: str | Path | None = None, freeze: bool = True,
                condition: dict | None = None) -> tuple[Path, dict]:
    """Create a branch run dir (config.resolved.yaml + branch.json) from a finished parent run. Returns
    (run_dir, cfg); run it with `Simulation(cfg, run_dir).run()` and then `check_prefix`."""
    parent_run = Path(parent_run).resolve()
    man = check_parent(parent_run, freeze=freeze)
    parent_cfg = read_config(parent_run)
    cfg = branch_config(parent_cfg, parent_run, at_day, overlay, salt=salt, days=days, condition=condition)
    if out is None:
        tag = f"b{int(at_day)}_{sha256_json(overlay or {})[:8]}" + (f"_salt{cfg['branch']['salt']}"
                                                                   if cfg["branch"]["salt"] is not None else "")
        out = parent_run.parent / f"{parent_run.name}__{tag}"
    out = Path(out)
    if (out / "manifest.json").exists():
        raise BranchError(f"{out}: already a run dir")
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config.resolved.yaml", "w"), sort_keys=False)
    (out / "branch.json").write_text(json.dumps(branch_record(cfg, parent_run, overlay, man), indent=1,
                                                sort_keys=True))
    return out, cfg


# ------------------------------------------------------------------------------------------- salt
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
