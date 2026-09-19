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


from backend.tracing.replay import (  # re-export the existing public runtime API
    PrefixMismatch, at_tick, salt_of, salted_parts, apply_salt, read_digests, prefix_digest,
    check_prefix, assert_prefix, manifest_block, on_branch_tick,
)
