"""Experiment designs (EXPERIMENT CONTROLLER layer): condition grids over config overlays.

A design file (`configs/designs/<name>.yaml`, docs/ONTOLOGY_V2.md §5) names a base config, factors whose
levels are config overlays (crossed full-factorially), extra control cells, seeds, the observer spec and
the pre-registered outcome columns:

    name: bottleneck_factorial
    base: configs/baseline.yaml
    seeds: [1, 2, 3]
    days: 2                          # optional simulation_days override
    observer: {backend: claude_cli, model: sonnet}
    common: {llm: {model: haiku}}    # optional overlay applied to every cell before its levels
    factors: {need: {off: {}, on: {...}}, ...}
    controls:
      no_events: {latent_events: {event_rate: 0}}              # base + common + overlay
      planted: {levels: {need: on, ...}, set: {controls: ...}}  # chosen factor levels + overlay
    outcomes: [n_emerged, n_grounded, funnel.some_key]        # dotted paths into outcomes.json
    runs_root: runs                  # optional; env MEMEWORLD_RUNS_ROOT overrides it

Every cell x seed is one run in `<runs_root>/<design>/<cell_id>/s<seed>` with seed = world_seed = s: all
cells of a seed share the world's random streams (common random numbers) and differ only by their overlays.
The controller only writes configs and reads the observer's `outcomes.json`; it never talks to agents.
"""
from __future__ import annotations

import copy
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from backend.config import REPO_ROOT, check_keys, deep_merge, load_config, observer_spec

DESIGN_KEYS = {"name", "description", "kind", "base", "seeds", "days", "observer", "common", "factors", "controls",
               "outcomes", "runs_root"}
NAME_RE = re.compile(r"^[A-Za-z0-9]+(_[A-Za-z0-9]+)*$")   # no '-' or '__': they separate cell-id parts
RESERVED_COLUMNS = {"cell", "control", "seed", "n", "observer", "inactive", "status", "run_dir"}
SIDECAR = "design_cell.json"    # {pid, host, started, stage, ended, error}: liveness + failure of a cell run
LOG = "design_cell.log"
STATUSES = ("missing", "running", "failed", "finished", "stale", "analyzed")


@dataclass
class Design:
    name: str
    path: Path
    base: str
    seeds: list[int]
    days: int | None
    observer: dict
    common: dict
    factors: dict[str, dict[str, dict]]      # factor -> level -> overlay, in file order
    controls: dict[str, dict]                # name -> {"levels": {factor: level | None}, "set": overlay}
    outcomes: list[str]
    runs_root: Path
    sha256: str


@dataclass
class Cell:
    design: str
    cell_id: str
    levels: dict[str, str | None]            # None: the factor's overlay is not applied (control cells)
    control: str | None
    seed: int
    base: str
    overrides: dict                          # everything on top of `base`, incl. seed, observer, _condition
    run_dir: Path

    def config(self) -> dict:
        return load_config(self.base, self.overrides)


# ------------------------------------------------------------------------------------------ loading
def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _level_name(v) -> str:
    """YAML 1.1 reads bare on/off/yes/no as booleans; level names are always strings."""
    return {True: "on", False: "off"}.get(v, str(v)) if isinstance(v, bool) else str(v)


def _check_name(name: str, what: str):
    if not NAME_RE.match(name):
        raise ValueError(f"{what} {name!r}: use letters, digits and single underscores "
                         f"('-' and '__' separate the parts of a cell id)")


def _leaves(d: dict, prefix: str = "") -> list[str]:
    out = []
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out += _leaves(v, f"{prefix}{k}.")
        else:
            out.append(f"{prefix}{k}")
    return out


def _check_overlay(ov, where: str):
    if not isinstance(ov, dict):
        raise ValueError(f"{where}: an overlay must be a mapping of config keys, got {ov!r}")
    bad = [k for k in ov if k in ("seed", "world_seed") or str(k).startswith("_")]
    if bad:
        raise ValueError(f"{where}: {bad} are set by the design runner (seed = world_seed = s, _condition)")
    check_keys(ov, where)


def load_design(path: str | Path, runs_root: str | Path | None = None) -> "Design | TreeDesign":
    """Parse and validate a design file. Factors must be orthogonal (no two factors may set the same
    config key), or the full factorial would silently collapse cells."""
    p = _abs(path)
    text = p.read_text()
    raw = yaml.safe_load(text) or {}
    kind = raw.get("kind", "factorial")
    if kind == "tree":
        return load_tree(p, raw, text, runs_root)
    if kind != "factorial":
        raise ValueError(f"{p}: unknown design kind {kind!r} (factorial | tree)")
    extra = set(raw) - DESIGN_KEYS
    if extra:
        raise ValueError(f"{p}: unknown design keys {sorted(extra)}; allowed: {sorted(DESIGN_KEYS)}")
    name = str(raw.get("name") or p.stem)
    _check_name(name, "design name")
    base = str(raw.get("base") or "configs/baseline.yaml")
    if not _abs(base).exists():
        raise FileNotFoundError(f"{p}: base config {base} not found")
    seeds = [int(s) for s in raw.get("seeds") or []]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"{p}: seeds must be a non-empty list of distinct integers")
    days = raw.get("days")
    observer = {"backend": None, "model": None, **(raw.get("observer") or {})}
    if set(observer) != {"backend", "model"}:
        raise ValueError(f"{p}: observer takes only backend and model, got {sorted(observer)}")
    common = raw.get("common") or {}
    _check_overlay(common, f"{name}: common")

    factors: dict[str, dict[str, dict]] = {}
    owners: list[tuple[str, str]] = []        # (leaf key, factor) for the orthogonality check
    for f, levels in (raw.get("factors") or {}).items():
        f = str(f)
        _check_name(f, "factor")
        if f in RESERVED_COLUMNS:
            raise ValueError(f"{p}: factor name {f!r} is reserved for a table column")
        if not isinstance(levels, dict) or not levels:
            raise ValueError(f"{p}: factor {f} needs a mapping level -> overlay")
        factors[f] = {}
        for lv, ov in levels.items():
            lv = _level_name(lv)
            _check_name(lv, f"level of factor {f}")
            ov = ov or {}
            _check_overlay(ov, f"{name}: factor {f}={lv}")
            factors[f][lv] = ov
            for leaf in _leaves(ov):
                for other, g in owners:
                    if g != f and (leaf == other or leaf.startswith(other + ".") or other.startswith(leaf + ".")):
                        raise ValueError(f"{p}: factors {g} and {f} both set {other if len(other) < len(leaf) else leaf}; "
                                         f"factors must touch disjoint config keys")
                owners.append((leaf, f))

    controls: dict[str, dict] = {}
    for c, spec in (raw.get("controls") or {}).items():
        c = str(c)
        _check_name(c, "control")
        spec = spec or {}
        structured = isinstance(spec, dict) and spec and set(spec) <= {"levels", "set"}
        levels = {str(k): _level_name(v) for k, v in ((spec.get("levels") or {}) if structured else {}).items()}
        ov = (spec.get("set") or {}) if structured else spec
        for f, lv in levels.items():
            if f not in factors or lv not in factors[f]:
                raise ValueError(f"{p}: control {c} refers to unknown level {f}={lv}")
        _check_overlay(ov, f"{name}: control {c}")
        controls[c] = {"levels": {f: levels.get(f) for f in factors}, "set": ov}

    outcomes = [str(o) for o in raw.get("outcomes") or []]
    clash = set(outcomes) & (RESERVED_COLUMNS | set(factors))
    if clash:
        raise ValueError(f"{p}: outcome names {sorted(clash)} clash with table columns")
    root = runs_root or os.environ.get("MEMEWORLD_RUNS_ROOT") or raw.get("runs_root") or "runs"
    return Design(name=name, path=p, base=base, seeds=seeds, days=int(days) if days is not None else None,
                  observer=observer, common=common, factors=factors, controls=controls, outcomes=outcomes,
                  runs_root=_abs(root), sha256=hashlib.sha256(text.encode()).hexdigest())


# ---------------------------------------------------------------------------------------- expansion
def cell_specs(design: Design) -> list[tuple[str, dict, str | None, dict]]:
    """(cell_id, levels, control, merged overlay) for every cell, factorial cells first, in file order."""
    if isinstance(design, TreeDesign):
        return [(n, {}, None, t.set) for n, t in design.nodes.items()]
    names = list(design.factors)
    out = []
    for combo in itertools.product(*(list(design.factors[f]) for f in names)):
        levels = dict(zip(names, combo))
        ov = copy.deepcopy(design.common)
        for f, lv in levels.items():
            ov = deep_merge(ov, design.factors[f][lv])
        out.append(("__".join(f"{f}-{lv}" for f, lv in levels.items()) or "base", levels, None, ov))
    for c, spec in design.controls.items():
        ov = copy.deepcopy(design.common)
        for f, lv in spec["levels"].items():
            if lv is not None:
                ov = deep_merge(ov, design.factors[f][lv])
        out.append((f"control-{c}", dict(spec["levels"]), c, deep_merge(ov, spec["set"])))
    ids = [s[0] for s in out]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"design {design.name}: duplicate cell ids {sorted(dup)}")
    return out


def design_dir(design: Design, backend_override: str | None = None) -> Path:
    """runs/<design>; runs/<design>__<backend> when an override changes the agents' backend, so e.g. mock
    runs of a real design never count as finished cells of the real experiment."""
    if backend_override and backend_override != load_config(design.base, design.common)["llm"]["backend"]:
        return design.runs_root / f"{design.name}__{backend_override}"
    return design.runs_root / design.name


def expand(design: Design, backend_override: str | None = None) -> list[Cell]:
    """All cell x seed runs, seed-major (every cell of seed 1 first), so an interrupted design still
    leaves complete replicates. `backend_override` switches both the agents and the observer."""
    if isinstance(design, TreeDesign):
        return expand_tree(design, backend_override)
    root = design_dir(design, backend_override)
    observer = dict(design.observer)
    if backend_override:
        observer["backend"] = backend_override
    specs = cell_specs(design)
    cells = []
    for s in design.seeds:
        for cid, levels, control, ov in specs:
            cond = {"design": design.name, "cell": cid, "levels": levels, "control": control, "seed": s,
                    "design_sha256": design.sha256}
            fixed = {"seed": s, "world_seed": s, "run_name": f"{design.name}__{cid}",
                     "analysis": {"observer": observer}, "_condition": cond}
            if design.days is not None:
                fixed["simulation_days"] = design.days
            if backend_override:
                fixed["llm"] = {"backend": backend_override}
                cond["backend_override"] = backend_override
            cells.append(Cell(design=design.name, cell_id=cid, levels=dict(levels), control=control, seed=s,
                              base=design.base, overrides=deep_merge(ov, fixed), run_dir=root / cid / f"s{s}"))
    return cells


def select(cells: list[Cell], only: list[str] | None = None, seeds: list[int] | None = None) -> list[Cell]:
    """Cells whose id contains any of `only` (substrings) and whose seed is in `seeds`."""
    return [c for c in cells if (not only or any(o in c.cell_id for o in only)) and (not seeds or c.seed in seeds)]


# ------------------------------------------------------------------------------------------- status
def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _alive(side: dict) -> bool:
    if side.get("ended"):
        return False
    if side.get("host") != socket.gethostname():
        return True                           # another machine's process: cannot check, assume alive
    try:
        os.kill(int(side["pid"]), 0)
    except (ProcessLookupError, KeyError, TypeError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def _observer_matches(out: dict, cell: Cell) -> bool:
    """False when outcomes.json was produced by a different observer (backend, model or analysis version)
    than the design now asks for; such runs are re-analyzed rather than pooled."""
    try:
        from backend.analysis.outcomes import ANALYSIS_VERSION
    except ImportError:
        ANALYSIS_VERSION = None
    rec = out.get("observer") or {}
    cfg = cell.config()
    b, m = observer_spec(cfg)
    want = {"backend": b or cfg["llm"]["backend"], "model": m or cfg["llm"].get("model"),
            "analysis_version": ANALYSIS_VERSION}
    return all(rec.get(k) is None or rec[k] == v for k, v in want.items() if v is not None)


def cell_status(cell: Cell) -> str:
    """missing | running | failed | finished (simulated, not analyzed) | stale (analyzed by another
    observer) | analyzed (outcomes.json present)."""
    d = cell.run_dir
    man, side = _read(d / "manifest.json"), _read(d / SIDECAR)
    if man and man.get("status") == "finished":
        if side and _alive(side):
            return "running"                  # analysis in progress
        out = _read(d / "outcomes.json")
        if out is None:
            return "finished"
        return "analyzed" if _observer_matches(out, cell) else "stale"
    if side:
        return "running" if _alive(side) and not side.get("error") else "failed"
    return "running" if man else "missing"    # a manifest without our sidecar: someone else's live run


def status(design: Design, backend_override: str | None = None, cells: list[Cell] | None = None) -> list[dict]:
    return [{"cell": c.cell_id, "seed": c.seed, "status": node_status(c) if isinstance(c, NodeRun) else cell_status(c),
             "run_dir": str(c.run_dir)}
            for c in (cells if cells is not None else expand(design, backend_override))]


# ------------------------------------------------------------------------------------------ running
def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _move_aside(d: Path) -> Path:
    i = 1
    while d.with_name(f"{d.name}.failed{i}").exists():
        i += 1
    target = d.with_name(f"{d.name}.failed{i}")
    d.rename(target)
    return target


def prepare(cell: Cell) -> str:
    """Status after clearing the way for a fresh simulation: a failed or partial run dir is renamed to
    s<seed>.failed<i> (kept for debugging), since the LLM log is appended to and must not mix attempts."""
    st = cell_status(cell)
    d = cell.run_dir
    partial = st == "missing" and d.exists() and any(p.name != LOG for p in d.iterdir())
    if st == "failed" or partial:
        _move_aside(d)
        return "missing"
    return st


def run_cell(cell: Cell, progress: bool = False) -> str:
    """Simulate (if needed) and analyze one cell x seed in this process with the design's observer.
    Returns the final status. Skips cells that are analyzed or running elsewhere."""
    st = prepare(cell)
    if st in ("analyzed", "running"):
        return st
    cfg = cell.config()
    cell.run_dir.mkdir(parents=True, exist_ok=True)
    side = {"pid": os.getpid(), "host": socket.gethostname(), "started": _now(),
            "stage": "simulate" if st == "missing" else "analyze", "cell": cell.cell_id, "seed": cell.seed}
    write = lambda: (cell.run_dir / SIDECAR).write_text(json.dumps(side, indent=1))  # noqa: E731
    if st == "missing":
        # claim the run dir atomically: two runners with overlapping --only filters must not both simulate
        try:
            fd = os.open(cell.run_dir / SIDECAR, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return "running"
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(side, indent=1))
    else:
        write()
    try:
        if st == "missing":
            from backend.simulation.engine import Simulation
            Simulation(cfg, cell.run_dir, progress=progress).run()
            side["stage"] = "analyze"
            write()
        from backend.analysis.pipeline import analyze
        backend, model = observer_spec(cfg)
        analyze(cell.run_dir, llm_backend=backend, llm_model=model, verbose=progress)
    except BaseException as e:
        side.update(error=f"{type(e).__name__}: {e}", ended=_now())
        write()
        raise
    side.update(stage="done", ended=_now())
    write()
    return cell_status(cell)


ACTIONS = {"missing": "simulate+analyze", "failed": "simulate+analyze (retry)", "finished": "analyze",
           "stale": "re-analyze"}


def cell_command(design: Design, cell: Cell, backend_override: str | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "backend.cli", "design", "run-cell", str(design.path), "--cell", cell.cell_id,
           "--seed", str(cell.seed), "--runs-root", str(design.runs_root)]
    return cmd + (["--backend", backend_override] if backend_override else [])


def run_design(design: Design, cells: list[Cell], parallel: int = 1, backend_override: str | None = None,
               dry_run: bool = False, log: Callable[[str], None] = print) -> list[dict]:
    """Run every cell that is not analyzed (or running elsewhere), one subprocess per cell x seed so a
    crash in one run cannot take the others down. Output goes to <run_dir>/design_cell.log."""
    if isinstance(design, TreeDesign):
        return run_tree(design, cells, backend_override=backend_override, dry_run=dry_run, log=log)
    plan = [(c, cell_status(c)) for c in cells]
    todo = [(c, st) for c, st in plan if st in ACTIONS]
    counts = {s: sum(1 for _, st in plan if st == s) for s in STATUSES}
    log(f"{design.name}: {len(plan)} runs; " + ", ".join(f"{k} {v}" for k, v in counts.items() if v)
        + f"; to do: {len(todo)}")
    if dry_run:
        rows = []
        for c, st in todo:
            log(f"  {c.cell_id} s{c.seed}: {st} -> {ACTIONS[st]}")
            rows.append({"cell": c.cell_id, "seed": c.seed, "status": st, "action": ACTIONS[st],
                         "run_dir": str(c.run_dir), "cmd": cell_command(design, c, backend_override)})
        return rows

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    def one(c: Cell, st: str) -> dict:
        t0 = time.time()
        now = prepare(c)
        if now not in ACTIONS:                # another runner got there since the plan was made
            return {"cell": c.cell_id, "seed": c.seed, "before": st, "rc": 0, "status": now, "seconds": 0.0,
                    "log": str(c.run_dir / LOG)}
        c.run_dir.mkdir(parents=True, exist_ok=True)
        with open(c.run_dir / LOG, "a") as fh:
            fh.write(f"# {_now()} {' '.join(cell_command(design, c, backend_override))}\n")
            fh.flush()
            rc = subprocess.run(cell_command(design, c, backend_override), cwd=REPO_ROOT, env=env,
                                stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
        return {"cell": c.cell_id, "seed": c.seed, "before": st, "rc": rc, "status": cell_status(c),
                "seconds": round(time.time() - t0, 1), "log": str(c.run_dir / LOG)}

    results = []
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        futs = [ex.submit(one, c, st) for c, st in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            flag = "" if r["rc"] == 0 else f"  FAILED rc={r['rc']}, see {r['log']}"
            log(f"[{i}/{len(todo)}] {r['cell']} s{r['seed']}: {r['before']} -> {r['status']} ({r['seconds']}s){flag}")
    return results


# ------------------------------------------------------------------------------------------- tables
def _outcome(out: dict, key: str):
    v = out
    for part in key.split("."):
        if not isinstance(v, dict) or part not in v:
            return None
        v = v[part]
    return len(v) if isinstance(v, list) else v


def table(design: Design, backend_override: str | None = None) -> list[dict]:
    """One row per analyzed run: cell, control, factor levels, seed, the design's outcomes (lists are
    counted), mechanisms the observer marked inactive (on in the config but never fired) and the observer."""
    rows = []
    for c in expand(design, backend_override):
        out = _read(c.run_dir / "outcomes.json")
        if out is None:
            continue
        obs = out.get("observer") or {}
        rows.append({"cell": c.cell_id, "control": c.control, **c.levels, "seed": c.seed,
                     **{o: _outcome(out, o) for o in design.outcomes},
                     "inactive": ",".join(sorted(k for k, v in (out.get("validity") or {}).items() if v == "inactive")),
                     "observer": "/".join(str(obs.get(k)) for k in ("backend", "model", "analysis_version")
                                          if k != "analysis_version" or obs.get(k))})
    return rows


def _num(v) -> float | None:
    if isinstance(v, bool):
        return float(v)
    return float(v) if isinstance(v, (int, float)) and not math.isnan(v) else None


def summarize(design: Design, rows: list[dict]) -> list[dict]:
    """Per cell (all cells, n = 0 where nothing is analyzed yet): outcome -> (mean, sd over seeds, n)."""
    out = []
    for cid, levels, control, _ in cell_specs(design):
        mine = [r for r in rows if r["cell"] == cid]
        stats = {}
        for o in design.outcomes:
            xs = [x for x in (_num(r.get(o)) for r in mine) if x is not None]
            mean = sum(xs) / len(xs) if xs else None
            sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
            stats[o] = (mean, sd, len(xs))
        inactive = sorted({m for r in mine for m in r["inactive"].split(",") if m})
        out.append({"cell": cid, "control": control, **levels, "n": len(mine), **stats,
                    "inactive": ",".join(inactive)})
    return out


def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, tuple):
        mean, sd, _ = v
        return "–" if mean is None else f"{mean:.2f}" + ("" if sd is None else f" ± {sd:.2f}")
    return f"{v:g}" if isinstance(v, float) else str(v)


def markdown(design: Design, rows: list[dict]) -> str:
    """Per-cell mean ± sd over seeds, one column per factor and pre-registered outcome."""
    cols = ["cell", *design.factors, "n", *design.outcomes, "inactive"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for s in summarize(design, rows):
        lines.append("| " + " | ".join(_fmt(s.get(c)) for c in cols) + " |")
    observers = sorted({r["observer"] for r in rows})
    if len(observers) > 1:
        lines.append(f"\nWARNING: rows were analyzed by different observers {observers}; do not compare them.")
    return "\n".join(lines)


def write_csv(design: Design, rows: list[dict], path: str | Path):
    """Per-run rows (one line per cell x seed), for analysis outside MemeWorld."""
    cols = ["cell", "control", *design.factors, "seed", *design.outcomes, "inactive", "observer"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# =========================================================================================== trees
# docs/ONTOLOGY_V3.md §6.3: `kind: tree` designs. A tree names a trunk and branches; every node x seed is one
# run in `<runs_root>/<design>/<node>/s<seed>`. A branch replays its parent's prefix (days < at_day) from the
# parent's llm_calls.jsonl (backend.experiment.branch) and runs live from T(at_day). Nodes run one at a time,
# in dependency order, seed-major.
TREE_KEYS = {"name", "description", "kind", "base", "seeds", "observer", "common", "freeze", "tree", "targets",
             "probes", "prereg", "outcomes", "runs_root"}
NODE_KEYS = {"parent", "at_day", "days", "set", "seeds"}
TREE_STATUSES = ("missing", "waiting_parent", "running", "paused", "failed", "finished", "stale", "analyzed")


@dataclass
class TreeNode:
    name: str
    parent: str | None
    at_day: int | None
    days: int
    set: dict
    seeds: list[int] | None                  # None: all of the design's seeds


@dataclass
class TreeDesign:
    name: str
    path: Path
    base: str
    seeds: list[int]
    observer: dict
    common: dict
    freeze: dict
    nodes: dict[str, TreeNode]               # dependency order (parents first)
    targets: dict
    probes: dict
    prereg: str | None
    outcomes: list[str]
    runs_root: Path
    sha256: str
    kind: str = "tree"
    days: int | None = None
    factors: dict = None                     # empty: keeps the factorial table/markdown helpers working
    controls: dict = None

    def __post_init__(self):
        self.factors = self.factors or {}
        self.controls = self.controls or {}


@dataclass
class NodeRun:
    design: str
    node: str
    seed: int
    parent: str | None
    at_day: int | None
    days: int
    base: str
    overrides: dict                          # base + common + fixed (trunk) or the node's `set` (branch)
    run_dir: Path
    parent_run: "NodeRun | None" = None
    freeze: bool = True
    condition: dict = None

    # Cell-compatible surface (status, select, CLI run-cell)
    @property
    def cell_id(self) -> str:
        return self.node

    @property
    def levels(self) -> dict:
        return {}

    @property
    def control(self):
        return None

    def config(self) -> dict:
        if self.parent_run is None:
            return load_config(self.base, self.overrides)
        from backend.experiment import branch as B
        cfg = B.branch_config(self.parent_run.config(), self.parent_run.run_dir, self.at_day, self.overrides,
                              days=self.days, condition=self.condition)
        cfg["run_name"] = f"{self.design}__{self.node}"
        return cfg


def _topo(nodes: dict[str, TreeNode], where) -> dict[str, TreeNode]:
    out: dict[str, TreeNode] = {}
    pending = dict(nodes)
    while pending:
        ready = [n for n, t in pending.items() if t.parent is None or t.parent in out]
        if not ready:
            raise ValueError(f"{where}: tree has a cycle or unknown parents among {sorted(pending)}")
        for n in ready:
            out[n] = pending.pop(n)
    return out


def load_tree(p: Path, raw: dict, text: str, runs_root=None) -> TreeDesign:
    from backend.experiment import branch as B
    extra = set(raw) - TREE_KEYS
    if extra:
        raise ValueError(f"{p}: unknown tree design keys {sorted(extra)}; allowed: {sorted(TREE_KEYS)}")
    name = str(raw.get("name") or p.stem)
    _check_name(name, "design name")
    base = str(raw.get("base") or "configs/v3_base.yaml")
    if not _abs(base).exists():
        raise FileNotFoundError(f"{p}: base config {base} not found")
    seeds = [int(s) for s in raw.get("seeds") or []]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"{p}: seeds must be a non-empty list of distinct integers")
    observer = {"backend": None, "model": None, **(raw.get("observer") or {})}
    if set(observer) != {"backend", "model"}:
        raise ValueError(f"{p}: observer takes only backend and model, got {sorted(observer)}")
    common = raw.get("common") or {}
    _check_overlay(common, f"{name}: common")
    nodes: dict[str, TreeNode] = {}
    for n, spec in (raw.get("tree") or {}).items():
        n = str(n)
        _check_name(n, "tree node")
        spec = spec or {}
        bad = set(spec) - NODE_KEYS
        if bad:
            raise ValueError(f"{p}: node {n}: unknown keys {sorted(bad)}; allowed: {sorted(NODE_KEYS)}")
        parent = spec.get("parent")
        if parent is None and spec.get("at_day") is not None:
            raise ValueError(f"{p}: node {n}: a root node has no at_day")
        if parent is not None and spec.get("at_day") is None:
            raise ValueError(f"{p}: node {n}: a branch needs at_day")
        if "days" not in spec:
            raise ValueError(f"{p}: node {n}: days is required")
        nseeds = [int(s) for s in spec["seeds"]] if spec.get("seeds") is not None else None
        if nseeds and not set(nseeds) <= set(seeds):
            raise ValueError(f"{p}: node {n}: seeds {nseeds} are not all design seeds")
        ov = spec.get("set") or {}
        if parent is not None:
            B.check_overlay(ov, int(spec["at_day"]))          # whitelist, rejected at load
        else:
            _check_overlay(ov, f"{name}: node {n}")           # a root's set is a free overlay (own world)
        nodes[n] = TreeNode(name=n, parent=str(parent) if parent is not None else None,
                            at_day=int(spec["at_day"]) if spec.get("at_day") is not None else None,
                            days=int(spec["days"]), set=ov, seeds=nseeds)
    if not nodes:
        raise ValueError(f"{p}: tree has no nodes")
    for t in nodes.values():
        if t.parent is not None and t.parent not in nodes:
            raise ValueError(f"{p}: node {t.name}: unknown parent {t.parent}")
    nodes = _topo(nodes, p)
    # dated-entry consistency against the parent's merged config (seed-independent parts)
    merged: dict[str, dict] = {}
    for t in nodes.values():
        if t.parent is None:
            merged[t.name] = load_config(base, deep_merge(deep_merge(common, t.set), {"simulation_days": t.days}))
        else:
            B.check_overlay(t.set, t.at_day, merged[t.parent])
            if t.at_day > int(merged[t.parent]["simulation_days"]) + 1:
                raise ValueError(f"{p}: node {t.name}: at_day {t.at_day} is after its parent's last day + 1")
            merged[t.name] = deep_merge(merged[t.parent], {**t.set, "simulation_days": t.at_day - 1 + t.days})
    outcomes = [str(o) for o in raw.get("outcomes") or []]
    root = runs_root or os.environ.get("MEMEWORLD_RUNS_ROOT") or raw.get("runs_root") or "runs"
    freeze = raw.get("freeze") or {}
    return TreeDesign(name=name, path=p, base=base, seeds=seeds, observer=observer, common=common, freeze=freeze,
                      nodes=nodes, targets=raw.get("targets") or {}, probes=raw.get("probes") or {},
                      prereg=raw.get("prereg"), outcomes=outcomes, runs_root=_abs(root),
                      sha256=hashlib.sha256(text.encode()).hexdigest())


def expand_tree(design: TreeDesign, backend_override: str | None = None) -> list[NodeRun]:
    """Every node x seed run, seed-major and in dependency order (a parent always precedes its branches)."""
    root = design_dir(design, backend_override)
    observer = dict(design.observer)
    if backend_override:
        observer["backend"] = backend_override
    freeze = bool(design.freeze.get("git_sha") or design.freeze.get("prompt_hashes"))
    out: list[NodeRun] = []
    for s in design.seeds:
        by_name: dict[str, NodeRun] = {}
        for t in design.nodes.values():
            if t.seeds is not None and s not in t.seeds:
                continue
            if t.parent is not None and t.parent not in by_name:
                continue                                   # the parent does not run for this seed
            cond = {"design": design.name, "node": t.name, "parent": t.parent, "at_day": t.at_day, "seed": s,
                    "design_sha256": design.sha256}
            if backend_override:
                cond["backend_override"] = backend_override
            if t.parent is None:
                fixed = {"seed": s, "world_seed": s, "run_name": f"{design.name}__{t.name}",
                         "simulation_days": t.days, "analysis": {"observer": observer}, "_condition": cond}
                if backend_override:
                    fixed["llm"] = {"backend": backend_override}
                ov = deep_merge(deep_merge(copy.deepcopy(design.common), t.set), fixed)
                parent_run = None
            else:
                ov = copy.deepcopy(t.set)
                parent_run = by_name[t.parent]
            nr = NodeRun(design=design.name, node=t.name, seed=s, parent=t.parent, at_day=t.at_day, days=t.days,
                         base=design.base, overrides=ov, run_dir=root / t.name / f"s{s}", parent_run=parent_run,
                         freeze=freeze, condition=cond)
            by_name[t.name] = nr
            out.append(nr)
    return out


def node_status(nr: NodeRun) -> str:
    """cell_status plus `paused` (manifest status paused, no live process) and `waiting_parent` (missing
    while the parent has not finished)."""
    d = nr.run_dir
    man, side = _read(d / "manifest.json"), _read(d / SIDECAR)
    if man and man.get("status") == "paused" and not (side and _alive(side) and not side.get("error")):
        return "paused"
    st = cell_status(nr)
    if st == "missing" and nr.parent_run is not None and node_status(nr.parent_run) not in (
            "finished", "analyzed", "stale"):
        return "waiting_parent"
    return st


def _prepare_node(nr: NodeRun) -> tuple[str, dict | None]:
    """Status after clearing the way; for a paused run returns the resume overlay (prefix replay from its own
    partial llm_calls.jsonl up to last_complete_tick + 1, §6.1)."""
    st = node_status(nr)
    d = nr.run_dir
    if st == "paused":
        man = _read(d / "manifest.json") or {}
        aside = _move_aside(d)
        T = int(man.get("last_complete_tick", -1)) + 1
        return "missing", {"llm": {"replay_from": str(aside / "llm_calls.jsonl"), "replay_until_tick": T}}
    partial = st == "missing" and d.exists() and any(p.name != LOG for p in d.iterdir())
    if st == "failed" or partial:
        _move_aside(d)
        return "missing", None
    return st, None


def _select_targets(design: TreeDesign, nr: NodeRun, log=print):
    sel = (design.targets or {}).get("select_at") or {}
    if sel.get("node") != nr.node:
        return None
    try:
        from backend.analysis.battery.targets import select_targets
    except ImportError:
        log(f"  {nr.node} s{nr.seed}: backend.analysis.battery.targets not available; target selection skipped")
        return None
    return select_targets(nr.run_dir, sel.get("checkpoint", "C3"))


def run_node(nr: NodeRun, progress: bool = False, analyze: bool = True, design: TreeDesign | None = None) -> str:
    """Simulate (if needed) and analyze one node x seed in this process. A branch first checks its parent
    (finished, zero LLM errors, frozen code) and afterwards verifies prefix identity."""
    from backend.experiment import branch as B
    st, resume = _prepare_node(nr)
    if st in ("analyzed", "running", "waiting_parent"):
        return st
    cfg = nr.config()
    if resume:
        cfg = deep_merge(cfg, resume)
    if nr.parent_run is not None and st == "missing":
        man = B.check_parent(nr.parent_run.run_dir, freeze=nr.freeze)
    nr.run_dir.mkdir(parents=True, exist_ok=True)
    side = {"pid": os.getpid(), "host": socket.gethostname(), "started": _now(),
            "stage": "simulate" if st == "missing" else "analyze", "cell": nr.node, "seed": nr.seed}
    write = lambda: (nr.run_dir / SIDECAR).write_text(json.dumps(side, indent=1))  # noqa: E731
    if st == "missing":
        try:
            fd = os.open(nr.run_dir / SIDECAR, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return "running"
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(side, indent=1))
    else:
        write()
    try:
        if st == "missing":
            if nr.parent_run is not None:
                ov = nr.overrides
                (nr.run_dir / "branch.json").write_text(json.dumps(
                    B.branch_record(cfg, nr.parent_run.run_dir, ov, man), indent=1, sort_keys=True))
            from backend.simulation.engine import Simulation
            Simulation(cfg, nr.run_dir, progress=progress).run()
            if nr.parent_run is not None:
                B.assert_prefix(nr.parent_run.run_dir, nr.run_dir, nr.at_day)
            if design is not None:
                _select_targets(design, nr)
            side["stage"] = "analyze"
            write()
        if analyze:
            from backend.analysis.pipeline import analyze as _analyze
            backend, model = observer_spec(cfg)
            _analyze(nr.run_dir, llm_backend=backend, llm_model=model, verbose=progress)
    except BaseException as e:
        side.update(error=f"{type(e).__name__}: {e}", ended=_now())
        write()
        raise
    side.update(stage="done", ended=_now())
    write()
    return node_status(nr)


def run_tree(design: TreeDesign, nodes: list[NodeRun], backend_override: str | None = None, dry_run: bool = False,
             log: Callable[[str], None] = print, analyze: bool = True) -> list[dict]:
    """Run every node x seed not yet analyzed, ONE simulation at a time (§6.6), in dependency order; one
    subprocess per node (`design run-cell`). A node whose parent failed in this pass is skipped."""
    plan = [(n, node_status(n)) for n in nodes]
    counts = {s: sum(1 for _, st in plan if st == s) for s in TREE_STATUSES}
    todo = [(n, st) for n, st in plan if (st in ACTIONS or st in ("waiting_parent", "paused"))
            and not (st == "finished" and not analyze)]
    log(f"{design.name}: {len(plan)} node runs; " + ", ".join(f"{k} {v}" for k, v in counts.items() if v)
        + f"; to do: {len(todo)}")
    if dry_run:
        for n, st in todo:
            log(f"  {n.node} s{n.seed}: {st}" + (f" (after {n.parent})" if n.parent else ""))
        return [{"cell": n.node, "seed": n.seed, "status": st, "parent": n.parent, "run_dir": str(n.run_dir),
                 "cmd": cell_command(design, n, backend_override)} for n, st in todo]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    results, failed = [], set()
    for i, (n, st) in enumerate(todo, 1):
        key = (n.parent, n.seed)
        if n.parent is not None and key in failed:
            failed.add((n.node, n.seed))
            results.append({"cell": n.node, "seed": n.seed, "before": st, "rc": None, "status": "waiting_parent",
                            "seconds": 0.0, "log": ""})
            log(f"[{i}/{len(todo)}] {n.node} s{n.seed}: skipped (parent failed)")
            continue
        t0 = time.time()
        n.run_dir.mkdir(parents=True, exist_ok=True)
        cmd = cell_command(design, n, backend_override) + ([] if analyze else ["--no-analyze"])
        with open(n.run_dir / LOG, "a") as fh:
            fh.write(f"# {_now()} {' '.join(cmd)}\n")
            fh.flush()
            rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL).returncode
        now = node_status(n)
        if rc != 0 or now not in ("finished", "analyzed"):
            failed.add((n.node, n.seed))
        results.append({"cell": n.node, "seed": n.seed, "before": st, "rc": rc, "status": now,
                        "seconds": round(time.time() - t0, 1), "log": str(n.run_dir / LOG)})
        log(f"[{i}/{len(todo)}] {n.node} s{n.seed}: {st} -> {now} ({round(time.time() - t0, 1)}s)"
            + ("" if rc == 0 else f"  FAILED rc={rc}, see {n.run_dir / LOG}"))
    return results


def tree_contrasts(design: TreeDesign, rows: list[dict]) -> list[dict]:
    """Seed-paired contrasts from the prereg file's `contrasts: [{name, a, b, outcome}]` (a - b per seed)."""
    if not design.prereg or not _abs(design.prereg).exists():
        return []
    pre = yaml.safe_load(_abs(design.prereg).read_text()) or {}
    out = []
    for c in pre.get("contrasts") or []:
        o = c["outcome"]
        by = {(r["cell"], r["seed"]): _num(r.get(o)) for r in rows}
        diffs = [by[(c["a"], s)] - by[(c["b"], s)] for s in design.seeds
                 if by.get((c["a"], s)) is not None and by.get((c["b"], s)) is not None]
        mean = sum(diffs) / len(diffs) if diffs else None
        out.append({"name": c["name"], "a": c["a"], "b": c["b"], "outcome": o, "n": len(diffs), "mean_diff": mean,
                    "diffs": diffs})
    return out


def probe_plan(design: TreeDesign, nodes: list[NodeRun], checkpoint: str | None = None) -> list[dict]:
    """(probe spec, node run, checkpoint id) for every `probes:` entry whose node run has finished."""
    plan = []
    for pname, spec in (design.probes or {}).items():
        ckpt = spec.get("checkpoint") or pname
        if checkpoint and ckpt != checkpoint and pname != checkpoint:
            continue
        want = spec.get("nodes") or ([spec["node"]] if spec.get("node") else [])
        seeds = spec.get("seeds")
        for n in nodes:
            if n.node in want and (not seeds or n.seed in seeds):
                plan.append({"probe": pname, "checkpoint": ckpt, "node": n.node, "seed": n.seed,
                             "run_dir": str(n.run_dir), "battery": spec.get("battery"),
                             "ablated": spec.get("ablated"), "status": node_status(n)})
    return plan
