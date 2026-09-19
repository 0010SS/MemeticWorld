"""MemeWorld command line.

  python -m backend.cli run --config configs/baseline.yaml [--set llm.backend=mock ...] [--analyze]
  python -m backend.cli replay runs/<run_id>          # deterministic re-execution from recorded LLM outputs
  python -m backend.cli analyze runs/<run_id>         # external meme analyzer (observer layer)
  python -m backend.cli compare runs/a runs/b ...      # cross-condition table
  python -m backend.cli design expand|run|status|table configs/designs/<name>.yaml   # factorial experiments and
                                                     # v3 design trees (kind: tree; nodes run in dependency order)
  python -m backend.cli design probe configs/designs/<tree>.yaml [--checkpoint C4]   # observer battery
  python -m backend.cli branch runs/<parent> --day N --set k=v ... [--days D] [--salt S] [--run]
  python -m backend.cli probe runs/<run> --checkpoint C4 [--battery lb1] [--modes memory,situated,ablated]
  python -m backend.cli serve [--port 8765]           # API + frontend

`analyze` and `run --analyze` use the config's `analysis.observer` unless --backend/--model are given.
`design run` writes runs/<design>/<cell>/s<seed> (runs root: --runs-root, env MEMEWORLD_RUNS_ROOT, or runs/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from backend.config import load_config, observer_spec, parse_overrides


def cmd_run(args):
    from backend.simulation.engine import Simulation, new_run_dir
    cfg = load_config(args.config, parse_overrides(args.set))
    if args.background:
        from backend.config import resolve_background
        cfg["shared_background"] = {"file": args.background}
        resolve_background(cfg)
    root = os.environ.get("MEMEWORLD_RUNS_ROOT")
    run_dir = Path(args.out) if args.out else new_run_dir(cfg, Path(root) if root else None)
    print(f"run dir: {run_dir}")
    sim = Simulation(cfg, run_dir)
    sim.run()
    print(json.dumps(sim.stats), json.dumps(sim.llm.stats))
    if args.analyze:
        from backend.analysis.pipeline import analyze
        backend, model = observer_spec(cfg, args.analysis_backend, args.analysis_model)
        analyze(run_dir, llm_backend=backend, llm_model=model)
    return run_dir


def cmd_replay(args):
    import yaml
    from backend.simulation.engine import Simulation, trace_digest
    src = Path(args.run_dir)
    cfg = yaml.safe_load(open(src / "config.resolved.yaml"))
    from backend.config import input_fingerprints
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("input_fingerprints") and manifest["input_fingerprints"] != input_fingerprints(cfg):
        raise ValueError("Replay requires the original population and initial-memory files")
    out = Path(args.out) if args.out else src.parent / (src.name + "_replay")
    sim = Simulation(cfg, out, replay_from=src / "llm_calls.jsonl")
    sim.run()
    a, b = trace_digest(src), trace_digest(out)
    print(f"original trace sha256: {a}\nreplayed trace sha256: {b}\nidentical: {a == b}")
    sys.exit(0 if a == b else 1)


def cmd_analyze(args):
    import yaml
    from backend.analysis.pipeline import analyze
    run_dir = Path(args.run_dir)
    resolved = run_dir / "config.resolved.yaml"
    cfg = yaml.safe_load(open(resolved)) if resolved.exists() else {}
    backend, model = observer_spec(cfg, args.backend, args.model)
    analyze(run_dir, llm_backend=backend, probes=not args.no_probes, llm_model=model)


def cmd_compare(args):
    from backend.analysis.compare import compare, markdown
    rows = compare(args.run_dirs)
    print(markdown(rows))
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


def cmd_serve(args):
    import uvicorn
    uvicorn.run("backend.api.server:app", host=args.host, port=args.port, reload=False)


# ------------------------------------------------------------------------------------------- design
def _design(args):
    from backend.experiment import design as D
    return D, D.load_design(args.design, args.runs_root)


def cmd_design_expand(args):
    D, d = _design(args)
    cells = D.expand(d, args.backend)
    if args.json:
        print(json.dumps([{"cell_id": c.cell_id, "levels": c.levels, "control": c.control, "seed": c.seed,
                           "run_dir": str(c.run_dir), "base": c.base, "overrides": c.overrides} for c in cells],
                         indent=1, default=str))
        return
    ids = list(dict.fromkeys(c.cell_id for c in cells))
    print(f"{d.name}: {len(ids)} cells x {len(d.seeds)} seeds = {len(cells)} runs in {D.design_dir(d, args.backend)}")
    w = max(map(len, ids))
    for cid in ids:
        c = next(c for c in cells if c.cell_id == cid)
        lv = " ".join(f"{f}={l}" for f, l in c.levels.items() if l is not None)
        print(f"  {cid:{w}}  {lv}{'  + control ' + c.control if c.control else ''}")


def cmd_design_run(args):
    D, d = _design(args)
    cells = D.select(D.expand(d, args.backend), args.only, args.seed)
    if not cells:
        sys.exit(f"no cells match --only {args.only} --seed {args.seed}")
    if isinstance(d, D.TreeDesign):
        res = D.run_tree(d, cells, backend_override=args.backend, dry_run=args.dry_run,
                         analyze=not args.no_analyze)
    else:
        res = D.run_design(d, cells, parallel=args.parallel, backend_override=args.backend, dry_run=args.dry_run)
    if args.dry_run:
        return
    failed = [r for r in res if r["rc"] != 0]
    done = ("analyzed",) if not getattr(args, "no_analyze", False) else ("analyzed", "finished")
    left = [r for r in D.status(d, cells=cells) if r["status"] not in done]
    print(f"{len(res) - len(failed)} ok, {len(failed)} failed; not analyzed: {len(left)}")
    sys.exit(1 if failed or left else 0)


def cmd_design_run_cell(args):
    """Internal: one cell x seed in this process (what `design run` launches per subprocess)."""
    D, d = _design(args)
    match = [c for c in D.expand(d, args.backend) if c.cell_id == args.cell and c.seed == args.seed
             and getattr(c, "replica", 0) == getattr(args, "replica", 0)]
    if not match:
        sys.exit(f"{d.name}: no cell {args.cell} with seed {args.seed}")
    if isinstance(d, D.TreeDesign):
        print(f"{args.cell} s{args.seed}: "
              f"{D.run_node(match[0], progress=True, analyze=not args.no_analyze, design=d)}")
        return
    print(f"{args.cell} s{args.seed}: {D.run_cell(match[0], progress=True)}")


def cmd_design_status(args):
    D, d = _design(args)
    rows = D.status(d, args.backend)
    by_cell: dict[str, list] = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(f"s{r['seed']}:{r['status']}")
    w = max(map(len, by_cell))
    for cid, sts in by_cell.items():
        print(f"  {cid:{w}}  {' '.join(sts)}")
    counts = {s: sum(r["status"] == s for r in rows)
              for s in (D.TREE_STATUSES if isinstance(d, D.TreeDesign) else D.STATUSES)}
    print(f"{d.name}: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v) + f" (of {len(rows)} runs)")


def cmd_design_table(args):
    D, d = _design(args)
    rows = D.table(d, args.backend)
    print(D.markdown(d, rows))
    print(f"\n{len(rows)} analyzed runs")
    if isinstance(d, D.TreeDesign):
        for c in D.tree_contrasts(d, rows):
            m = "–" if c["mean_diff"] is None else f"{c['mean_diff']:.3f}"
            print(f"  contrast {c['name']}: {c['a']} - {c['b']} on {c['outcome']}: mean {m} (n={c['n']} seeds)")
    if args.csv:
        D.write_csv(d, rows, args.csv)
        print(f"per-run rows -> {args.csv}")


def cmd_design_probe(args):
    D, d = _design(args)
    if not isinstance(d, D.TreeDesign):
        sys.exit("design probe needs a kind: tree design")
    plan = D.probe_plan(d, D.select(D.expand(d, args.backend), args.only, args.seed), args.checkpoint)
    rc = 0
    for p in plan:
        if p["status"] not in ("finished", "analyzed", "stale"):
            print(f"  {p['probe']} {p['node']} s{p['seed']}: node {p['status']}, skipped")
            continue
        print(f"  {p['probe']} {p['node']} s{p['seed']} {p['checkpoint']}: battery {p['battery']}")
        if args.dry_run:
            continue
        try:
            _run_probe(p["run_dir"], p["checkpoint"], battery=p["battery"], ablated=p["ablated"])
        except NotImplementedError as e:
            print(f"    not available: {e}")
            rc = 1
    sys.exit(rc)


def _run_probe(run_dir, checkpoint, battery=None, modes=None, ablated=None):
    """Delegate to the observer's battery runner (backend.analysis.battery.runner.run_battery), which runs
    on isolated copies with its own LLM client and cache (docs/ONTOLOGY_V3.md §5.2)."""
    try:
        from backend.analysis.battery.runner import run_battery
    except ImportError as e:
        raise NotImplementedError(f"backend.analysis.battery.runner is not available ({e})") from e
    import inspect
    params = inspect.signature(run_battery).parameters
    want = {"modes": tuple(modes) if modes else None, "plan": battery if isinstance(battery, str) else None,
            "battery": battery, "ablated": ablated}
    kw = {k: v for k, v in want.items() if v is not None and k in params}
    return run_battery(run_dir, checkpoint, **kw)


def cmd_probe(args):
    run_dir = Path(args.run_dir)
    if not (run_dir / "checkpoints" / args.checkpoint).exists():
        sys.exit(f"{run_dir}: no checkpoint {args.checkpoint}")
    modes = [m for m in (args.modes or "").split(",") if m] or None
    try:
        res = _run_probe(run_dir, args.checkpoint, battery=args.battery, modes=modes)
    except NotImplementedError as e:
        sys.exit(str(e))
    if res is not None:
        print(json.dumps(res, indent=1, default=str) if not isinstance(res, str) else res)


def cmd_branch(args):
    from backend.config import parse_overrides
    from backend.experiment import branch as B
    overlay = parse_overrides(args.set)
    out, cfg = B.make_branch(args.parent_run, args.day, overlay, salt=args.salt, days=args.days, out=args.out,
                             freeze=not args.no_freeze)
    print(f"branch run dir: {out} (T = {cfg['llm']['replay_until_tick']}, days = {cfg['simulation_days']})")
    if args.run:
        from backend.simulation.engine import Simulation
        Simulation(cfg, out).run()
        res = B.check_prefix(args.parent_run, out, args.day)
        print(f"prefix identity: {'ok' if res['ok'] else 'FAILED: ' + '; '.join(res['failures'])}")
        sys.exit(0 if res["ok"] else 1)


def _add_design(sub):
    d = sub.add_parser("design", help="factorial experiments over config overlays (configs/designs/*.yaml)")
    dsub = d.add_subparsers(dest="design_cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("design", help="design file, e.g. configs/designs/bottleneck_factorial.yaml")
    common.add_argument("--backend", default=None,
                        help="override agents' AND observer's LLM backend (e.g. mock); runs go to runs/<design>__<backend>")
    common.add_argument("--runs-root", default=None, help="default: env MEMEWORLD_RUNS_ROOT, design runs_root, runs/")
    e = dsub.add_parser("expand", parents=[common], help="list the cells")
    e.add_argument("--json", action="store_true", help="full cells incl. config overrides")
    e.set_defaults(fn=cmd_design_expand)
    r = dsub.add_parser("run", parents=[common], help="run and analyze every cell x seed not yet analyzed")
    r.add_argument("--parallel", type=int, default=1, help="concurrent runs (one subprocess each)")
    r.add_argument("--only", action="append", default=None, help="only cells whose id contains this (repeatable)")
    r.add_argument("--seed", action="append", type=int, default=None, help="only these seeds (repeatable)")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--no-analyze", action="store_true", help="simulate only (trees)")
    r.set_defaults(fn=cmd_design_run)
    rc = dsub.add_parser("run-cell", parents=[common], help=argparse.SUPPRESS)
    rc.add_argument("--cell", required=True)
    rc.add_argument("--seed", type=int, required=True)
    rc.add_argument("--replica", type=int, default=0)
    rc.add_argument("--no-analyze", action="store_true")
    rc.set_defaults(fn=cmd_design_run_cell)
    pr = dsub.add_parser("probe", parents=[common], help="observer battery on finished tree nodes (probes:)")
    pr.add_argument("--checkpoint", default=None, help="only this checkpoint / probe entry (e.g. C4)")
    pr.add_argument("--only", action="append", default=None)
    pr.add_argument("--seed", action="append", type=int, default=None)
    pr.add_argument("--dry-run", action="store_true")
    pr.set_defaults(fn=cmd_design_probe)
    s = dsub.add_parser("status", parents=[common], help="missing/running/failed/finished/stale/analyzed per run")
    s.set_defaults(fn=cmd_design_status)
    t = dsub.add_parser("table", parents=[common], help="per-cell mean ± sd of the design's outcomes")
    t.add_argument("--csv", default=None, help="also write per-run rows as CSV")
    t.set_defaults(fn=cmd_design_table)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="memeworld")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", default="configs/baseline.yaml")
    r.add_argument("--set", nargs="*", default=[])
    r.add_argument("--out")
    r.add_argument("--background", help="Shared UTF-8 Markdown introduction to the world")
    r.add_argument("--analyze", action="store_true")
    r.add_argument("--analysis-backend", default=None, help="default: the config's analysis.observer")
    r.add_argument("--analysis-model", default=None, help="default: the config's analysis.observer")
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("replay")
    p.add_argument("run_dir")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_replay)
    a = sub.add_parser("analyze")
    a.add_argument("run_dir")
    a.add_argument("--backend", default=None, help="observer LLM backend (default: analysis.observer, else the run's)")
    a.add_argument("--no-probes", action="store_true")
    a.add_argument("--model", default=None, help="observer LLM model (default: analysis.observer, else the run's)")
    a.set_defaults(fn=cmd_analyze)
    c = sub.add_parser("compare")
    c.add_argument("run_dirs", nargs="+")
    c.add_argument("--json")
    c.set_defaults(fn=cmd_compare)
    _add_design(sub)
    from backend.research.commands import add_parsers
    add_parsers(sub)
    b = sub.add_parser("branch", help="ad-hoc branch of a finished run (prefix replay up to --day)")
    b.add_argument("parent_run")
    b.add_argument("--day", type=int, required=True, help="at_day: the first live day")
    b.add_argument("--set", nargs="*", default=[], help="whitelisted overlay keys, e.g. records.transitions=...")
    b.add_argument("--days", type=int, default=None, help="live days (default: to the parent's end)")
    b.add_argument("--salt", default=None, type=int)
    b.add_argument("--out", default=None)
    b.add_argument("--run", action="store_true", help="also simulate it and check prefix identity")
    b.add_argument("--no-freeze", action="store_true", help="skip the git sha / prompt hash equality check")
    b.set_defaults(fn=cmd_branch)
    pb = sub.add_parser("probe", help="observer battery on a run's checkpoint (separate process and cache)")
    pb.add_argument("run_dir")
    pb.add_argument("--checkpoint", required=True)
    pb.add_argument("--battery", default=None, help="battery plan (default: the runner's)")
    pb.add_argument("--modes", default=None, help="comma-separated probe modes (default: the runner's)")
    pb.set_defaults(fn=cmd_probe)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(fn=cmd_serve)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    # Redirected output is still UTF-8 on Windows, matching saved reports and API jobs.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
