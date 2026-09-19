"""MemeWorld command line.

  python -m backend.cli run --config configs/baseline.yaml [--set llm.backend=mock ...] [--analyze]
  python -m backend.cli replay runs/<run_id>          # deterministic re-execution from recorded LLM outputs
  python -m backend.cli analyze runs/<run_id>         # external meme analyzer (observer layer)
  python -m backend.cli compare runs/a runs/b ...      # cross-condition table
  python -m backend.cli design expand|run|status|table configs/designs/<name>.yaml   # factorial experiments
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
    res = D.run_design(d, cells, parallel=args.parallel, backend_override=args.backend, dry_run=args.dry_run)
    if args.dry_run:
        return
    failed = [r for r in res if r["rc"] != 0]
    left = [r for r in D.status(d, cells=cells) if r["status"] != "analyzed"]
    print(f"{len(res) - len(failed)} ok, {len(failed)} failed; not analyzed: {len(left)}")
    sys.exit(1 if failed else 0)


def cmd_design_run_cell(args):
    """Internal: one cell x seed in this process (what `design run` launches per subprocess)."""
    D, d = _design(args)
    match = [c for c in D.expand(d, args.backend) if c.cell_id == args.cell and c.seed == args.seed]
    if not match:
        sys.exit(f"{d.name}: no cell {args.cell} with seed {args.seed}")
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
    counts = {s: sum(r["status"] == s for r in rows) for s in D.STATUSES}
    print(f"{d.name}: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v) + f" (of {len(rows)} runs)")


def cmd_design_table(args):
    D, d = _design(args)
    rows = D.table(d, args.backend)
    print(D.markdown(d, rows))
    print(f"\n{len(rows)} analyzed runs")
    if args.csv:
        D.write_csv(d, rows, args.csv)
        print(f"per-run rows -> {args.csv}")


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
    r.set_defaults(fn=cmd_design_run)
    rc = dsub.add_parser("run-cell", parents=[common], help=argparse.SUPPRESS)
    rc.add_argument("--cell", required=True)
    rc.add_argument("--seed", type=int, required=True)
    rc.set_defaults(fn=cmd_design_run_cell)
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
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(fn=cmd_serve)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
