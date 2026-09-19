"""MemeWorld command line.

  python -m backend.cli run --config configs/baseline.yaml [--set llm.backend=mock ...]
  python -m backend.cli replay runs/<run_id>          # deterministic re-execution from recorded LLM outputs
  python -m backend.cli analyze runs/<run_id>         # external meme analyzer (observer layer)
  python -m backend.cli compare runs/a runs/b ...      # cross-condition table
  python -m backend.cli serve [--port 8765]           # API + frontend
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.config import load_config, parse_overrides


def cmd_run(args):
    from backend.simulation.engine import Simulation, new_run_dir
    cfg = load_config(args.config, parse_overrides(args.set))
    run_dir = Path(args.out) if args.out else new_run_dir(cfg)
    print(f"run dir: {run_dir}")
    sim = Simulation(cfg, run_dir)
    sim.run()
    print(json.dumps(sim.stats), json.dumps(sim.llm.stats))
    if args.analyze:
        from backend.analysis.pipeline import analyze
        analyze(run_dir, llm_backend=args.analysis_backend)
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
    from backend.analysis.pipeline import analyze
    analyze(Path(args.run_dir), llm_backend=args.backend, probes=not args.no_probes, llm_model=args.model)


def cmd_compare(args):
    from backend.analysis.compare import compare, markdown
    rows = compare(args.run_dirs)
    print(markdown(rows))
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


def cmd_serve(args):
    import uvicorn
    uvicorn.run("backend.api.server:app", host=args.host, port=args.port, reload=False)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="memeworld")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", default="configs/baseline.yaml")
    r.add_argument("--set", nargs="*", default=[])
    r.add_argument("--out")
    r.add_argument("--analyze", action="store_true")
    r.add_argument("--analysis-backend", default=None)
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("replay")
    p.add_argument("run_dir")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_replay)
    a = sub.add_parser("analyze")
    a.add_argument("run_dir")
    a.add_argument("--backend", default=None, help="LLM backend for analyzer (default: run's backend)")
    a.add_argument("--no-probes", action="store_true")
    a.add_argument("--model", default=None, help="LLM model for analyzer (default: run's model)")
    a.set_defaults(fn=cmd_analyze)
    c = sub.add_parser("compare")
    c.add_argument("run_dirs", nargs="+")
    c.add_argument("--json")
    c.set_defaults(fn=cmd_compare)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(fn=cmd_serve)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
