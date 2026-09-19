"""Public command line for full studies and open-ended research on existing runs."""
from __future__ import annotations

import json
import time
from pathlib import Path

from backend.research.common import read_json


def execute(args):
    from backend.research import experiments as E
    from backend.experiment import design as D
    action = args.research_action
    if action == "list":
        for item in E.catalog():
            d = E.resolve(item["id"], args.runs_root)
            print(f"{item['id']:18} {len(D.expand(d)) :4} runs  {item['title']}")
        return
    if action in ("expand", "status", "run", "report", "synthesize"):
        d = E.resolve(args.experiment, args.runs_root)
        cells = D.select(D.expand(d, args.backend), getattr(args, "only", None), getattr(args, "seed", None))
        if action == "expand":
            print(json.dumps({"name": d.name, "questions": d.questions, "runs": [
                {"cell": c.cell_id, "seed": c.seed, "replica": c.replica, "levels": c.levels,
                 "run_dir": str(c.run_dir), "overrides": c.overrides} for c in cells]}, indent=2))
        elif action == "status":
            print(json.dumps(D.status(d, cells=cells), indent=2))
        elif action == "run":
            result = E.run(d, cells, backend_override=args.backend, parallel=args.parallel,
                           dry_run=args.dry_run, questions=not args.no_questions)
            if not args.dry_run:
                print(json.dumps({k: result[k] for k in ("status", "report_dir", "errors")}, indent=2))
                if result["status"] != "complete":
                    raise SystemExit(1)
        elif action == "synthesize":
            result = E.synthesize(d, args.question, backend_override=args.backend)
            print(result["report_dir"])
        else:
            print(E.report(d, args.backend)["report_dir"])
        return
    if action == "observe":
        from backend.research.observer import observe
        last = None
        while True:
            result = observe(args.run_dir, args.backend, args.model, publish=not args.no_publish,
                             fresh=args.fresh and last is None)
            if result["analysis_id"] != last:
                print(json.dumps({k: result[k] for k in ("analysis_id", "status", "coverage", "summary")}))
                last = result["analysis_id"]
            if not args.watch or read_json(Path(args.run_dir) / "manifest.json", {}).get("status") != "running":
                return
            time.sleep(max(1, args.interval))
    elif action == "inquire":
        from backend.research.inquiries import inquire
        result = inquire(args.run_dir, args.question, backend=args.backend, model=args.model,
                         start=args.start, end=args.end, actor=args.agent,
                         include_private=not args.public_only, limit=args.limit)
        print(result["answer"])
        print(Path(args.run_dir) / "inquiries" / result["id"] / "report.html")
    elif action == "export":
        from backend.research.reports import export_run
        run = Path(args.run_dir)
        path = run / "analyses" / args.analysis_id / "analysis.json" if args.analysis_id else run / "analysis.json"
        result = read_json(path)
        if not result or result.get("kind") != "memetics":
            raise ValueError("Run observe first, or select a saved memetics analysis")
        print(export_run(run, result))
    elif action == "pause":
        from backend.experiment.execution import request_pause
        print(json.dumps(request_pause(args.run_dir)))
    elif action == "continue":
        from backend.experiment.execution import continue_run
        out = continue_run(args.run_dir, args.out, args.days)
        if args.analyze:
            from backend.research.observer import observe
            observe(out, args.backend, args.model)
        print(out)
    elif action == "audit":
        from backend.research.audit import audit
        print(json.dumps(audit(args.run_dir, backend=args.backend, model=args.model,
                               sample=args.sample, seed=args.seed), indent=2))


def add_parsers(sub):
    exp = sub.add_parser("experiment", help="Run complete memetics studies; presets are editable starting points")
    commands = exp.add_subparsers(dest="experiment_action", required=True)
    p = commands.add_parser("list")
    p.add_argument("--runs-root")
    p.set_defaults(fn=execute, research_action="list")
    for action in ("expand", "status", "run", "report", "synthesize"):
        p = commands.add_parser(action)
        p.add_argument("experiment", help="Library ID or custom design YAML")
        p.add_argument("--runs-root")
        p.add_argument("--backend", help="Override agents and observer together; mock is software verification only")
        if action in ("expand", "run"):
            p.add_argument("--only", action="append")
            p.add_argument("--seed", action="append", type=int)
        if action == "run":
            p.add_argument("--parallel", type=int, default=1)
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--no-questions", action="store_true", help="Skip optional question-specific inquiries; discovery and reports still run")
        if action == "synthesize":
            p.add_argument("question", help="Any cross-society research question")
        p.set_defaults(fn=execute, research_action=action)
    for action in ("observe", "inquire", "export", "pause", "continue", "audit"):
        p = sub.add_parser(action)
        p.add_argument("run_dir")
        if action in ("observe", "inquire", "continue", "audit"):
            p.add_argument("--backend")
            p.add_argument("--model")
        if action == "observe":
            p.add_argument("--watch", action="store_true", help="Analyze committed prefixes while a simulation runs")
            p.add_argument("--interval", type=float, default=30)
            p.add_argument("--no-publish", action="store_true", help="Save an alternative observer snapshot without replacing the current analysis")
            p.add_argument("--fresh", action="store_true", help="Record another judgment pass with a separate cache, retaining existing observations")
        elif action == "inquire":
            p.add_argument("question")
            p.add_argument("--start", type=int, help="Inclusive start tick")
            p.add_argument("--end", type=int, help="Inclusive end tick")
            p.add_argument("--agent")
            p.add_argument("--public-only", action="store_true")
            p.add_argument("--limit", type=int, default=100)
        elif action == "export":
            p.add_argument("--analysis-id")
        elif action == "continue":
            p.add_argument("--out", required=True)
            p.add_argument("--days", type=int, help="Total simulation horizon, including the recorded prefix")
            p.add_argument("--analyze", action="store_true")
        elif action == "audit":
            p.add_argument("--sample", type=int, default=40)
            p.add_argument("--seed", type=int, default=0)
        p.set_defaults(fn=execute, research_action=action)
