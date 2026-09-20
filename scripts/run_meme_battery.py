#!/usr/bin/env python
"""Run the graded meme battery over a run's checkpoints and print the four memes side by side.

    # what it would cost, without spending anything
    .venv/bin/python scripts/run_meme_battery.py runs/<id> --dry-run

    # probe, judge the explanations, write memes.json + meme_analysis.json, print the table
    .venv/bin/python scripts/run_meme_battery.py runs/<id> --checkpoints C2,C4,C6,final

    # the table again from what is already on disk (no LLM calls)
    .venv/bin/python scripts/run_meme_battery.py runs/<id> --report

COST IS THE BINDING CONSTRAINT. N memes x 4 gradient levels x foils x agents x checkpoints explodes: four
memes, 100 agents and four checkpoints is 3,200 probe calls before a single explanation is judged. Two
things keep it affordable and neither costs a measurement:
  - ONE PANEL answers every meme (agreeableness becomes a within-subject constant rather than noise);
  - the gradient and its foils are ONE call, not one per situation, exactly as the v3 A-form does.
That is `memes.battery.sample.max_agents` x memes x 2 calls per checkpoint. --dry-run prints the number.

Nothing here writes into the run except `probes/graded/`, `memes.json` and `meme_analysis.json`; the
isolation protocol (checksums before and after, copies discarded) is battery/runner.py's.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.analysis.battery import boundary as B  # noqa: E402
from backend.analysis.battery import graded as G  # noqa: E402
from backend.analysis.battery import registry as REG  # noqa: E402


def _checkpoints(run_dir: Path, arg: str | None, bcfg: dict) -> list[str]:
    if arg:
        return [c.strip() for c in arg.split(",") if c.strip()]
    if bcfg["checkpoints"]:
        return list(bcfg["checkpoints"])
    on_disk = sorted((p.name for p in (run_dir / "checkpoints").glob("C*") if p.is_dir()),
                     key=lambda n: int(n[1:]) if n[1:].isdigit() else 10 ** 9)
    return on_disk + ["final"] if (run_dir / "agents_final").is_dir() else on_disk


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--checkpoints", default=None, help="comma-separated (default: memes.battery.checkpoints, "
                                                        "else every checkpoint on disk plus 'final')")
    ap.add_argument("--sample", type=int, default=None, help="override memes.battery.sample.max_agents")
    ap.add_argument("--per-cohort", type=int, default=None, help="override memes.battery.sample.per_cohort")
    ap.add_argument("--backend", default=None, help="override probe.backend (e.g. mock; mock never counts)")
    ap.add_argument("--dry-run", action="store_true", help="project the cost and validate the registry only")
    ap.add_argument("--report", action="store_true", help="re-read what is on disk; make no LLM calls")
    ap.add_argument("--no-judge", action="store_true", help="skip explanation classification")
    ap.add_argument("--topic", action="store_true", help="also compute the topic-drift control (embeddings)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the comparison document instead of the table")
    a = ap.parse_args(argv)

    from backend.analysis.rundata import RunData
    run_dir = Path(a.run)
    rd = RunData(run_dir)
    specs = REG.load_registry(rd.cfg, only_enabled=False)
    if not specs:
        print(f"{run_dir}: no memes.registry in config.resolved.yaml - nothing to probe", file=sys.stderr)
        return 2
    bcfg = REG.battery_cfg(rd.cfg)
    if a.sample is not None:
        bcfg["sample"]["max_agents"] = int(a.sample)
    if a.per_cohort is not None:
        bcfg["sample"]["per_cohort"] = int(a.per_cohort)
    if a.no_judge:
        bcfg["judge_explanations"] = False
    cks = _checkpoints(run_dir, a.checkpoints, bcfg)

    errs = REG.validate(specs)
    for e in errs:
        print(f"REGISTRY: {e}", file=sys.stderr)
    contam = REG.world_contamination(specs, rd.world_text)
    for mid, c in contam.items():
        if c["phrase"] or c["words"]:
            print(f"CONTAMINATION {mid}: the world emitted the phrase {c['phrase']}x and its distinctive "
                  f"words {c['words']} - transmission is not separable from rediscovery (R1/R2)", file=sys.stderr)

    cost = G.project_cost(rd, specs, cks, bcfg)
    print(json.dumps({"checkpoints": cks, "cost": cost}, indent=1))
    if a.dry_run:
        return 1 if errs else 0

    if not a.report:
        backend = None
        if a.backend:
            from backend.llm.client import make_backend
            backend = make_backend({"backend": a.backend, "model": "haiku"})
        for ck in cks:
            try:
                s = G.run_graded(run_dir, ck, backend=backend, force=a.force, specs=specs, bcfg=bcfg)
            except FileNotFoundError as e:
                print(f"skip {ck}: {e}", file=sys.stderr)
                continue
            print(f"{ck}: calls={s['n_calls']} errors={s['n_errors']} valid={s['valid']} "
                  f"panel={len(s['panel'])}{' (cached)' if s.get('cached') else ''}")
            if bcfg["judge_explanations"] and bcfg["explain"]:
                d = B.classify_explanations(run_dir, ck, specs=specs)
                print(f"{ck}: explanations judged={d['n_judge_calls']} by {d['provenance']}")

    from backend.analysis.meaning import analyze_memes
    doc = analyze_memes(run_dir, checkpoints=[c for c in cks if c in G.available_checkpoints(run_dir)],
                        topic=a.topic)
    if not doc:
        print("no graded responses on disk yet", file=sys.stderr)
        return 2
    print(json.dumps(doc, indent=1, sort_keys=True, default=str) if a.json else B.render_table(doc))
    print(f"\nwrote {run_dir / 'memes.json'} and {run_dir / 'meme_analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
