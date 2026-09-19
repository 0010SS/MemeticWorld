"""Headless runs and analysis. From backend/:

    python -m app.cli run --days 2 --seed 7              # one run (full condition)
    python -m app.cli run --days 2 --seed 7 --control    # full run, then a control run with the same seed
    python -m app.cli list
    python -m app.cli memes 3                            # top detected memes for run 3
    python -m app.cli trace 3 "tray-gate"                # trace any phrase through run 3
    python -m app.cli compare 3 4                        # full vs control

Runs write to the same SQLite file as the API, so you can watch a CLI run in the UI.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from app.analysis import meme_detector, pipeline, propagation
from app.config import settings
from app.db import database as db
from app.simulation import runner
from app.simulation.world import SimConfig


def _progress(conn, run_id: int, stop: asyncio.Event):
    async def loop():
        while not stop.is_set():
            run = db.get_run(conn, run_id, with_config=False)
            if run:
                stats = run["stats"]
                print(f"\r  run {run_id}: tick {run['current_tick']}/{run['total_ticks']}  "
                      f"llm calls {stats.get('calls', 0)} (cached {stats.get('cache_hits', 0)}, "
                      f"failed {stats.get('failures', 0)})", end="", flush=True)
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
        print()
    return loop()


async def _run(args) -> None:
    conn = db.connect()
    conditions = ["full", "no_speech_memory"] if args.control else [args.condition]
    print(f"LLM: {settings.model_label} ({settings.llm_provider}), embeddings: {settings.embed_provider}")
    for condition in conditions:
        cfg = SimConfig(days=args.days, condition=condition, seed=args.seed)
        run_id = runner.create_run(conn, cfg, args.name)
        print(f"Run {run_id}: {condition}, {args.days} day(s), seed {args.seed}")
        stop = asyncio.Event()
        progress = asyncio.create_task(_progress(conn, run_id, stop))
        status = await runner.execute_run(conn, run_id)
        stop.set()
        await progress
        run = db.get_run(conn, run_id, with_config=False)
        print(f"  -> {status}" + (f": {run['error']}" if run.get("error") else ""))
        if status == "finished":
            _print_memes(conn, run_id, 10)


def _print_memes(conn, run_id: int, limit: int) -> None:
    corpus, memes, control = pipeline.analyze_run(conn, run_id)
    summary = meme_detector.summarize(memes)
    print(f"  {len(corpus.utterances)} utterances, {summary['n_memes']} candidate memes "
          f"({summary['n_strong']} strong), {summary['total_adopters']} adoptions "
          f"({summary['total_confirmed']} memory-confirmed)"
          + (f"; baselined against control run {control.run_id}" if control else "; no control run to baseline against"))
    for m in memes[:limit]:
        origin = corpus.names.get(m["originator"], m["originator"])
        ctrl = f"  control adopters {m['control']['n_adopters']}" if m.get("control") else ""
        print(f"  {m['tier']:10} {m['score']:6.2f}  {m['phrase']!r:28} users {m['n_users']}  "
              f"adopters {m['n_adopters']} (confirmed {m['n_confirmed']})  indep {len(m['independent'])}  "
              f"from {origin}{ctrl}" + (f"  [{m['referent']}]" if m.get("referent") else ""))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run a simulation")
    p.add_argument("--days", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--condition", choices=["full", "no_speech_memory"], default="full")
    p.add_argument("--control", action="store_true", help="run full, then a no_speech_memory control")
    p.add_argument("--name")
    sub.add_parser("list", help="list runs")
    p = sub.add_parser("memes", help="show detected memes")
    p.add_argument("run_id", type=int)
    p.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("trace", help="trace one phrase")
    p.add_argument("run_id", type=int)
    p.add_argument("phrase")
    p = sub.add_parser("compare", help="compare two runs")
    p.add_argument("a", type=int)
    p.add_argument("b", type=int)
    args = parser.parse_args()

    if args.cmd == "run":
        asyncio.run(_run(args))
        return
    conn = db.connect()
    if args.cmd == "list":
        for run in db.list_runs(conn):
            print(f"{run['id']:4}  {run['status']:11} {run['condition']:17} seed {run['seed']:<5} "
                  f"{run['current_tick']}/{run['total_ticks']} ticks  {run['model']}  {run['name']}")
    elif args.cmd == "memes":
        _print_memes(conn, args.run_id, args.limit)
    elif args.cmd == "trace":
        corpus = meme_detector.load_corpus(conn, args.run_id)
        stats = meme_detector.trace_phrase(corpus, " ".join(meme_detector.tokenize(args.phrase)))
        if not stats["n_uses"]:
            print("No uses (note: phrases made only of names/places/event words are not indexed).")
            sys.exit(1)
        for use in stats["uses"]:
            print(f"  {use['sim_time']:18} {corpus.names.get(use['speaker'], use['speaker']):8} "
                  f"{use['role']:11} {use['text']}")
        for edge in stats["edges"]:
            print(f"  {edge['source']} -> {edge['target']} at tick {edge['tick']} ({edge['evidence']})")
    elif args.cmd == "compare":
        result = propagation.compare(meme_detector.load_corpus(conn, args.a), meme_detector.load_corpus(conn, args.b))
        for side in ("a", "b"):
            r = result[side]
            print(f"run {r['run_id']} ({r['condition']}): {r['summary']}")
        for row in result["rows"]:
            print(f"  {row['phrase']!r:32} adopters {row['a']['n_adopters']:>2} vs {row['b']['n_adopters']:>2}   "
                  f"users {row['a']['n_users']:>2} vs {row['b']['n_users']:>2}")


if __name__ == "__main__":
    main()
