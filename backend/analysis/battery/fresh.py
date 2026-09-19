"""C0 fresh-agent prior baseline (ONTOLOGY_V3 §5.6). OBSERVER ONLY.

Every persona of the universe (founders and reserves) with EMPTY memory answers P (all items, 2 menu orders on
K1c) and N (W, NONCE, NONE; X and Y once targets exist). A persona-free responder ("a student who volunteers at a
campus makerspace co-op") answers P in 3 menu orders. Gives the pretrained action distribution per item, the prior
reading of each cue, prior production of X (targets.py) and cue diagnosticity (CUE items).

Writes `<out_root>/C0/{responses.jsonl, llm_calls.jsonl, meta.json}` with scope `fresh:C0:...`; no run is read
except its config (population, seed, probe settings). Run per content pack and mapping at calibration.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path

from backend import ga_compat
from backend.analysis.battery import gt as GT
from backend.analysis.battery.items import load_items
from backend.analysis.battery.runner import (PERSONA_FREE, PERSONA_FREE_ISS, Responder, Session, cue_table,
                                             run_items)

CKPT = "C0"


def fresh_profiles(cfg: dict) -> dict:
    from backend.agents.profile import load_population
    try:
        res = load_population(cfg["population"], None, include_reserves=True)
    except TypeError:
        res = load_population(cfg["population"], None)
    return res[0] if isinstance(res, tuple) else res


def run_fresh(cfg: dict, out_root, *, mapping: str | None = None, targets: dict | None = None, llm=None,
              backend=None, profiles: dict | None = None, persona_free_orders: int = 3,
              cues=("W", "NONCE", "NONE"), start: str = "2026-09-17T22:15:00") -> dict:
    """C0 prior. `cues` default W/NONCE/NONE; pass ("X", "Y") with `targets` after target selection."""
    from backend.agents.agent import Agent
    from backend.llm.embeddings import make_embedder
    from backend.modules.base import ModuleStack
    out_dir = Path(out_root) / CKPT
    out_dir.mkdir(parents=True, exist_ok=True)
    own = llm is None
    if own:
        from backend.llm.client import observer_client
        llm = observer_client("probe", out_dir / "llm_calls.jsonl", cfg, backend=backend)
    ga_compat.load(None, make_embedder(cfg.get("embedding")))
    pack = (cfg.get("workshop") or {}).get("content", "laser_alpha")
    mapping = mapping or GT.mapping_for(cfg)
    items = load_items(pack, mapping)
    when = dt.datetime.fromisoformat(start)
    pc = cfg.get("probe") or {}
    sess = Session(llm=llm, seed=int(cfg.get("seed", 0)), ckpt=CKPT, prefix="fresh", weekday=when.strftime("%A"),
                   k=int(pc.get("k", 6)), pack=pack, mapping=mapping, regime="A", raw_meta={})
    tbl = cue_table(targets or {"W": "F4"}, cfg.get("seed", 0))
    framings = list(pc.get("framings") or ["note", "text", "overheard"])
    profiles = profiles if profiles is not None else fresh_profiles(cfg)
    acfg = copy.deepcopy(cfg)
    recs = []
    for aid in sorted(profiles):
        a = Agent(profiles[aid], acfg, ModuleStack([]), cfg.get("seed", 0))     # empty memory
        a.set_time(when)
        a.state.activity = "thinking"
        a.sync_scratch()
        r = Responder(aid, a.iss(), a.profile.first_name, None)
        recs += run_items(sess, r, items, "P", False)
        for ck in cues:
            if ck != "NONE" and not tbl.get(ck):
                continue
            for fr in framings:
                recs.append(sess.note(r, ck, tbl.get(ck), fr))
    pf = Responder("persona_free", PERSONA_FREE_ISS, PERSONA_FREE)
    for it in items:
        for order in range(persona_free_orders):
            recs.append(sess.act(pf, it, "P", order, False))
    for rec in recs:
        rec.update({"ckpt": CKPT, "regime": None, "mapping": mapping, "pack": pack})
    recs.sort(key=lambda x: x["scope"])
    with open(out_dir / "responses.jsonl", "w") as fh:
        for rec in recs:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    summary = {"ckpt": CKPT, "pack": pack, "mapping": mapping, "n_calls": len(recs),
               "n_errors": sum(1 for x in recs if x.get("error")), "agents": sorted(profiles),
               "targets": tbl}
    (out_dir / "meta.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    if own:
        llm.close()
    return summary


def prior_production(responses: list[dict], expr: str) -> float:
    """Share of C0 P descriptions containing `expr` (targets.py: must be < 10%)."""
    ds = [str(r.get("describe") or "").lower() for r in responses if r.get("form") == "P" and r.get("describe")]
    return sum(expr.lower() in d for d in ds) / len(ds) if ds else 0.0
