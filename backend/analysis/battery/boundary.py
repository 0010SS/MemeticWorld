"""Scoring the graded battery, and putting the four memes side by side (6b/6c). OBSERVER ONLY.

THE HEADLINE NUMBER IS THE BOUNDARY. For one meme, one checkpoint and one cohort, the gradient runs
literal -> near -> mid -> far; the boundary is how far out "applies" is still said by at least half the
cohort. It is reported as an index so its movement is a single legible number:

    0  not even the literal case          the phrase means nothing here any more
    1  literal only                       still used, still only about its origin  -> calcified
    2  through near
    3  through mid
    4  through far                        applies to things well beyond its origin -> broadened

Movement across checkpoints is `d_boundary`; movement across the repair is the survival question.

EVERY APPLIES-RATE IS REPORTED BESIDE ITS FOIL FALSE-POSITIVE RATE, always. A model drifting toward
agreeing with everything produces exactly the same applies-rates as genuine broadening; the only thing that
separates them is whether it also says "fits" to situations that share the surface and not the structure.
A cohort whose foil rate is at or above `FOIL_MAX` is marked invalid and its boundary is reported as None:
that cohort's result is not evidence, and we say so rather than quietly averaging it in.

Explanation types come from the pluggable judge (analysis.judge), not from a new path and not from a
keyword list. A mock verdict is a placeholder and never counts, as everywhere else in this codebase. The
judge's `function` vocabulary is description | warning | joke | instruction | label | greeting | other,
which covers most of the pragmatic-change sequence the memo names (description -> warning -> excuse ->
joke) but NOT "excuse": an excuse currently lands in "other", so a shift into excusing is visible as a
rise in "other" rather than named. Adding it is a one-word change to judge.FUNCTIONS and its prompt.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.analysis.battery import graded as G
from backend.analysis.battery import registry as REG

LEVELS = REG.LEVELS
FOIL_MAX = 0.33            # a third of the foils called "fits" is an agreeable model, not a broader meme
APPLIES = 0.5              # "applies" holds at a level when at least half the cohort says it fits
MIN_N = 3                  # fewer answers than this at a level says nothing about a boundary


def _share(rows, pred) -> float | None:
    rs = [r for r in rows if r.get("valid")]
    return round(sum(1 for r in rs if pred(r)) / len(rs), 4) if rs else None


def _mean(rows, key="fit") -> float | None:
    vs = [r[key] for r in rows if r.get("valid") and r.get(key) is not None]
    return round(sum(vs) / len(vs), 4) if vs else None


def level_rates(rows: list[dict]) -> dict:
    """applies / fit-mean / n per gradient level, plus the foil false-positive rate."""
    out = {}
    for lvl in LEVELS:
        rs = [r for r in rows if r["form"] == "G" and r.get("kind") == "gradient" and r.get("level") == lvl]
        out[lvl] = {"applies": _share(rs, lambda r: r["answer"] == "fits"), "fit_mean": _mean(rs),
                    "n": sum(1 for r in rs if r.get("valid")), "n_asked": len(rs)}
    foils = [r for r in rows if r["form"] == "G" and r.get("kind") == "foil"]
    out["foil"] = {"false_positive": _share(foils, lambda r: r["answer"] == "fits"), "fit_mean": _mean(foils),
                   "n": sum(1 for r in foils if r.get("valid")), "n_asked": len(foils)}
    return out


def boundary_index(rates: dict) -> int | None:
    """How many levels, outward from `literal`, still hold at >= APPLIES. None when the literal level has
    too few valid answers to say anything (an empty cohort is not a boundary of 0)."""
    lit = rates.get("literal") or {}
    if (lit.get("n") or 0) < MIN_N:
        return None
    idx = 0
    for lvl in LEVELS:
        r = rates.get(lvl) or {}
        if (r.get("n") or 0) < MIN_N or r.get("applies") is None or r["applies"] < APPLIES:
            break
        idx += 1
    return idx


def cohort_cell(rows: list[dict], functions: dict | None = None) -> dict:
    """One meme x checkpoint x cohort: the rates, the boundary, the foil verdict and the explanation mix."""
    rates = level_rates(rows)
    fpr = (rates["foil"] or {}).get("false_positive")
    invalid = fpr is not None and fpr >= FOIL_MAX
    bidx = boundary_index(rates)
    ex = [r for r in rows if r["form"] == "E"]
    mix: dict[str, int] = {}
    for r in ex:
        f = (functions or {}).get(r.get("scope"))
        if f:
            mix[f] = mix.get(f, 0) + 1
    return {"levels": rates, "foil_false_positive": fpr,
            "boundary_index": None if invalid else bidx, "boundary_raw": bidx,
            "boundary_level": None if not bidx else LEVELS[bidx - 1],
            "foil_invalid": invalid, "n_agents": len({r["agent"] for r in rows}),
            "explanations": {"n": len(ex), "n_valid": sum(1 for r in ex if r.get("valid")),
                             "heard_before": {k: _share(ex, lambda r, k=k: str(r.get("heard_before") or "").lower().startswith(k))
                                              for k in ("yes", "no", "not")},
                             "function_mix": dict(sorted(mix.items())),
                             "function_provenance": (functions or {}).get("_provenance")}}


def checkpoint_cells(rows: list[dict], specs: list[REG.MemeSpec], functions: dict | None = None) -> dict:
    """{meme: {"all": cell, "by_cohort": {cohort: cell}, "by_site": {site: cell}}} for one checkpoint."""
    out = {}
    for s in specs:
        mr = [r for r in rows if r.get("meme") == s.id]
        if not mr:
            continue
        cohorts = sorted({r.get("cohort") for r in mr if r.get("cohort")})
        sites = sorted({r.get("site") for r in mr if r.get("site")})
        out[s.id] = {
            "cell": s.cell, "phrase": s.phrase,
            "all": cohort_cell(mr, functions),
            "by_cohort": {c: cohort_cell([r for r in mr if r.get("cohort") == c], functions) for c in cohorts},
            # 6b: the unexposed hall on its own - two groups may infer different things from one phrase
            "by_site": {t: cohort_cell([r for r in mr if r.get("site") == t], functions) for t in sites},
        }
    return out


# ------------------------------------------------------------------------------------- explanation types
def classify_explanations(run_dir, ckpt: str, *, judge=None, specs: list[REG.MemeSpec] | None = None,
                          max_calls: int | None = None, probes_root=None, write: bool = True) -> dict:
    """Pragmatic function per explanation, via the pluggable judge. Identical explanations are judged once
    (a run where everyone says the same thing should not cost N calls), and the result carries its own
    provenance so a placeholder verdict can never be read as a finding."""
    from backend.analysis import judge as J
    from backend.analysis.rundata import RunData
    run_dir = Path(run_dir)
    rd = RunData(run_dir)
    specs = specs if specs is not None else REG.load_registry(rd.cfg, only_enabled=False)
    by_id = REG.by_id(specs)
    rows = [r for r in G.read_responses(run_dir, ckpt, probes_root) if r["form"] == "E" and r.get("valid")]
    own = judge is None
    judge = judge if judge is not None else J.get_judge(rd.cfg, run_dir=run_dir)
    bcfg = REG.battery_cfg(rd.cfg)
    cap = int(max_calls if max_calls is not None else bcfg["max_judge_calls"])
    seen: dict[tuple, dict] = {}
    out: dict[str, str] = {}
    detail = []
    try:
        for r in sorted(rows, key=lambda r: r["scope"]):
            s = by_id.get(r["meme"])
            text = f"{r.get('meaning') or ''} {r.get('use') or ''}".strip()
            key = (r["meme"], text.lower())
            v = seen.get(key)
            if v is None:
                if len(seen) >= cap:
                    break
                # one explanation by one copy: say so, rather than letting the prompt's "used N times by
                # several speakers" line imply a community usage the judge is not being shown
                expr = {"id": r["meme"], "phrase": (s.phrase if s else r["meme"]),
                        "variants": [s.phrase] if s else [], "n_uses": 1, "n_speakers": 1}
                v = judge.judge(expr, [text], {"status": "probe_explanation"})
                seen[key] = v
            out[r["scope"]] = v.get("function")
            detail.append({"scope": r["scope"], "meme": r["meme"], "agent": r["agent"],
                           "cohort": r.get("cohort"), "site": r.get("site"), "function": v.get("function"),
                           "gloss": v.get("gloss"), "text": text[:400]})
    finally:
        if own:
            judge.close()
    desc = judge.describe()
    real = J.is_real_verdict(desc)
    out["_provenance"] = f"{desc.get('provider')}:{desc.get('model')}" + ("" if real else " (PLACEHOLDER)")
    doc = {"ckpt": ckpt, "judge": desc, "real_judge": real, "placeholder": not real,
           "n_explanations": len(rows), "n_judge_calls": len(seen), "capped": len(seen) >= cap,
           "functions": {k: v for k, v in out.items() if k != "_provenance"},
           "provenance": out["_provenance"], "detail": detail}
    if write:
        root = (Path(probes_root) if probes_root else run_dir / "probes") / "graded" / ckpt
        root.mkdir(parents=True, exist_ok=True)
        (root / "explanations.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    return doc


def load_functions(run_dir, ckpt: str, probes_root=None) -> dict:
    p = (Path(probes_root) if probes_root else Path(run_dir) / "probes") / "graded" / ckpt / "explanations.json"
    if not p.exists():
        return {}
    doc = json.loads(p.read_text())
    # a placeholder judge's labels are recorded but must never be read as an explanation mix
    if not doc.get("real_judge"):
        return {"_provenance": doc.get("provenance")}
    return dict(doc.get("functions") or {}, _provenance=doc.get("provenance"))


# -------------------------------------------------------------------------------------- the comparison
def _usage_summary(rd, spec: REG.MemeSpec) -> dict:
    """In-simulation use, split at the meme's own repair day. The split itself is trends'
    `survival_after_repair`, so the figure in this document and the figure in the adoption panel are the
    same number rather than two definitions that can drift apart."""
    from backend.analysis.trends import survival_after_repair
    us = REG.find_usages(rd, spec)
    tpd = rd.ticks_per_day
    seeds = REG.seed_agents(rd, spec)
    last = max((u["tick"] for u in rd.utterances), default=0)
    sv = survival_after_repair(us, spec.repaired_day, tpd, seeds, last_tick=last)
    return {"uses": len(us), "unique_users": len({u["speaker"] for u in us}),
            "unique_adopters": len({u["speaker"] for u in us} - set(seeds)),
            "variants": sorted({u["variant"] for u in us}), "first_tick": min((u["tick"] for u in us), default=None),
            "last_tick": max((u["tick"] for u in us), default=None),
            "last_day": max((u["tick"] // tpd + 1 for u in us), default=None),
            "repaired_day": spec.repaired_day, "uses_before_repair": sv["uses_before"],
            "uses_after_repair": sv["uses_after"], "users_after_repair": sv["users_after"],
            "survival_ratio": sv["survival_ratio"], "survival": sv}


def outcome(usage: dict, boundaries: dict) -> str:
    """dies / calcifies / broadens, the three outcomes the design must be able to tell apart.

    `boundaries` is {checkpoint: index} in checkpoint order. UNDETERMINED is returned rather than a guess
    whenever the foil rate invalidated the boundary or the meme was never said at all."""
    vals = [v for v in boundaries.values() if v is not None]
    if not usage["uses"]:
        return "never_used"
    if not vals:
        return "undetermined"
    first, last = vals[0], vals[-1]
    dead_use = usage["repaired_day"] is not None and usage["uses_after_repair"] == 0
    if last == 0 or (dead_use and last <= 1):
        return "died"
    if last > first:
        return "broadened"
    if last <= 1:
        return "calcified"
    return "broad_from_the_start" if first >= 2 and last >= first else "narrowed"


def compare(run_dir, *, checkpoints: list[str] | None = None, probes_root=None, write: bool = True) -> dict:
    """The comparative output: four memes side by side, per checkpoint and cohort, so "which survived and
    why" reads off the result instead of being reconstructed by hand."""
    from backend.analysis.rundata import RunData
    run_dir = Path(run_dir)
    rd = RunData(run_dir)
    specs = REG.load_registry(rd.cfg, only_enabled=False)
    cks = checkpoints if checkpoints is not None else G.available_checkpoints(run_dir, probes_root)
    per_ck = {}
    for ck in cks:
        rows = G.read_responses(run_dir, ck, probes_root)
        if rows:
            per_ck[ck] = checkpoint_cells(rows, specs, load_functions(run_dir, ck, probes_root))
    memes = {}
    for s in specs:
        usage = _usage_summary(rd, s)
        bnd = {ck: ((per_ck[ck].get(s.id) or {}).get("all") or {}).get("boundary_index") for ck in per_ck}
        memes[s.id] = {
            "id": s.id, "phrase": s.phrase, "cell": s.cell, "grounding": s.grounding, "breadth": s.breadth,
            "repaired_day": s.repaired_day, "usage": usage, "boundary": bnd,
            "d_boundary": _delta(bnd), "outcome": outcome(usage, bnd),
            "foil_false_positive": {ck: ((per_ck[ck].get(s.id) or {}).get("all") or {}).get("foil_false_positive")
                                    for ck in per_ck},
            "by_checkpoint": {ck: per_ck[ck].get(s.id) for ck in per_ck},
            "seed_graph": REG.seed_graph_stats(rd, s),
        }
    doc = {"run": str(run_dir), "checkpoints": list(per_ck),
           "design": {"cells": REG.cells(specs), "registry_errors": REG.validate(specs),
                      "foil_max": FOIL_MAX, "applies_threshold": APPLIES, "min_n": MIN_N},
           "memes": memes, "table": table_rows(memes, list(per_ck))}
    if write:
        (run_dir / "memes.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    return doc


def _delta(bnd: dict):
    vals = [v for v in bnd.values() if v is not None]
    return (vals[-1] - vals[0]) if len(vals) >= 2 else None


def table_rows(memes: dict, cks: list[str]) -> list[dict]:
    """One row per meme: the 2x2 factors, the boundary at each checkpoint, the foil rate, the outcome."""
    rows = []
    for m in sorted(memes.values(), key=lambda m: (m["grounding"], m["breadth"], m["id"])):
        rows.append({"meme": m["id"], "phrase": m["phrase"], "grounding": m["grounding"],
                     "breadth": m["breadth"], "uses": m["usage"]["uses"], "users": m["usage"]["unique_users"],
                     "survival_ratio": m["usage"]["survival_ratio"],
                     "boundary": [m["boundary"].get(c) for c in cks],
                     "foil_fpr": [m["foil_false_positive"].get(c) for c in cks],
                     "d_boundary": m["d_boundary"], "outcome": m["outcome"]})
    return rows


def render_table(doc: dict) -> str:
    """The side-by-side table as text. The boundary column is the headline; the foil column is the licence
    to believe it."""
    cks = doc["checkpoints"]
    w = max([len(r["meme"]) for r in doc["table"]] + [4])
    head = (f"{'meme':<{w}}  {'grounding':<11} {'breadth':<8} {'uses':>5} {'users':>5} {'surv':>6}  "
            f"{'boundary ' + '/'.join(cks):<28} {'foil_fpr':<22} {'outcome':<12}")
    lines = [head, "-" * len(head)]
    for r in doc["table"]:
        b = "/".join("-" if v is None else str(v) for v in r["boundary"])
        f = "/".join("-" if v is None else f"{v:.2f}" for v in r["foil_fpr"])
        s = "-" if r["survival_ratio"] is None else f"{r['survival_ratio']:.2f}"
        lines.append(f"{r['meme']:<{w}}  {r['grounding']:<11} {r['breadth']:<8} {r['uses']:>5} "
                     f"{r['users']:>5} {s:>6}  {b:<28} {f:<22} {r['outcome']:<12}")
    bad = [f"{r['meme']}={v:.2f}" for r in doc["table"] for v in r["foil_fpr"] if v is not None and v >= FOIL_MAX]
    if bad:
        lines.append(f"INVALID (foil false-positive >= {FOIL_MAX}): {', '.join(bad)} - these boundaries are "
                     "not evidence of broadening")
    errs = (doc.get("design") or {}).get("registry_errors") or []
    lines += [f"REGISTRY: {e}" for e in errs]
    return "\n".join(lines)
