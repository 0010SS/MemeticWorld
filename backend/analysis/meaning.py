"""Meaning metrics on isolated checkpoint probes (OBSERVER ONLY; ontology v3 §5.5).

All scoring uses the contract ground truth (v3common.gt); no LLM judges correctness.
- ACC_cur(i,k,S): share of choices on item set S equal to GT(item, regime at k).
- OLD / NEW / OTHER (+ HEDGE = `slow`, reported separately): shares equal to GT(A), GT(B), neither (K1 items).
- ORR = OLD on K1c.  KNOW(k) = ½(NEW − OLD) on K1c, memory-only, pooled over the cell.
- SRI(c,k) = ½{[p_B(c) − p_B(NONCE)] − [p_A(c) − p_A(NONCE)]} from N choices pooled over members × framings.
- LAG(c) = [ΔSRI(c) − ΔKNOW] (C4→C6), shift arm minus keep_noshift (computed in outcomes_v3 across nodes).
- EXT(c,k,type): mean fit (fits 1, not sure .5, no 0) by item type; breadth(c,k): # situations with cell-mean
  fit ≥ .5 (applicability breadth).
- persistence: FORM (in-simulation uses of an expression per K1 job before vs after the change) and answer
  persistence (within-agent same-item agreement between two checkpoints).
- prior-adjusted: metric − the same persona's metric at C0.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from backend.analysis import v3common as V

K1C = ("K1c",)


def score(rows: list[dict], cfg: dict, ckpt) -> list[dict]:
    """Adds gt_cur (GT at the checkpoint's regime), gt_A, gt_B and cat (old/new/other) to P-type rows."""
    reg = V.regime_on_day(cfg, V.ckpt_day(ckpt)) if str(ckpt) != "C0" else "A"
    fx = V.regime_fixes(cfg)
    out = []
    for r in rows:
        r = dict(r)
        r["gt_cur"] = V.gt(r.get("type") or "", reg, cfg)
        r["gt_A"], r["gt_B"] = V.gt(r.get("type") or "", "A", cfg), V.gt(r.get("type") or "", "B", cfg)
        a = r.get("action")
        r["cat"] = None if a is None else ("old" if a == fx["A"] else "new" if a == fx["B"] else "other")
        out.append(r)
    return out


def _p(rows, form="P", types=None, agents=None):
    return [r for r in rows if r.get("form") == form and r.get("valid", True) and r.get("action") is not None
            and (types is None or r.get("type") in types) and (agents is None or r.get("agent") in agents)]


def acc_cur(rows, types=K1C, form="P", agents=None, items=None) -> float | None:
    rs = [r for r in _p(rows, form, types, agents) if r.get("gt_cur") and (items is None or r.get("item") in items)]
    return V.share(rs, lambda r: r["action"] == r["gt_cur"])


def three_way(rows, types=K1C, form="P", agents=None) -> dict:
    rs = _p(rows, form, types, agents)
    return {"old": V.share(rs, lambda r: r["cat"] == "old"), "new": V.share(rs, lambda r: r["cat"] == "new"),
            "other": V.share(rs, lambda r: r["cat"] == "other"),
            "hedge": V.share(rs, lambda r: r["action"] == V.HEDGE), "n": len(rs)}


def orr(rows, form="P", agents=None) -> float | None:
    return three_way(rows, K1C, form, agents)["old"]


def know(rows, agents=None) -> float | None:
    tw = three_way(rows, K1C, "P", agents)
    return None if tw["n"] == 0 else round(0.5 * (tw["new"] - tw["old"]), 4)


def sri(rows, cue: str, cfg: dict, agents=None) -> float | None:
    fx = V.regime_fixes(cfg)
    n = [r for r in rows if r.get("form") == "N" and r.get("action") is not None and r.get("valid", True)
         and (agents is None or r.get("agent") in agents)]
    c = [r for r in n if r.get("cue") == cue]
    z = [r for r in n if r.get("cue") == "NONCE"]
    if not c or not z:
        return None
    pb = lambda rs: sum(r["action"] == fx["B"] for r in rs) / len(rs)
    pa = lambda rs: sum(r["action"] == fx["A"] for r in rs) / len(rs)
    return round(0.5 * ((pb(c) - pb(z)) - (pa(c) - pa(z))), 4)


def ext(rows, cue: str, agents=None) -> dict:
    a = [r for r in rows if r.get("form") == "A" and r.get("cue") == cue and r.get("fit") is not None
         and (agents is None or r.get("agent") in agents)]
    by_type: dict[str, list] = {}
    by_item: dict[str, list] = {}
    for r in a:
        by_type.setdefault(r.get("type") or "?", []).append(r["fit"])
        by_item.setdefault(r.get("item") or "?", []).append(r["fit"])
    return {"ext": {t: V.mean(v) for t, v in sorted(by_type.items())},
            "breadth": sum(1 for v in by_item.values() if sum(v) / len(v) >= 0.5) if by_item else None,
            "n_situations": len(by_item)}


def answer_persistence(rows_a, rows_b, agents=None, types=None) -> float | None:
    """Within-agent agreement of memory-only choices on the same item (and order) at two checkpoints."""
    key = lambda r: (r["agent"], r.get("item"), r.get("order"))
    a = {key(r): r["action"] for r in _p(rows_a, "P", types, agents)}
    b = {key(r): r["action"] for r in _p(rows_b, "P", types, agents)}
    ks = sorted(set(a) & set(b), key=str)
    return V.share(ks, lambda k: a[k] == b[k])


def form_persistence(rd, expr: str, change_day: int | None, table: dict | None = None) -> dict:
    """In-simulation uses of `expr` (utterances + binder writes) per K1 job, before vs after change_day."""
    from backend.analysis import jobs as J
    table = table if table is not None else J.job_table(rd)
    tpd = V.ticks_per_day(rd)
    pat = re.compile(r"\b" + re.escape(expr.lower()) + r"\b")
    texts = [(u["tick"], u.get("text") or "") for u in rd.utterances]
    texts += [(w["tick"], w.get("text") or "") for w in rd.of("record_write") if w.get("text")]
    cut = (change_day - 1) * tpd if change_day else None
    k1 = [j for j in table.values() if j["class"] == "K1"]
    pre_j = sum(1 for j in k1 if cut is None or (j["day"] - 1) * tpd < cut)
    post_j = len(k1) - pre_j
    pre = sum(1 for t, x in texts if pat.search(x.lower()) and (cut is None or t < cut))
    post = sum(1 for t, x in texts if pat.search(x.lower()) and cut is not None and t >= cut)
    rpre = pre / pre_j if pre_j else None
    rpost = post / post_j if post_j else None
    return {"uses_pre": pre, "uses_post": post, "k1_pre": pre_j, "k1_post": post_j,
            "rate_pre": rpre and round(rpre, 4), "rate_post": rpost and round(rpost, 4),
            "use_ratio": round(rpost / rpre, 4) if rpre and rpost is not None else None}


def classify(use_ratio, d_sri, d_sri_noshift, d_sri_y, sri_c6, retest_sd=None, new_expr_sri=None) -> str:
    """§5.5 classification per cell. retest_sd None (test-retest stubbed) drops that criterion."""
    if use_ratio is not None and use_ratio == 0:
        return "LOSS"
    if (use_ratio is not None and use_ratio >= 0.5 and d_sri is not None and d_sri >= 0.4
            and (d_sri_noshift is None or d_sri > d_sri_noshift) and (d_sri_y is None or d_sri > d_sri_y)
            and (retest_sd is None or d_sri > 2 * retest_sd)):
        return "SEMANTIC_CHANGE"
    if new_expr_sri is not None and new_expr_sri > 0 and ((use_ratio is not None and use_ratio < 0.5)
                                                          or (sri_c6 is not None and sri_c6 < 0)):
        return "REPLACEMENT"
    if sri_c6 is not None and sri_c6 < 0:
        return "INERTIA"
    return "UNCLASSIFIED"


def prior_adjusted(value_by_agent: dict, c0_by_agent: dict) -> dict:
    return {a: V.sub(v, c0_by_agent.get(a)) for a, v in value_by_agent.items()}


def cell(rows: list[dict], cfg: dict, ckpt, cohorts: dict | None = None, targets: dict | None = None) -> dict:
    """All §5.5 memory metrics for one run node x checkpoint; per agent, cohort and item class."""
    rows = score(rows, cfg, ckpt)
    agents = sorted({r["agent"] for r in rows if r.get("agent")})
    cohorts = cohorts or {}
    groups: dict[str, list] = {}
    for a in agents:
        groups.setdefault(cohorts.get(a, "unknown"), []).append(a)
    types = sorted({r.get("type") for r in rows if r.get("type") and r.get("form") == "P"})
    sit_items = {r.get("item") for r in rows if r.get("form") == "P-sit"}
    abl_items = {r.get("item") for r in rows if r.get("form") == "P-abl"}
    per_agent = {a: {"acc_cur_k1c": acc_cur(rows, K1C, "P", [a]), "orr": orr(rows, "P", [a]),
                     "three_way": three_way(rows, K1C, "P", [a]),
                     "acc_by_type": {t: acc_cur(rows, (t,), "P", [a]) for t in types},
                     "acc_sit": acc_cur(rows, None, "P-sit", [a]), "acc_abl": acc_cur(rows, None, "P-abl", [a]),
                     "acc_p_on_abl_items": acc_cur(rows, None, "P", [a], abl_items) if abl_items else None}
                 for a in agents}
    p_on_sit = acc_cur(rows, None, "P", None, sit_items) if sit_items else None
    cues = sorted({r.get("cue") for r in rows if r.get("cue") and r.get("form") in ("N", "A")} | set((targets or {})))
    hit = [r["hit"] for r in rows if r.get("hit") is not None]
    return {
        "checkpoint": str(ckpt), "regime": V.regime_on_day(cfg, V.ckpt_day(ckpt)) if str(ckpt) != "C0" else None,
        "agents": agents, "per_agent": per_agent,
        "acc_cur_k1c": acc_cur(rows), "orr": orr(rows), "three_way": three_way(rows), "know": know(rows),
        "acc_by_type": {t: acc_cur(rows, (t,)) for t in types},
        "by_cohort": {c: {"acc_cur_k1c": acc_cur(rows, K1C, "P", m), "orr": orr(rows, "P", m),
                          "three_way": three_way(rows, K1C, "P", m), "n_agents": len(m)} for c, m in sorted(groups.items())},
        "sri": {c: sri(rows, c, cfg) for c in cues if c not in ("NONCE", "NONE")},
        "ext": {c: ext(rows, c) for c in cues if any(r.get("form") == "A" and r.get("cue") == c for r in rows)},
        "internalization": _ratio(p_on_sit, acc_cur(rows, None, "P-sit")),
        "situated_gap": V.sub(acc_cur(rows, None, "P-sit"), p_on_sit),
        "retrieval_hit_rate": round(sum(bool(h) for h in hit) / len(hit), 4) if hit else None,
        "n_responses": len(rows),
    }


def _ratio(a, b):
    return None if a is None or not b else round(a / b, 4)


def available_checkpoints(run_dir) -> list[str]:
    ks = set()
    for base in [Path(run_dir)] + ([V.parent_run(run_dir)] if V.parent_run(run_dir) else []):
        pr = base / "probes"
        if pr.exists():
            ks |= {p.parent.name for p in pr.glob("C*/responses.jsonl") if p.parent.name != "C0"}
    return sorted(ks, key=V.ckpt_day)


def analyze_meaning(run_dir, cohorts: dict | None = None, targets: dict | None = None, write: bool = True) -> dict:
    from backend.analysis.rundata import RunData
    rd = RunData(Path(run_dir))
    out = {"checkpoints": {}}
    for k in available_checkpoints(run_dir):
        out["checkpoints"][k] = cell(V.probe_responses(run_dir, k), rd.cfg, k, cohorts, targets)
    c0 = V.c0_responses(run_dir)
    if c0:
        out["C0"] = cell(c0, rd.cfg, "C0", None, targets)
    if write:
        json.dump(out, open(Path(run_dir) / "meaning.json", "w"), indent=1, sort_keys=True, default=str)
    return out
