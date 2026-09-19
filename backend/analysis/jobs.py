"""Behavioural job metrics (OBSERVER ONLY; ontology v3 §5.5 "behavioural", §7.4 PO1 convergent, S1, S3).

- job_table: one row per job with its hidden truth (class/cause under the active regime), the first-attempt
  action and outcome. Truth comes from the hidden `job_truth` trace records when present, else from the
  regime-independent world-script job record (`hidden.class/cause` keyed by regime) plus the schedule.
- first-attempt accuracy: first attempt == GT(job class, active regime). Unparseable decisions (counted as
  `stop` by the agent layer) are excluded from accuracy (§4.3).
- CRN-matched comparison: the same job ids in two nodes of one seed (common world script and uniforms).
- OLD / NEW / OTHER of post-shift K1 first attempts, HEDGE, switch latency, defers, not delivered.
- follow rate: the first choice equals the fix recommended by the newest relevant binder entry in the view.
"""
from __future__ import annotations

from pathlib import Path

from backend.analysis import v3common as V
from backend.analysis.rundata import RunData


def _rd(x) -> RunData:
    return x if isinstance(x, RunData) else RunData(Path(x))


def _invalid(d: dict) -> bool:
    return d.get("valid") is False or bool(d.get("parse_error") or d.get("unparsed") or d.get("invalid"))


def _job_day(jid: str, rec: dict, tpd: int) -> int | None:
    if rec.get("day") is not None:
        return int(rec["day"])
    if rec.get("start_tick") is not None:
        return V.day_of_tick(rec["start_tick"], tpd)
    try:
        return int(str(jid).lstrip("j").split(".")[0])
    except ValueError:
        return None


def job_table(run) -> dict:
    rd = _rd(run)
    cfg, tpd = rd.cfg, V.ticks_per_day(rd)
    pack = V.pack_of(cfg)
    script = {r["id"]: r for r in V.load_jsonl(rd.dir / "world_script.jsonl") if r.get("kind") == "job" and r.get("id")}
    truth = {r.get("job"): r for r in rd.of("job_truth") if r.get("job")}
    starts = {r.get("job"): r for r in rd.of("job_start") if r.get("job")}
    ends = {r.get("job"): r for r in rd.of("job_end") if r.get("job")}
    attempts: dict[str, list] = {}
    for r in rd.of("job_attempt"):
        attempts.setdefault(r.get("job"), []).append(r)
    decisions: dict[str, list] = {}
    for r in rd.of("job_decision"):
        decisions.setdefault(r.get("job"), []).append(r)
    ids = set(script) | set(truth) | set(starts) | set(decisions)
    out = {}
    for jid in sorted(i for i in ids if i):
        s, t, st = script.get(jid, {}), truth.get(jid, {}), starts.get(jid, {})
        day = _job_day(jid, {**s, **st, **t}, tpd)
        regime = t.get("regime") or (V.regime_on_day(cfg, day) if day else "A")
        hid = s.get("hidden") or {}
        cls = t.get("class") or t.get("klass") or st.get("class")
        cause = t.get("cause")
        if cls is None:
            c = hid.get("class")
            cls = c.get(regime) if isinstance(c, dict) else c
        if cause is None:
            c = hid.get("cause")
            cause = c.get(regime) if isinstance(c, dict) else c
        if cls is None and hid.get("fault") is False:
            cls = "K0"
        ds = sorted((d for d in decisions.get(jid, []) if (d.get("action") in V.ACTIONS or _invalid(d))),
                    key=lambda d: (int(d.get("attempt") or 0), d.get("tick", 0)))
        first = ds[0] if ds else None
        gt = V.fix_of(cause, pack) if cause else V.gt(cls or "", regime, cfg)
        att = sorted(attempts.get(jid, []), key=lambda a: (int(a.get("attempt") or 0), a.get("tick", 0)))
        e = ends.get(jid, {})
        delivered = e.get("delivered")
        if delivered is None and att:
            delivered = any(a.get("success") for a in att)
        out[jid] = {
            "job": jid, "day": day, "shift": s.get("shift") or st.get("shift"), "regime": regime, "class": cls,
            "cause": cause, "gt": gt, "faulted": cls not in (None, "K0"),
            "operator": (first or {}).get("agent") or s.get("operator") or st.get("operator"),
            "first_action": (first or {}).get("action"), "first_valid": bool(first) and not _invalid(first),
            "first_tick": (first or {}).get("tick"), "first_decision": first,
            "first_success": (att[0].get("success") if att else None),
            "first_correct": (first.get("action") == gt) if first and not _invalid(first) and gt else None,
            "attempts": [{k: a.get(k) for k in ("attempt", "action", "success", "agent", "tick")} for a in att],
            "delivered": delivered, "deferred": bool(first) and first.get("action") == "stop",
        }
    return out


def _sel(table: dict, days=None, classes=("K1", "K2")) -> list[dict]:
    return [j for j in table.values() if j["faulted"] and (classes is None or j["class"] in classes)
            and (days is None or j["day"] in days)]


def first_attempt_accuracy(table: dict, days=None, classes=("K1", "K2")) -> dict:
    js = [j for j in _sel(table, days, classes) if j["first_valid"]]
    return {"acc": V.share(js, lambda j: j["first_correct"]), "n": len(js)}


def matched_first_attempts(table_a: dict, table_b: dict, days=None, classes=("K1", "K2")) -> dict:
    """CRN-matched: jobs present, faulted and validly decided in both nodes (same job id = same job, class,
    uniforms). Returns acc in each node and the paired difference a - b."""
    a = {j["job"]: j for j in _sel(table_a, days, classes) if j["first_valid"]}
    b = {j["job"]: j for j in _sel(table_b, days, classes) if j["first_valid"]}
    ids = sorted(set(a) & set(b))
    acc_a = V.share(ids, lambda i: a[i]["first_correct"])
    acc_b = V.share(ids, lambda i: b[i]["first_correct"])
    return {"n": len(ids), "jobs": ids, "acc_a": acc_a, "acc_b": acc_b, "delta": V.sub(acc_a, acc_b)}


def old_new_other(table: dict, cfg: dict, days=None) -> dict:
    """Post-change three-way outcome of K1 first attempts: old = GT(A), new = GT(B), other = anything else
    (HEDGE = the `slow` share, reported separately and inside other)."""
    fx = V.regime_fixes(cfg)
    js = [j for j in _sel(table, days, ("K1",)) if j["first_valid"]]
    return {"old": V.share(js, lambda j: j["first_action"] == fx["A"]),
            "new": V.share(js, lambda j: j["first_action"] == fx["B"]),
            "other": V.share(js, lambda j: j["first_action"] not in (fx["A"], fx["B"])),
            "hedge": V.share(js, lambda j: j["first_action"] == V.HEDGE), "n": len(js)}


def switch_latency(table: dict, cfg: dict, from_day: int | None = None) -> dict:
    """Post-shift K1 jobs (in order) until the first window where 2 of 3 consecutive first attempts are the
    B-fix. latency = 1-based index of the last job in that window; None (censored) if it never happens."""
    from_day = from_day or V.shift_day(cfg)
    if from_day is None:
        return {"latency": None, "censored": True, "n": 0, "applicable": False}
    bfix = V.regime_fixes(cfg)["B"]
    js = sorted((j for j in _sel(table, None, ("K1",)) if j["day"] >= from_day and j["first_valid"]),
                key=lambda j: (j["first_tick"] or 0, j["job"]))
    hits = [j["first_action"] == bfix for j in js]
    for i in range(2, len(hits)):
        if sum(hits[i - 2:i + 1]) >= 2:
            return {"latency": i + 1, "censored": False, "n": len(js), "applicable": True}
    if len(hits) == 2 and all(hits):
        return {"latency": 2, "censored": False, "n": 2, "applicable": True}
    return {"latency": None, "censored": True, "n": len(js), "applicable": True}


def delivery(table: dict, days=None) -> dict:
    js = [j for j in table.values() if days is None or j["day"] in days]
    return {"jobs": len(js), "not_delivered": sum(1 for j in js if j["delivered"] is False),
            "defers": sum(1 for j in js if j["deferred"])}


def follow_rate(run, table: dict | None = None, days=None) -> dict:
    """First choice == the fix recommended by the newest relevant entry in the decision's view. Relevant: an
    entry whose single recommended action is a fix for this job's class family (K1: A/B fix; K2: belt;
    K3: B fix). Ambiguous entries (several actions) are skipped (coder fallback stubbed)."""
    from backend.analysis.record_lineage import binder_index
    rd = _rd(run)
    table = table or job_table(rd)
    idx = binder_index(rd)
    fx = V.regime_fixes(rd.cfg)
    belt = V.fix_of("BELT", V.pack_of(rd.cfg))
    fam = {"K1": {fx["A"], fx["B"]}, "K2": {belt}, "K3": {fx["B"]}}
    n = followed = 0
    for j in table.values():
        d = j.get("first_decision")
        if not d or not j["first_valid"] or not j["faulted"] or (days is not None and j["day"] not in days):
            continue
        best = None
        for eid in d.get("binder_entry_ids") or []:
            e = idx.get(eid)
            if not e:
                continue
            act, _amb = V.recommended_action(e.get("text"))
            if act in fam.get(j["class"], set()) and (best is None or e["tick"] > best[0]):
                best = (e["tick"], act)
        if best:
            n += 1
            followed += int(j["first_action"] == best[1])
    return {"follow_rate": round(followed / n, 4) if n else None, "n": n}


def analyze_jobs(run) -> dict:
    rd = _rd(run)
    t = job_table(rd)
    sd = V.shift_day(rd.cfg)
    days = sorted({j["day"] for j in t.values() if j["day"]})
    post = [d for d in days if sd and d >= sd] or None
    return {"by_day": {d: first_attempt_accuracy(t, [d]) for d in days},
            "by_class": {c: first_attempt_accuracy(t, None, (c,)) for c in ("K1", "K2", "K3")},
            "old_new_other_post": old_new_other(t, rd.cfg, post) if post else None,
            "switch_latency": switch_latency(t, rd.cfg, sd), "delivery": delivery(t),
            "follow_rate": follow_rate(rd, t)}
