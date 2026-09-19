"""Binder artifact lineage and record metrics (OBSERVER ONLY; ontology v3 §5.8). Extends v2 lineage.py
without editing it.

Lineage graph: entry/revision -> reads (agent, tick) -> record memory nodes (MemoryMeta.record_ids) ->
utterances (v2 provenance: retrieved nodes of the speaker) -> job decisions (binder_entry_ids + retrieved).

Inputs (trace): record_write {agent, tick, offer, choice, entry_id|rev_id, text}, record_read {agent, tick,
context, entry_ids, rev_ids, new_ids, view_sha}, record_transition {day, mode, archived_binder_id},
memory_encoded {node_id, agent, source_type, record_ids?}, job_decision {agent, job, binder_entry_ids,
retrieved}, roster_change {day, agent, kind}. A checkpoint binder.json (latest C*) fills in entries the trace
lacks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from backend.analysis import v3common as V
from backend.analysis.rundata import RunData


def _rd(x) -> RunData:
    return x if isinstance(x, RunData) else RunData(Path(x))


def _ids(x) -> list:
    if not x:
        return []
    out = []
    for i in x:
        out.append(i.get("id") or i.get("node_id") if isinstance(i, dict) else i)
    return [i for i in out if i]


def binder_index(run) -> dict:
    """id -> {kind: entry|rev, author, tick, text, context, job_id}."""
    rd = _rd(run)
    out = {}
    ck = sorted((rd.dir / "checkpoints").glob("C*/binder.json"), key=lambda p: V.ckpt_day(p.parent.name)) \
        if (rd.dir / "checkpoints").exists() else []
    if ck:
        b = V.load_json(ck[-1], {}) or {}

        def walk(bd):
            for r in bd.get("front") or []:
                out[r.get("rev_id")] = {"kind": "rev", "author": r.get("author"), "tick": r.get("tick"),
                                        "text": r.get("text"), "context": "front", "job_id": None}
            for e in bd.get("log") or []:
                out[e.get("entry_id")] = {"kind": "entry", "author": e.get("author"), "tick": e.get("tick"),
                                          "text": e.get("text"), "context": e.get("context"), "job_id": e.get("job_id")}
            for a in bd.get("archived") or []:
                walk(a)
        walk(b.get("binder", b) if isinstance(b, dict) else {})
    for w in rd.of("record_write"):
        eid = w.get("entry_id") or w.get("rev_id")
        if not eid or w.get("choice") in (None, "none"):
            continue
        out[eid] = {"kind": "rev" if w.get("rev_id") or w.get("choice") == "front" else "entry",
                    "author": w.get("agent"), "tick": w.get("tick"), "text": w.get("text"),
                    "context": w.get("offer"), "job_id": w.get("job") or w.get("job_id")}
    out.pop(None, None)
    return out


def front_history(run) -> list[dict]:
    """Chronological front-page states: [{tick, rev_id, text, author}]; a wipe resets the front to empty."""
    rd = _rd(run)
    tpd = V.ticks_per_day(rd)
    ev = [(r["tick"], 0, r_id, r) for r_id, r in binder_index(rd).items() if r["kind"] == "rev" and r["tick"] is not None]
    for t in rd.of("record_transition"):
        if t.get("mode") == "wipe":
            tick = t.get("tick") if t.get("tick") is not None else (int(t.get("day", 1)) - 1) * tpd
            ev.append((tick, -1, None, {"text": "", "author": None}))
    ev.sort(key=lambda x: (x[0], x[1]))
    return [{"tick": tk, "rev_id": rid, "text": r.get("text") or "", "author": r.get("author")} for tk, _o, rid, r in ev]


def front_at(history: list[dict], tick: int) -> dict:
    cur = {"tick": None, "rev_id": None, "text": "", "author": None}
    for h in history:
        if h["tick"] <= tick:
            cur = h
    return cur


def departed_by(rd, tick: int) -> set:
    tpd = V.ticks_per_day(rd)
    return {r.get("agent") for r in rd.of("roster_change")
            if r.get("kind") == "depart" and (int(r.get("day", 1)) - 1) * tpd <= tick}


_SENT = re.compile(r"(?<=[.!?;])\s+|\n+")


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def departed_author_share(rd, history: list[dict], tick: int) -> float | None:
    """Share of the current front-page characters whose sentence first appeared in a revision by an author
    who has departed by `tick` (sentence-level attribution through the rewrite chain)."""
    cur = front_at(history, tick)
    if not cur["text"]:
        return None
    gone = departed_by(rd, tick)
    first_author = {}
    for h in history:
        if h["tick"] > tick:
            break
        if not h["text"]:
            continue
        for s in _SENT.split(h["text"]):
            first_author.setdefault(_norm(s), h["author"])
    sents = [s for s in _SENT.split(cur["text"]) if _norm(s)]
    tot = sum(len(s) for s in sents)
    dep = sum(len(s) for s in sents if first_author.get(_norm(s), cur["author"]) in gone)
    return round(dep / tot, 4) if tot else None


def lineage_graph(run) -> dict:
    rd = _rd(run)
    idx = binder_index(rd)
    nodes = [{"id": i, "type": r["kind"], "author": r["author"], "tick": r["tick"]} for i, r in sorted(idx.items())]
    edges = []
    for r in rd.of("record_read"):
        new = set(r.get("new_ids") or [])
        for i in (r.get("rev_ids") or []) + (r.get("entry_ids") or []):
            edges.append({"src": i, "dst": f"read:{r.get('agent')}@{r.get('tick')}", "kind": "read",
                          "context": r.get("context"), "new": i in new})
    rec_nodes = {}
    for m in rd.of("memory_encoded"):
        rids = m.get("record_ids") or []
        if rids or m.get("source_type") == "record":
            rec_nodes[m["node_id"]] = rids
            for i in rids:
                edges.append({"src": i, "dst": m["node_id"], "kind": "memory", "agent": m.get("agent")})
    for nid, meta in (rd.memory_meta or {}).items():
        if isinstance(meta, dict) and meta.get("record_ids") and nid not in rec_nodes:
            rec_nodes[nid] = meta["record_ids"]
            edges += [{"src": i, "dst": nid, "kind": "memory"} for i in meta["record_ids"]]
    for u in rd.utterances:
        for nid in _ids(u.get("retrieved") or u.get("retrieved_ids")):
            if nid in rec_nodes:
                edges.append({"src": nid, "dst": u["id"], "kind": "utterance"})
    for d in rd.of("job_decision"):
        dst = f"decision:{d.get('job')}#{d.get('attempt')}"
        for i in d.get("binder_entry_ids") or []:
            edges.append({"src": i, "dst": dst, "kind": "view"})
        for nid in _ids(d.get("retrieved")):
            if nid in rec_nodes:
                edges.append({"src": nid, "dst": dst, "kind": "retrieved"})
    return {"nodes": nodes, "edges": edges, "record_memory_nodes": sorted(rec_nodes)}


def record_metrics(run, table: dict | None = None) -> dict:
    from backend.analysis import jobs as J
    rd = _rd(run)
    tpd, tmin = V.ticks_per_day(rd), int(rd.manifest.get("tick_minutes") or 15)
    table = table if table is not None else J.job_table(rd)
    fx = V.regime_fixes(rd.cfg)
    idx = binder_index(rd)
    writes = rd.of("record_write")
    reads = rd.of("record_read")
    faulted = [j for j in table.values() if j["faulted"]]
    days = max(1, len({j["day"] for j in table.values() if j["day"]}) or int(rd.manifest.get("ticks", tpd)) // tpd)
    made = [w for w in writes if w.get("choice") not in (None, "none")]
    hist = front_history(rd)
    # front-page staleness: hours from the first B-fault (K1 job in regime B) until the front stops recommending A
    sd = V.shift_day(rd.cfg)
    b_jobs = sorted((j for j in faulted if j["class"] == "K1" and j["regime"] == "B"),
                    key=lambda j: j["first_tick"] if j["first_tick"] is not None else (j["day"] - 1) * tpd)
    staleness = censored = None
    if sd and b_jobs:
        t0 = b_jobs[0]["first_tick"] if b_jobs[0]["first_tick"] is not None else (b_jobs[0]["day"] - 1) * tpd
        end = int(rd.manifest.get("ticks") or (max(j["day"] for j in table.values()) * tpd))
        stale_until, censored = None, True
        if fx["A"] not in V.recommended_actions(front_at(hist, t0)["text"]):
            stale_until, censored = t0, False
        else:
            for h in hist:
                if h["tick"] > t0 and fx["A"] not in V.recommended_actions(h["text"]):
                    stale_until, censored = h["tick"], False
                    break
        staleness = round(((stale_until if stale_until is not None else end) - t0) * tmin / 60, 2)
    # stale-view exposure: post-change K1 decisions whose view recommends A with no B correction
    stale_n = stale_k = 0
    for j in faulted:
        d = j.get("first_decision")
        if not d or j["class"] != "K1" or not sd or j["day"] < sd:
            continue
        acts = set().union(*[V.recommended_actions((idx.get(i) or {}).get("text")) for i in d.get("binder_entry_ids") or []] or [set()])
        stale_n += 1
        stale_k += int(fx["A"] in acts and fx["B"] not in acts)
    end_tick = int(rd.manifest.get("ticks") or 0) or max([h["tick"] for h in hist] + [0])
    talk = [u for u in rd.utterances if re.search(r"\bbinder\b", u.get("text") or "", re.I)]
    return {
        "reads": len(reads), "reads_decision": sum(1 for r in reads if r.get("context") == "decision"),
        "reads_per_faulted_job": round(sum(1 for r in reads if r.get("context") == "decision") / len(faulted), 4)
        if faulted else None,
        "offers": len(writes), "writes": len(made),
        "writes_per_offer": round(len(made) / len(writes), 4) if writes else None,
        "writes_per_day": round(len(made) / days, 4),
        "front_revisions": sum(1 for w in made if w.get("choice") == "front"),
        "staleness_hours": staleness, "staleness_censored": censored,
        "stale_view_share": round(stale_k / stale_n, 4) if stale_n else None, "stale_view_n": stale_n,
        "follow_rate": J.follow_rate(rd, table)["follow_rate"],
        "departed_author_share": departed_author_share(rd, hist, end_tick),
        "binder_talk_mentions": len(talk),
        "transitions": [{k: t.get(k) for k in ("day", "mode", "archived_binder_id")} for t in rd.of("record_transition")],
    }


def write_lineage(run) -> Path:
    rd = _rd(run)
    p = rd.dir / "lineage.json"
    json.dump(lineage_graph(rd), open(p, "w"), indent=1, sort_keys=True, default=str)
    return p
