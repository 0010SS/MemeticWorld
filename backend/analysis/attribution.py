"""Transmission vs independent rediscovery (OBSERVER ONLY; ontology v3 §5.7).

1. B-fix adoption: per agent in a shift arm, the first B-fix first attempt on a K1 job (post-change), and the
   first checkpoint where >= 5 of 8 K1c memory-only answers are the B-fix. Exposures before the adopting
   decision are labelled (multi-label):
   RECORD: the view shown at the decision, or a retrieved record node, recommends the B-fix;
   TALK:   a retrieved conversation/overheard node, or a clarify conversation for this job, recommends it;
   OWN:    a retrieved own or witnessed episode (source perception/self, event_ids ∋ a job) shows the B-fix
           succeeding;
   NONE:   none of these. Compared with the C0 rate of choosing the B-fix unprompted (rediscovery base rate).
   "Recommends" uses the keyword map (v3common.recommended_actions); the coder fallback is stubbed.
2. Expression spread for targets (X, Y and the world-anchored W = "F4"): v2 emergence_for over utterances AND
   binder entries, with binder reads counted as exposures (the reader is the listener of the entry's author).
   W is always flagged in_world_text, so it can never count as emergence; agents who perceived the panel fact
   before their first use are reported as world-exposed rather than independent coiners.
"""
from __future__ import annotations

import re
from pathlib import Path

from backend.analysis import v3common as V
from backend.analysis.rundata import RunData

LABELS = ("RECORD", "TALK", "OWN", "NONE")


def _rd(x) -> RunData:
    return x if isinstance(x, RunData) else RunData(Path(x))


def _node_texts(rd) -> dict:
    out = {}
    for m in rd.of("memory_encoded"):
        out[m["node_id"]] = {"text": m.get("text") or m.get("description") or "", "source": m.get("source_type"),
                             "events": m.get("originating_event_ids") or m.get("event_ids") or [],
                             "record_ids": m.get("record_ids") or []}
    return out


def label_decision(rd, decision: dict, bfix: str, table: dict, idx: dict, nodes: dict) -> list[str]:
    labels = set()
    for eid in decision.get("binder_entry_ids") or []:
        if bfix in V.recommended_actions((idx.get(eid) or {}).get("text")):
            labels.add("RECORD")
    ok_jobs = {jid for jid, j in table.items() for a in j["attempts"] if a.get("action") == bfix and a.get("success")}
    for x in decision.get("retrieved") or []:
        nid = x.get("id") or x.get("node_id") if isinstance(x, dict) else x
        n = nodes.get(nid) or {}
        text = n.get("text") or (x.get("text") if isinstance(x, dict) else "") or ""
        src = n.get("source") or (x.get("source_type") if isinstance(x, dict) else None)
        rec = bfix in V.recommended_actions(text)
        if rec and (src == "record" or n.get("record_ids")):
            labels.add("RECORD")
        elif rec and src in ("conversation", "overheard"):
            labels.add("TALK")
        if src in ("perception", "self", None) and set(n.get("events") or []) & ok_jobs:
            labels.add("OWN")
    jid = decision.get("job")
    for c in rd.conversations.values():      # clarify conversations for this job (topic clarify, operator in it)
        if (c.get("topic") == "clarify" and decision.get("agent") in (c.get("participants") or [])
                and (c.get("job") in (None, jid)) and c.get("tick", 0) <= decision.get("tick", 0)):
            text = " ".join(t[1] if isinstance(t, (list, tuple)) and len(t) > 1 else str(t)
                            for t in c.get("transcript") or [])
            if bfix in V.recommended_actions(text):
                labels.add("TALK")
    for c in rd.of("clarification"):
        if c.get("job") == jid and c.get("tick", 0) <= decision.get("tick", 0):
            text = " ".join(t if isinstance(t, str) else " ".join(map(str, t)) for t in c.get("transcript") or [])
            if bfix in V.recommended_actions(text):
                labels.add("TALK")
    return sorted(labels) or ["NONE"]


def adoption(run, table: dict | None = None, c0_rows: list | None = None) -> dict:
    from backend.analysis import jobs as J
    from backend.analysis.record_lineage import binder_index
    rd = _rd(run)
    sd = V.shift_day(rd.cfg)
    if sd is None:
        return {"applicable": False}
    table = table if table is not None else J.job_table(rd)
    bfix = V.regime_fixes(rd.cfg)["B"]
    idx, nodes = binder_index(rd), _node_texts(rd)
    per = {}
    for j in sorted(table.values(), key=lambda j: (j["first_tick"] or 0, j["job"])):
        d = j.get("first_decision")
        if (j["class"] != "K1" or j["day"] < sd or not d or not j["first_valid"] or j["first_action"] != bfix
                or d.get("agent") in per):
            continue
        per[d["agent"]] = {"job": j["job"], "tick": d.get("tick"),
                           "labels": label_decision(rd, d, bfix, table, idx, nodes)}
    # first checkpoint with >= 5 of 8 K1c memory-only answers equal to the B-fix
    from backend.analysis.meaning import available_checkpoints
    ck_first = {}
    for k in available_checkpoints(rd.dir):
        if V.ckpt_day(k) < sd:
            continue
        by: dict[str, list] = {}
        for r in V.probe_responses(rd.dir, k):
            if r.get("form") == "P" and r.get("type") == "K1c" and r.get("action"):
                by.setdefault(r["agent"], []).append(r["action"] == bfix)
        for a, hits in by.items():
            if a not in ck_first and hits and sum(hits) / len(hits) >= 5 / 8:
                ck_first[a] = k
    n = len(per)
    shares = {l: (round(sum(l in p["labels"] for p in per.values()) / n, 4) if n else None) for l in LABELS}
    c0 = [r for r in (c0_rows if c0_rows is not None else V.c0_responses(rd.dir))
          if r.get("form") == "P" and r.get("type") == "K1c" and r.get("action")]
    return {"applicable": True, "b_fix_first_attempt": per, "b_fix_checkpoint": ck_first, "n_adopters": n,
            "shares": shares, "c0_bfix_rate": V.share(c0, lambda r: r["action"] == bfix)}


def _usages(rd, expr: str) -> tuple[list, dict]:
    """(usages for emergence_for, first-use tick per producer): utterances + binder entries (reads as exposures)."""
    from backend.analysis.record_lineage import binder_index
    pat = re.compile(r"(?<![a-z0-9])" + re.escape(expr.lower()) + r"(?![a-z0-9])")
    us = []
    for u in rd.utterances:
        if pat.search((u.get("text") or "").lower()):
            us.append({"utterance_id": u["id"], "tick": u["tick"], "speaker": u.get("speaker"),
                       "listeners": u.get("listeners") or [], "conversation_id": u.get("conversation_id"),
                       "idx": u.get("idx", 0), "channel": "talk"})
    idx = binder_index(rd)
    hits = {i: e for i, e in idx.items() if pat.search((e.get("text") or "").lower())}
    for i, e in hits.items():
        us.append({"utterance_id": i, "tick": e["tick"], "speaker": e["author"], "listeners": [],
                   "conversation_id": None, "idx": 0, "channel": "binder"})
    for r in rd.of("record_read"):
        for i in (r.get("entry_ids") or []) + (r.get("rev_ids") or []):      # every displayed item is an exposure
            if i in hits and r.get("agent") != hits[i]["author"]:
                us.append({"utterance_id": f"{i}@{r.get('agent')}:{r.get('tick')}", "tick": r.get("tick"),
                           "speaker": hits[i]["author"], "listeners": [r.get("agent")], "conversation_id": None,
                           "idx": 0, "channel": "read", "exposure_only": True})
    first = {}
    for u in us:
        if not u.get("exposure_only") and u["speaker"]:
            first[u["speaker"]] = min(first.get(u["speaker"], u["tick"]), u["tick"])
    return us, first


def expression_spread(run, expr: str, world_anchored: bool = False) -> dict:
    """v2 emergence over talk + binder, binder reads as exposures. world_anchored (W = F4): flagged
    in_world_text; producers who perceived the world form before their first use are `world_exposed`."""
    from backend.analysis.emergence import emergence_for
    rd = _rd(run)
    us, first = _usages(rd, expr)
    # the author "re-speaking" at read ticks is an exposure device only; keep the author's own first use first
    pop = sorted(set(rd.agents))
    in_world = world_anchored or expr.lower() in (rd.world_text if hasattr(rd, "world_text") else "")
    res = emergence_for(us, pop, in_world_text=bool(in_world))
    world_exposed = []
    if world_anchored:
        pat = expr.lower()
        seen = {}
        for v in rd.of("viewpoint", "perception"):
            txt = " ".join(str(f.get("perceived") or f.get("text") or "") for f in v.get("facts", []) or [])
            if pat in txt.lower() and v.get("agent"):
                seen[v["agent"]] = min(seen.get(v["agent"], v["tick"]), v["tick"])
        world_exposed = sorted(a for a, t in first.items() if a in seen and seen[a] <= t)
    res.update({"expr": expr, "world_anchored": world_anchored, "producers": sorted(first),
                "n_binder_uses": sum(1 for u in us if u["channel"] == "binder"),
                "n_read_exposures": sum(1 for u in us if u["channel"] == "read"),
                "world_exposed_producers": world_exposed})
    return res


def analyze_attribution(run, targets: dict | None = None, table: dict | None = None) -> dict:
    rd = _rd(run)
    targets = dict(targets or {})
    pc = ((rd.cfg.get("workshop") or {}).get("panel_code") or {})
    if pc.get("enabled") and "W" not in targets:
        targets["W"] = pc.get("text") or "F4"
    return {"adoption": adoption(rd, table),
            "spread": {k: expression_spread(rd, v, world_anchored=(k == "W")) for k, v in sorted(targets.items()) if v}}
