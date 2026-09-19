"""Transmission graph for each meme candidate (OBSERVER ONLY).

When B first uses expression m after having heard A use it, A -> B is a
candidate transmission path. We keep *all* plausible prior exposures, each with
a confidence (recency-weighted, boosted if B's own encoded memory of that
exposure retained the wording), normalised against an "independent invention"
alternative. The most recent speaker is not assumed to be causal. An exposure counts only when it
causally precedes the use (`rundata.precedes`): same-tick remarks, conversation openings and parallel
conversations cannot have heard each other.
"""
from __future__ import annotations

import math
from collections import defaultdict

from backend.analysis.rundata import chron_key, precedes

INDEPENDENT_PRIOR = 0.15


def analyze_transmission(cand: dict, rd, retained: dict) -> dict:
    """retained: (agent, utterance_id) -> memory text containing the utterance (from memory_encoded)."""
    idx = {uid: u.get("idx") for uid, u in (getattr(rd, "utt_by_id", None) or {}).items()}
    usages = sorted((dict(u, idx=idx.get(u["utterance_id"], u.get("idx"))) for u in cand["usages"]), key=chron_key)
    variants = cand["variants"]
    by_speaker_first = {}
    for u in usages:
        by_speaker_first.setdefault(u["speaker"], u)
    exposures = defaultdict(list)   # listener -> [(tick, speaker, utterance_id)]
    for u in usages:
        for l in u["listeners"]:
            if l != u["speaker"]:
                exposures[l].append(u)
    edges, inventors = [], []
    tpd = rd.manifest["ticks_per_day"]
    tm = rd.manifest["tick_minutes"]
    for spk, first in sorted(by_speaker_first.items(), key=lambda kv: chron_key(kv[1])):
        prior = [e for e in exposures.get(spk, []) if precedes(e, first)]
        if not prior:
            inventors.append({"agent": spk, "tick": first["tick"], "utterance_id": first["utterance_id"]})
            continue
        weights = defaultdict(float)
        detail = defaultdict(list)
        for e in prior:
            hours = (first["tick"] - e["tick"]) * tm / 60.0
            w = math.exp(-hours / 12.0)
            mem = retained.get((spk, e["utterance_id"]), "").lower()
            if any(v in mem for v in variants):
                w *= 2.0
            weights[e["speaker"]] += w
            detail[e["speaker"]].append({"utterance_id": e["utterance_id"], "tick": e["tick"],
                                         "retained_in_memory": any(v in mem for v in variants)})
        z = sum(weights.values()) + INDEPENDENT_PRIOR
        for src, w in sorted(weights.items()):
            ex = detail[src]
            edges.append({"meme_id": cand["id"], "source_agent": src, "target_agent": spk,
                          "exposure_timestamp": min(x["tick"] for x in ex), "exposures": ex,
                          "first_reuse_timestamp": first["tick"], "first_reuse_utterance": first["utterance_id"],
                          "confidence": round(w / z, 3),
                          "cross_group": not (rd.groups_of(src) & rd.groups_of(spk))})
        if INDEPENDENT_PRIOR / z > 0.5:
            inventors.append({"agent": spk, "tick": first["tick"], "utterance_id": first["utterance_id"],
                              "partial": round(INDEPENDENT_PRIOR / z, 3)})
    # depth: longest chain following the most confident parent of each adopter
    best_parent = {}
    for e in edges:
        t = e["target_agent"]
        if t not in best_parent or e["confidence"] > best_parent[t]["confidence"]:
            best_parent[t] = e

    def depth(a, seen=()):
        p = best_parent.get(a)
        if not p or p["source_agent"] in seen:
            return 0
        return 1 + depth(p["source_agent"], seen + (a,))
    max_depth = max([depth(a) for a in by_speaker_first] or [0])
    # adoption curve
    curve, users = [], set()
    for u in usages:
        users.add(u["speaker"])
        curve.append({"tick": u["tick"], "users": len(users), "uses": len(curve) + 1})
    per_day = defaultdict(int)
    for u in usages:
        per_day[u["tick"] // tpd + 1] += 1
    cross = [e for e in edges if e["cross_group"] and e["confidence"] >= 0.3]
    return {"edges": edges, "inventors": inventors, "depth": max_depth, "adoption_curve": curve,
            "uses_per_day": dict(sorted(per_day.items())),
            "first_cross_group": min(cross, key=lambda e: e["first_reuse_timestamp"]) if cross else None,
            "n_exposed": len({l for u in usages for l in u["listeners"]} | set(by_speaker_first))}
