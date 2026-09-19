"""Symbol lineage (OBSERVER ONLY): soft parent-child links between variants and
between candidates, from lexical similarity + context similarity + temporal
proximity + exposure overlap. Low-confidence links are kept as scores but not
asserted (no forced hard lineage).
"""
from __future__ import annotations

import math

import numpy as np

from backend.analysis.candidates import STOP
from backend.llm.embeddings import HashEmbedder, cos

_E = HashEmbedder(256)


def lexical_sim(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    jac = len(ta & tb) / max(1, len(ta | tb))
    contain = 0.85 if (a in b or b in a) else 0.0
    stem = 0.0
    for x in ta:
        for y in tb:
            if x not in STOP and y not in STOP and len(x) >= 4 and len(y) >= 4:
                k = 0
                while k < min(len(x), len(y)) and x[k] == y[k]:
                    k += 1
                if k >= 4:
                    stem = max(stem, 0.5 + 0.4 * k / max(len(x), len(y)))
    return max(jac, contain, stem, 0.7 * cos(_E(a), _E(b)))


def variant_tree(cand: dict) -> list[dict]:
    first = {}
    for u in cand["usages"]:
        first.setdefault(u["variant"], u)
    order = sorted(first.items(), key=lambda kv: (kv[1]["tick"], kv[1]["utterance_id"]))
    edges = []
    for i, (v, u) in enumerate(order):
        if i == 0:
            continue
        best = max(order[:i], key=lambda kv: lexical_sim(kv[0], v))
        conf = lexical_sim(best[0], v)
        edges.append({"parent": best[0], "child": v, "confidence": round(conf, 3),
                      "child_first_tick": u["tick"], "asserted": conf >= 0.5})
    return edges


def candidate_lineage(cands: list[dict], rd, sem: dict) -> list[dict]:
    edges = []
    tm = rd.manifest["tick_minutes"]
    for a in cands:
        for b in cands:
            if a is b or a["first_occurrence"]["tick"] > b["first_occurrence"]["tick"]:
                continue
            if a["first_occurrence"]["tick"] == b["first_occurrence"]["tick"] and a["id"] >= b["id"]:
                continue
            lex = max(lexical_sim(x, y) for x in a["variants"] for y in b["variants"])
            if lex < 0.3:
                continue
            ca = sem.get(a["id"], {}).get("_centroid")
            cb = sem.get(b["id"], {}).get("_centroid")
            ctx = cos(ca, cb) if ca is not None and cb is not None else 0.0
            dh = (b["first_occurrence"]["tick"] - a["first_occurrence"]["tick"]) * tm / 60
            temporal = math.exp(-dh / 24)
            exposed_a = {l for u in a["usages"] for l in u["listeners"]} | set(a["speakers"])
            overlap = len(set(b["speakers"]) & exposed_a) / max(1, len(b["speakers"]))
            conf = 0.45 * lex + 0.25 * max(0.0, ctx) + 0.15 * temporal + 0.15 * overlap
            edges.append({"parent": a["id"], "child": b["id"], "parent_form": a["canonical_form"],
                          "child_form": b["canonical_form"], "confidence": round(conf, 3),
                          "components": {"lexical": round(lex, 3), "context": round(float(ctx), 3),
                                         "temporal": round(temporal, 3), "exposure_overlap": round(overlap, 3)},
                          "asserted": conf >= 0.55})
    return sorted(edges, key=lambda e: -e["confidence"])


def centroid(cand, embed):
    from backend.analysis.semantics import _mask
    vs = [np.array(embed(_mask(u["context"], cand["variants"]))) for u in cand["usages"]]
    return np.mean(vs, axis=0) if vs else None
