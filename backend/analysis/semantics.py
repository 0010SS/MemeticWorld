"""Semantic analysis of candidate usages (OBSERVER ONLY).

Context embeddings are computed with the expression itself masked out, so the
coherence score measures what the expression is used *about*, not the fact
that the same words recur.
"""
from __future__ import annotations

import itertools
import re
from collections import defaultdict

import numpy as np

from backend.llm.embeddings import cos


def _mask(text: str, variants: list[str]) -> str:
    out = text
    for v in sorted(variants, key=len, reverse=True):
        out = re.sub(re.escape(v), " ", out, flags=re.I)
    return out


def _mean_pairwise(vecs) -> float | None:
    if len(vecs) < 2:
        return None
    sims = [cos(a, b) for a, b in itertools.combinations(vecs, 2)]
    return float(np.mean(sims))


def analyze_semantics(cand: dict, rd, embed) -> dict:
    usages = cand["usages"]
    vecs = [np.array(embed(_mask(u["context"], cand["variants"]))) for u in usages]
    for u, v in zip(usages, vecs):
        u["_vec"] = v
    if not vecs:
        return {}
    centroid = np.mean(vecs, axis=0)
    coherence = _mean_pairwise(vecs)
    # group structure (a speaker may belong to several groups; usage counted in each)
    by_group = defaultdict(list)
    for u in usages:
        for g in rd.groups_of(u["speaker"]) or {"(none)"}:
            by_group[g].append(u["_vec"])
    within = {g: _mean_pairwise(v) for g, v in by_group.items() if len(v) >= 2}
    between = []
    gs = [g for g in by_group if by_group[g]]
    for g1, g2 in itertools.combinations(gs, 2):
        c1, c2 = np.mean(by_group[g1], axis=0), np.mean(by_group[g2], axis=0)
        between.append(cos(c1, c2))
    # meaning over time: per-day centroid vs overall centroid and vs first day's centroid
    tpd = rd.manifest["ticks_per_day"]
    by_day = defaultdict(list)
    for u in usages:
        by_day[u["tick"] // tpd + 1].append(u["_vec"])
    days = sorted(by_day)
    first_c = np.mean(by_day[days[0]], axis=0)
    over_time = [{"day": d, "n": len(by_day[d]),
                  "sim_to_overall": round(cos(np.mean(by_day[d], axis=0), centroid), 3),
                  "sim_to_first_day": round(cos(np.mean(by_day[d], axis=0), first_c), 3)} for d in days]
    for u in usages:
        u["sim_to_centroid"] = round(cos(u["_vec"], centroid), 3)
        del u["_vec"]
    return {"coherence": None if coherence is None else round(coherence, 3),
            "within_group": {g: round(v, 3) for g, v in within.items() if v is not None},
            "between_group_centroid_sim": round(float(np.mean(between)), 3) if between else None,
            "over_time": over_time}
