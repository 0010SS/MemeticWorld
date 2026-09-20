"""Semantic analysis of candidate usages (OBSERVER ONLY).

Context embeddings are computed with the expression itself masked out, so the
coherence score measures what the expression is used *about*, not the fact
that the same words recur.

`topic_control` exists because the memo is blunt that these measures "can confuse topic change with
meaning change": it reports the same similarity drop for the whole run's talk, so a drop the run shows
everywhere is read as topic drift rather than as the meme's meaning moving.
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


def _split(items, key, cut: int | None):
    if cut is None:
        return list(items), []
    return [x for x in items if key(x) < cut], [x for x in items if key(x) >= cut]


def _drift(pre_vecs, post_vecs) -> dict:
    """Mean pairwise coherence each side of a cut, and the similarity between the two centroids."""
    out = {"n_pre": len(pre_vecs), "n_post": len(post_vecs),
           "coherence_pre": _mean_pairwise(pre_vecs), "coherence_post": _mean_pairwise(post_vecs),
           "centroid_sim": None}
    if pre_vecs and post_vecs:
        out["centroid_sim"] = round(cos(np.mean(pre_vecs, axis=0), np.mean(post_vecs, axis=0)), 3)
    return out


def topic_control(usages: list[dict], variants: list[str], rd, embed, cut_tick: int | None,
                  sample: int = 120) -> dict:
    """Context drift across a cut, WITH the run-wide topic drift computed the same way beside it.

    NOT EVIDENCE OF MEANING CHANGE ON ITS OWN. A fall in centroid similarity is explained just as well by
    the community having moved on to other subjects; `excess` (the meme's drop minus the run's drop) is the
    part a topic change does not already account for, and even that is a competing explanation to rule out
    next to the probe boundary, never a second measurement of it.
    """
    pre, post = _split(usages, lambda u: u["tick"], cut_tick)
    mv = lambda us: [np.array(embed(_mask(u.get("context") or u.get("text") or "", variants))) for u in us]
    meme = _drift(mv(pre), mv(post))
    # the baseline is every utterance of the run, deterministically thinned to `sample` per side so a long
    # run does not embed tens of thousands of lines for a control
    allu = sorted(rd.utterances, key=lambda u: (u["tick"], u.get("id") or ""))
    bpre, bpost = _split(allu, lambda u: u["tick"], cut_tick)
    thin = lambda us: us[:: max(1, len(us) // sample)][:sample]
    base = _drift([np.array(embed(u.get("text") or "")) for u in thin(bpre)],
                  [np.array(embed(u.get("text") or "")) for u in thin(bpost)])
    excess = None
    if meme["centroid_sim"] is not None and base["centroid_sim"] is not None:
        excess = round(base["centroid_sim"] - meme["centroid_sim"], 3)
    return {"cut_tick": cut_tick, "meme": meme, "run_baseline": base, "excess_drift": excess,
            "caveat": "context similarity cannot separate topic change from meaning change; read the "
                      "probe boundary (battery.boundary) for meaning"}
