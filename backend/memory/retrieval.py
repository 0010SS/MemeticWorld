"""Stochastic memory retrieval built on upstream GA `new_retrieve` scoring.

Upstream: score = w_rec*recency + w_rel*relevance + w_imp*importance (each
min-max normalised with GA's `normalize_dict_floats`), then deterministic top-k.

MemeWorld: the same normalised components (GA's `extract_importance`,
`extract_relevance` and `normalize_dict_floats` are called directly), but
  * recency is time-based, exp(-decay_rate * hours since last access), instead of
    GA's rank-based 0.99^i (which, upstream, also assigns the *highest* recency to
    the *oldest* node -- see docs/DECISIONS.md),
  * experimental modules may re-weight the scores (conformity, prestige, ...),
  * k memories are *sampled* with P(m) ∝ exp(s(m)/τ) (Gumbel-top-k, seeded).
    τ = 0 reproduces GA's deterministic top-k.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from backend import ga_compat
from backend.memory.store import recency_score

ga = ga_compat.load()


@dataclass
class RetrievalResult:
    focal: str
    nodes: list
    scores: dict  # node_id -> {"s", "rel", "rec", "imp"}


def retrieve(agent, focal_points: list[str], k: int | None = None, rng: np.random.Generator | None = None,
             kinds=("event", "thought", "chat"), exclude_ids=(), touch=True) -> dict[str, RetrievalResult]:
    cfg = agent.cfg["retrieval"]
    mcfg = agent.cfg["memory"]
    k = k or int(cfg["top_k"])
    tau = float(cfg["temperature"])
    rng = rng if rng is not None else agent.rng
    now = agent.scratch.curr_time
    out: dict[str, RetrievalResult] = {}
    pool = [n for n in agent.a_mem.all_nodes()
            if n.type in kinds and n.node_id not in exclude_ids and "idle" not in n.embedding_key]
    pool.sort(key=lambda n: (n.last_accessed, n.node_id))
    for focal in focal_points:
        if not pool:
            out[focal] = RetrievalResult(focal, [], {})
            continue
        rec = {n.node_id: recency_score(n, now, float(mcfg["decay_rate"])) for n in pool}
        rec = ga.retrieve.normalize_dict_floats(rec, 0, 1)
        imp = ga.retrieve.normalize_dict_floats(ga.retrieve.extract_importance(agent, pool), 0, 1)
        rel = ga.retrieve.normalize_dict_floats(ga.retrieve.extract_relevance(agent, pool, focal), 0, 1)
        scores = {nid: cfg["beta"] * rec[nid] + cfg["alpha"] * rel[nid] + cfg["gamma"] * imp[nid]
                  for nid in rec}
        scores = agent.mods.modify_retrieval(agent, pool, scores, focal)
        ids = list(scores)
        s = np.array([scores[i] for i in ids], dtype=float)
        kk = min(k, len(ids))
        if tau <= 1e-6:
            order = sorted(range(len(ids)), key=lambda i: (-s[i], ids[i]))[:kk]
        else:
            g = rng.gumbel(size=len(ids))
            keys = s / tau + g
            order = list(np.argsort(-keys, kind="stable")[:kk])
        chosen = [agent.a_mem.id_to_node[ids[i]] for i in order]
        if touch:
            for n in chosen:
                n.last_accessed = now
        out[focal] = RetrievalResult(focal, chosen, {
            ids[i]: {"s": round(float(s[i]), 3), "rel": round(float(rel[ids[i]]), 3),
                     "rec": round(float(rec[ids[i]]), 3), "imp": round(float(imp[ids[i]]), 3)}
            for i in order})
    return out


def merged_nodes(results: dict[str, RetrievalResult], limit: int | None = None) -> list:
    seen, out = set(), []
    for r in results.values():
        for n in r.nodes:
            if n.node_id not in seen:
                seen.add(n.node_id)
                out.append(n)
    return out[:limit] if limit else out


def softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    z = sum(e)
    return [v / z for v in e]
