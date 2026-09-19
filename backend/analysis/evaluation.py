"""Ground-truth evaluation of candidates against hidden latent event types
(OBSERVER ONLY; the mapping is never exposed to agents).

A usage is linked to world events through simulator-side metadata: the
originating_event_ids of the memories the speaker retrieved when producing the
utterance (plus the private trigger of reaction-initiated conversations).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def usage_types(u: dict, events: dict) -> Counter:
    c = Counter()
    ids = u.get("retrieved_event_ids", [])
    for e in ids:
        if e in events:
            c[events[e]["latent_type"]] += 1.0 / len(ids)
    return c


def base_rates(rd) -> dict:
    """Latent-type distribution over ALL utterances (for lift)."""
    tot = Counter()
    n = 0
    for u in rd.utterances:
        c = usage_types(u, rd.events)
        if c:
            n += 1
            for k, v in c.items():
                tot[k] += v
    return {k: round(v / max(1, n), 3) for k, v in tot.items()} | {"_linked_fraction": round(n / max(1, len(rd.utterances)), 3)}


def _best_precision(usages: list, label: dict) -> float:
    """Share of usages linked to the most common latent type (the post-hoc 'best type')."""
    dist = Counter()
    for u in usages:
        ids = [e for e in u.get("retrieved_event_ids", []) if e in label]
        for e in ids:
            dist[label[e]] += 1.0 / len(ids)
    if not dist:
        return 0.0
    z = max(dist, key=lambda k: dist[k])
    return sum(1 for u in usages if any(label.get(e) == z for e in u.get("retrieved_event_ids", []))) / len(usages)


def permutation_null(usages: list, events: dict, n: int = 500, seed: int = 0) -> dict:
    """Chance baseline for alignment: shuffle latent-type labels across the run's events (keeping the
    type counts and every usage's event links fixed) and recompute best-type precision. The best type
    is chosen post hoc, so the null picks it post hoc too. p = P(null >= observed)."""
    ids = sorted(events)
    types = [events[e]["latent_type"] for e in ids]
    obs = _best_precision(usages, dict(zip(ids, types)))
    rng = np.random.default_rng(seed)
    null = np.array([_best_precision(usages, dict(zip(ids, rng.permutation(types)))) for _ in range(n)])
    return {"observed": round(obs, 3), "null_mean": round(float(null.mean()), 3),
            "null_p95": round(float(np.quantile(null, 0.95)), 3),
            "p_value": round(float((1 + (null >= obs - 1e-12).sum()) / (n + 1)), 4)}


def evaluate(cand: dict, rd, probes: dict | None, rates: dict) -> dict:
    events = rd.events
    usages = cand["usages"]
    dist = Counter()
    linked = 0
    for u in usages:
        c = usage_types(u, events)
        u["latent_types"] = dict(c)
        if c:
            linked += 1
        for k, v in c.items():
            dist[k] += v
    if not dist:
        return {"latent_distribution": {}, "best_type": None, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "alignment": 0.0, "linked_usages": 0, "generalization": None, "drift": None, "lift": None,
                "permutation": None}
    z = max(dist, key=lambda k: dist[k])
    hits = [u for u in usages if z in u["latent_types"]]
    precision = len(hits) / len(usages)
    t0 = cand["first_occurrence"]["tick"]
    later = [e for e in events.values() if e["latent_type"] == z and e["start_tick"] >= t0]
    triggered = {e for u in usages for e in u.get("retrieved_event_ids", [])}
    recall = (sum(1 for e in later if e["id"] in triggered) / len(later)) if later else None
    hold = [e for e in later if e.get("holdout")]
    gen_usage = (sum(1 for e in hold if e["id"] in triggered) / len(hold)) if hold else None
    half = len(usages) // 2
    drift = None
    if half >= 2:
        p1 = sum(1 for u in usages[:half] if z in u["latent_types"]) / half
        p2 = sum(1 for u in usages[half:] if z in u["latent_types"]) / (len(usages) - half)
        drift = {"precision_first_half": round(p1, 3), "precision_second_half": round(p2, 3),
                 "drift": round(p1 - p2, 3)}
    r = recall if recall is not None else 0.0
    f1 = 2 * precision * r / (precision + r) if precision + r > 0 else 0.0
    probe_acc = None
    if probes:
        answered = [p for p in probes.values() if p.get("match_family")]
        if answered:
            probe_acc = sum(1 for p in answered if p["match_family"] == z) / len(answered)
    return {"latent_distribution": {k: round(v, 3) for k, v in dist.items()}, "best_type": z,
            "linked_usages": linked, "precision": round(precision, 3),
            "recall": None if recall is None else round(recall, 3), "f1": round(f1, 3),
            "alignment": round(precision if recall is None else f1, 3),
            "lift": round(precision / rates.get(z, 1e-9), 3) if rates.get(z) else None,
            "generalization": {"holdout_instances": len(hold), "spontaneous_use_rate": gen_usage,
                               "probe_match_accuracy": None if probe_acc is None else round(probe_acc, 3)},
            "drift": drift,
            "permutation": permutation_null(usages, events, seed=int(rd.cfg.get("seed", 0)))}


def status(cand: dict, rd) -> str:
    total = rd.manifest["ticks"]
    tpd = rd.manifest["ticks_per_day"]
    last = cand["usages"][-1]["tick"]
    n_users = len(cand["speakers"])
    pop = len(rd.agents)
    if last < total - tpd * 0.5 and total > tpd:
        return "fading"
    new_recent = {u["speaker"] for u in cand["usages"] if u["tick"] >= total - tpd} - \
                 {u["speaker"] for u in cand["usages"] if u["tick"] < total - tpd}
    if n_users >= max(3, pop // 2):
        return "established"
    if new_recent:
        return "spreading"
    return "emerging"
