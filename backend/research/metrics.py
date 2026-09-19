"""Deterministic measurements over evidence-linked interpretations.

Counts describe the observer's discovered repertoire, not an exhaustive inventory of culture.
Unobserved interpretations stay missing; exposure is distinct from reuse and endorsement.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean

from backend.research.evidence import precedes

DEFINITIONS = {
    "n_threads": "Concept histories proposed by this observer, including private and single-occurrence histories.",
    "n_public_threads": "Histories with at least one speech or authored-record occurrence.",
    "n_multi_agent_threads": "Histories with public uses by at least two distinct agents; a descriptive count, not a universal meme definition.",
    "n_changes": "Evidence-linked changes judged by the observer; uncertainty remains attached to each claim.",
    "mean_public_users": "Mean distinct public users per publicly expressed concept history.",
    "mean_senses": "Mean number of observer-distinguished senses per history.",
    "mean_persistence_days": "Mean span between first and last observed public use, in simulated days; not an extinction estimate.",
    "n_supported_reuses": "First public uses with at least one logged preceding exposure to a related occurrence; not a causal attribution.",
    "mean_agent_agreement": "Average daily pairwise agreement of agent-normalized sense distributions among agents with observed public uses.",
}


def distribution(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def js_distance(a, b):
    keys = set(a) | set(b)
    if not a or not b:
        return None
    mid = {k: (a.get(k, 0) + b.get(k, 0)) / 2 for k in keys}
    def kl(p):
        return sum(v * math.log2(v / mid[k]) for k, v in p.items() if v)
    return math.sqrt(max(0.0, (kl(a) + kl(b)) / 2))


def measure(threads, changes, evidence):
    all_ids = {o["event_id"] for t in threads for o in t["occurrences"]}
    sources = {e["id"]: e for e in evidence.events(ids=all_ids)}
    exposure_by = defaultdict(list)
    for e in evidence.exposures():
        exposure_by[e["source"]].append(e)
    populations = evidence.populations()
    by_thread, totals = {}, []
    for thread in threads:
        occurrences = sorted(thread["occurrences"], key=lambda o: (o["tick"], sources[o["event_id"]]["seq"], o["id"]))
        public = [o for o in occurrences if o["channel"] in ("speech", "record") and o["actor"]]
        first, users = {}, set()
        for o in public:
            first.setdefault(o["actor"], o)
            users.add(o["actor"])
        edges, exposed_users, uptake = [], set(), {}
        for o in public:
            exposed_users.update(x["actor"] for x in exposure_by[o["event_id"]])
        for actor, target in first.items():
            parents = []
            for origin in public:
                if origin["actor"] == actor or not precedes(sources[origin["event_id"]], sources[target["event_id"]]):
                    continue
                for receipt in exposure_by[origin["event_id"]]:
                    if receipt["actor"] != actor:
                        continue
                    timely = receipt["tick"] < target["tick"] or (
                        receipt["tick"] == target["tick"] and receipt["via"] == "heard"
                        and precedes(sources[origin["event_id"]], sources[target["event_id"]]))
                    if not timely:
                        continue
                    edge = {"thread": thread["id"], "source": origin["actor"], "target": actor,
                            "source_event": origin["event_id"], "target_event": target["event_id"],
                            "source_occurrence": origin["id"], "target_occurrence": target["id"],
                            "exposure_tick": receipt["tick"], "reuse_tick": target["tick"], "via": receipt["via"],
                            "same_sense": origin["sense"] == target["sense"],
                            "same_exchange": bool(origin["conversation"] and origin["conversation"] == target["conversation"]),
                            "attribution": "possible_source"}
                    parents.append(edge)
            edges.extend(parents)
            uptake[actor] = {"first_use_tick": target["tick"], "prior_exposure": bool(parents),
                             "source_candidates": len(parents),
                             "later_exchange": any(not p["same_exchange"] for p in parents)}
        by_day = []
        prior_distributions = {}
        for day in sorted(populations):
            active = set(populations[day])
            rows = [o for o in public if o["day"] == day and o["actor"] in active]
            per_agent = defaultdict(Counter)
            stance = defaultdict(Counter)
            for o in rows:
                per_agent[o["actor"]][o["sense"]] += 1
                stance[o["actor"]][o["stance"]] += 1
            agent_dist = {a: distribution(c) for a, c in per_agent.items()}
            collective = {s: mean(d.get(s, 0) for d in agent_dist.values())
                          for s in {s for d in agent_dist.values() for s in d}} if agent_dist else {}
            names = sorted(agent_dist)
            similarities = [sum(agent_dist[a].get(s, 0) * agent_dist[b].get(s, 0)
                                for s in set(agent_dist[a]) | set(agent_dist[b]))
                            for i, a in enumerate(names) for b in names[i + 1:]]
            individual_change = {a: js_distance(prior_distributions[a], d)
                                 for a, d in agent_dist.items() if a in prior_distributions}
            prior_distributions.update(agent_dist)
            groups = {}
            for group, members in evidence.manifest.get("groups", {}).items():
                ds = [agent_dist[a] for a in members if a in agent_dist]
                groups[group] = {"observed_agents": len(ds),
                                 "distribution": {s: mean(d.get(s, 0) for d in ds) for s in collective} if ds else {}}
            by_day.append({"day": day, "active_agents": len(active), "observed_agents": len(agent_dist),
                           "public_occurrences": len(rows), "public_users": sorted(per_agent),
                           "reach": len(per_agent) / len(active) if active else None,
                           "coverage": len(agent_dist) / len(active) if active else None,
                           "agent_distributions": agent_dist, "collective_distribution": collective,
                           "agent_stances": {a: dict(c) for a, c in stance.items()},
                           "agreement": mean(similarities) if similarities else None,
                           "within_agent_distribution_change": individual_change, "groups": groups})
        last = max((o["tick"] for o in public), default=None)
        start = min((o["tick"] for o in public), default=None)
        first_origin = occurrences[0]
        origin_type = ("initial_material" if first_origin["channel"] == "initial" else
                       "private_representation" if first_origin["channel"] == "private" else
                       "world_provided" if first_origin["channel"] == "environment" else "agent_expression")
        rec = {"id": thread["id"], "public_users": len(users), "users": sorted(users),
               "public_occurrences": len(public), "private_occurrences": sum(o["channel"] == "private" for o in occurrences),
               "first_observed_tick": first_origin["tick"], "first_public_tick": start, "last_public_tick": last,
               "origin_evidence": first_origin["event_id"], "origin_type": origin_type,
               "observed_span_days": (last - start) / evidence.tpd if start is not None else None,
               "senses": len(thread["senses"]), "forms": dict(Counter(o["quote"] for o in public)),
               "channels": dict(Counter(o["channel"] for o in occurrences)),
               "stances": dict(Counter(o["stance"] for o in public)),
               "exposed_agents": sorted(exposed_users), "uptake": uptake,
               "possible_transmission": edges, "daily": by_day,
               "changes": [c for c in changes if c["thread"] == thread["id"]]}
        by_thread[thread["id"]] = rec
        totals.append(rec)
    pub = [r for r in totals if r["public_occurrences"]]
    agreements = [d["agreement"] for r in totals for d in r["daily"] if d["agreement"] is not None]
    summary = {"n_threads": len(threads), "n_public_threads": len(pub),
               "n_multi_agent_threads": sum(r["public_users"] >= 2 for r in totals), "n_changes": len(changes),
               "mean_public_users": mean(r["public_users"] for r in pub) if pub else None,
               "mean_senses": mean(r["senses"] for r in totals) if totals else None,
               "mean_persistence_days": mean(r["observed_span_days"] for r in pub) if pub else None,
               "n_supported_reuses": sum(x["prior_exposure"] for r in totals for x in r["uptake"].values()),
               "mean_agent_agreement": mean(agreements) if agreements else None}
    return {"summary": summary, "threads": by_thread, "definitions": DEFINITIONS,
            "populations": populations, "notes": [
                "Semantic interpretations are observer judgments, not direct access to beliefs.",
                "Distribution change can reflect changing sense prevalence without creating a new sense.",
                "Transmission edges are possible sources with logged exposure, not established causes.",
                "No observed use is missing evidence about an agent's interpretation.",
                "Spans and uptake use the available observation horizon; late appearances are censored.",
            ]}
