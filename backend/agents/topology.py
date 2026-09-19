"""Generated social topology (ontology v2 §3, `topology.mode: generated`).

The population file hand-writes who knows whom. To make tie structure an experimental variable,
`generate` partitions the population into disjoint circles with dense ties inside and sparse weak
ties between, adds a few bridge agents with one strong tie into another circle, and (with
`shared_meals`) gives each circle a shared dinner slot so that who is co-located agrees with who is
tied. Deterministic given the rng. Agents are never told about circles: they only get ordinary
relationship statements and an ordinary routine entry.

Roommates (agents sharing a home room) always stay in the same circle, so a generated tie never
contradicts the shared room that profiles and routines already imply.
"""
from __future__ import annotations

import dataclasses
from itertools import combinations

from backend.agents.profile import AgentProfile, Relationship, RoutineEntry

MEAL_MINUTES = 60                 # a shared dinner lasts at least this long; plans during it start after it
DINNER_WINDOW = "18:00-19:30"     # used when conversation.group.windows has no evening window
DINNER_ACTIVITY = "eating dinner"  # routine wording already in the population lexicon
STRONG = 0.6                      # familiarity of a strong tie (the catch-up threshold)


def _mins(v) -> int:
    h, m = map(int, str(v).split(":"))
    return h * 60 + m


def _hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def _key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _home(p: AgentProfile) -> tuple:
    h = p.home or {}
    return h.get("location"), h.get("arena")


def _partition(profiles: dict[str, AgentProfile], n: int, rng) -> dict[str, list[str]]:
    """Disjoint circles of near-equal size (>= 3 members when the population allows), built from
    households (agents sharing a home room) so roommates are never split."""
    ids = sorted(profiles)
    n = min(int(n), len(ids) // 3)
    if n <= 0:
        return {}
    by_home: dict[tuple, list[str]] = {}
    for a in ids:
        by_home.setdefault(_home(profiles[a]), []).append(a)
    units = list(by_home.values())
    units = [units[int(i)] for i in rng.permutation(len(units))]
    units.sort(key=len, reverse=True)           # stable: random order among equal sizes
    base, extra = divmod(len(ids), n)
    target = [base + (i < extra) for i in range(n)]
    members: list[list[str]] = [[] for _ in range(n)]
    for u in units:
        i = max(range(n), key=lambda c: (target[c] - len(members[c]), -c))
        members[i] += u
    members = [m for m in members if len(m) >= 3]   # an oversized household can starve a circle
    return {f"c{i + 1}": sorted(m) for i, m in enumerate(members)}


def _dinner_slots(cfg: dict, circles: dict[str, list[str]]) -> list[dict]:
    """One (time, venue) per circle. Circles share venues from conversation.group.venues and are staggered
    across its evening window, so each circle's dinner falls where and when group talk can happen."""
    from backend.simulation.world import ARENAS
    gc = (cfg.get("conversation") or {}).get("group") or {}
    venues = [tuple(v) for v in (gc.get("venues") or [["Dining Hall", "Main Floor"]])]
    for loc, arena in venues:
        if arena not in ARENAS.get(loc, []):
            raise ValueError(f"conversation.group.venues: unknown venue {loc!r}/{arena!r}")
    evening = [w for w in (gc.get("windows") or []) if _mins(w.split("-")[0]) >= 16 * 60] or [DINNER_WINDOW]
    w0, w1 = (_mins(x) for x in max(evening, key=lambda w: _mins(w.split("-")[0])).split("-"))
    per_venue = -(-len(circles) // len(venues))
    step = max(15, (w1 - w0) // per_venue // 15 * 15)
    return [{"circle": cid, "time": _hhmm(w0 + (i // len(venues)) * step), "location": venues[i % len(venues)][0],
             "arena": venues[i % len(venues)][1], "minutes": MEAL_MINUTES}
            for i, cid in enumerate(circles)]


def generate(profiles: dict[str, AgentProfile], cfg: dict, rng) -> dict:
    """-> {"circles": {cid: [ids]}, "relationships": {(a, b): Relationship} (a < b; unlisted pairs are
    strangers), "routine_additions": {aid: [RoutineEntry]}, "metrics": {...}} from `topology.generated`."""
    g = cfg["topology"]["generated"]
    ids = sorted(profiles)
    circles = _partition(profiles, g["n_circles"], rng)
    circle_of = {a: c for c, ms in circles.items() for a in ms}
    fw, aw = float(g["familiarity_within"]), float(g["affinity_within"])
    rels: dict[tuple[str, str], Relationship] = {}
    for a, b in combinations(ids, 2):
        roommates = _home(profiles[a]) == _home(profiles[b])
        if roommates or (a in circle_of and circle_of[a] == circle_of.get(b)):
            rels[(a, b)] = Relationship("roommate" if roommates else "friend", fw, aw)
        elif rng.random() < float(g["p_between"]):
            rels[(a, b)] = Relationship("acquaintance", float(g["familiarity_between"]), Relationship().affinity)
    # bridges: distinct agents, circles taken in a random round-robin, each with one strong tie elsewhere
    cids = list(circles)
    if len(cids) >= 2:
        order = [cids[int(i)] for i in rng.permutation(len(cids))]
        used: set[str] = set()
        for b in range(int(g.get("n_bridges", 0))):
            home = order[b % len(order)]
            cands = [a for a in circles[home] if a not in used]
            if not cands:
                continue
            a = cands[int(rng.integers(len(cands)))]
            others = [c for c in cids if c != home]
            tc = others[int(rng.integers(len(others)))]
            targets = [t for t in circles[tc] if rels.get(_key(a, t), Relationship()).familiarity < STRONG]
            if not targets:
                continue
            t = targets[int(rng.integers(len(targets)))]
            rels[_key(a, t)] = Relationship("friend", fw, aw)
            used.add(a)
    additions: dict[str, list[RoutineEntry]] = {}
    meals = _dinner_slots(cfg, circles) if g.get("shared_meals") and circles else []
    for slot in meals:
        for a in circles[slot["circle"]]:
            additions[a] = [RoutineEntry(time=slot["time"], location=slot["location"],
                                         activity=DINNER_ACTIVITY, arena=slot["arena"])]
    return {"circles": circles, "relationships": rels, "routine_additions": additions,
            "metrics": _metrics(ids, circles, rels, "generated", meals)}


def _with_meal(routine: list[RoutineEntry], adds: list[RoutineEntry]) -> list[RoutineEntry]:
    """Replace the agent's own dinner by the shared one, which lasts MEAL_MINUTES: the plan that falls
    inside the meal (the latest one; others are skipped that day), or else the next plan, starts right
    after it. Otherwise a member would linger at the venue until its next entry, mixing with the next
    circle's dinner."""
    out = [r for r in routine if "dinner" not in r.activity.lower()]
    for add in adds:
        t0 = _mins(add.time)
        t1 = t0 + MEAL_MINUTES
        during = [r for r in out if t0 <= _mins(r.time) < t1]
        after = [r for r in out if _mins(r.time) >= t1]
        nxt = max(during, key=lambda r: _mins(r.time)) if during else min(after, key=lambda r: _mins(r.time), default=None)
        out = [r for r in out if r not in during and r is not nxt]
        if nxt is not None:
            out.append(dataclasses.replace(nxt, time=_hhmm(t1)))
        out.append(dataclasses.replace(add))
    return sorted(out, key=lambda r: _mins(r.time))


def apply(profiles: dict[str, AgentProfile], topo: dict) -> None:
    """Rewrite profiles' relationships (all pairs, symmetric) and routines in place. Call before the
    agents (GA scratch, seed memories) and the world script (planned positions) are built."""
    rels = topo["relationships"]
    ids = sorted(profiles)
    for a in ids:
        # File friendships no longer describe the generated ties. Do not expose circle labels.
        profiles[a].friend_groups = []
        profiles[a].relationships = {b: dataclasses.replace(rels.get(_key(a, b), Relationship()))
                                     for b in ids if b != a}
    for aid, adds in topo.get("routine_additions", {}).items():
        prof = profiles[aid]
        prof.routine = _with_meal(prof.routine, adds)
        prof.known_locations = sorted(set(prof.known_locations) | {e.location for e in adds})


def metrics(profiles: dict[str, AgentProfile], circles: dict[str, list[str]], mode: str = "file") -> dict:
    """Topology metrics of the profiles' actual ties (for the manifest in either topology mode)."""
    ids = sorted(profiles)
    return _metrics(ids, circles, {(a, b): profiles[a].rel(b) for a, b in combinations(ids, 2)}, mode)


def _metrics(ids: list[str], circles: dict, rels: dict, mode: str, meals=()) -> dict:
    circle_of = {a: c for c, ms in circles.items() for a in ms}
    stranger = Relationship()

    def tie(r):
        return r is not None and r.relation_type != "stranger"

    within, between = [], []
    for a, b in combinations(ids, 2):
        same = a in circle_of and circle_of[a] == circle_of.get(b)
        (within if same else between).append(rels.get((a, b)) or stranger)

    def density(rs):
        return round(sum(tie(r) for r in rs) / len(rs), 4) if rs else None

    def mean_fam(rs):
        return round(sum(r.familiarity for r in rs) / len(rs), 4) if rs else None

    # familiarity-weighted Newman modularity of the circle partition (free agents are singletons)
    w = {k: r.familiarity for k, r in rels.items() if tie(r)}
    m = sum(w.values())
    modularity = None
    if m > 0:
        comm = {a: circle_of.get(a, f"free:{a}") for a in ids}
        deg = dict.fromkeys(ids, 0.0)
        inside: dict[str, float] = {}
        for (a, b), x in w.items():
            deg[a] += x
            deg[b] += x
            if comm[a] == comm[b]:
                inside[comm[a]] = inside.get(comm[a], 0.0) + x
        tot: dict[str, float] = {}
        for a in ids:
            tot[comm[a]] = tot.get(comm[a], 0.0) + deg[a]
        modularity = round(sum(inside.get(c, 0.0) / m - (d / (2 * m)) ** 2 for c, d in tot.items()), 4)
    bridges = [{"a": a, "b": b, "circles": [circle_of[a], circle_of[b]]}
               for (a, b), r in sorted(rels.items())
               if r is not None and r.familiarity >= STRONG and a in circle_of and b in circle_of
               and circle_of[a] != circle_of[b]]
    return {"mode": mode, "n_agents": len(ids), "circles": {c: list(ms) for c, ms in circles.items()},
            "free": [a for a in ids if a not in circle_of],
            "density": density(within + between), "density_within": density(within),
            "density_between": density(between), "mean_familiarity_within": mean_fam(within),
            "mean_familiarity_between": mean_fam(between), "modularity": modularity,
            "bridges": bridges, "n_bridges": len(bridges), "shared_meals": list(meals)}
