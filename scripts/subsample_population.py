#!/usr/bin/env python3
"""Cut a smaller, socially connected population out of a larger one, deterministically.

No network, LLM calls, or simulation runs. Reads a population file
(``agents`` / ``relationships`` / ``groups`` / ``circles``) plus the matching per-agent seed-memory
files, and writes the same three files restricted to a chosen subset.

Why not simply take the first N agents: the relationship graph is built over the whole source
population, so a prefix keeps only the few edges whose *both* endpoints happen to land inside it
(100 of homewood500 keeps 60 of 1,404 edges) and inherits whatever cohort mix the file order
happens to give. This script instead picks the subset.

Method (deterministic; ``--seed`` only breaks exact ties):

1. **Naming label.** Each agent's label comes from its primary seed memory: the first ``--split``
   label that occurs in the text, else the residual pool ``(other)``. A text carrying two declared
   labels is an error, because a silent misclassification would corrupt the split.
2. **Quotas.** Agents are bucketed by ``(demographics.category, demographics.year, label)``. Each
   declared label's target count is spread over its buckets by largest remainder, in proportion to
   the source bucket sizes; the residual pool gets ``N - sum(targets)``. So the naming split is
   exact and the category/year mix stays proportional to the source.
3. **Snowball.** Growth starts from the densest group cluster (a ``groups:`` entry, ranked by
   internal relationship edges per possible member pair) and then repeatedly admits the candidate
   with the most ties *into the set already chosen*, tie-broken by shared group memberships, then
   source degree, then a seeded per-agent key. Only candidates whose bucket still has quota are
   admissible, so growth cannot drift off the split. Because a greedy snowball is sensitive to
   where it starts, the growth is repeated once per candidate seed group in density order and the
   best-scoring result is kept (``--seed-groups 1`` keeps only the densest).
4. **Split repair.** Swap until every label's count equals its target, preferring a replacement
   from the same category and year so the demographic mix survives. This is a safety net: when a
   label follows the cohort (as in homewood500, where only first-years carry "Hopkins Cafe") step 2
   already pins the split and this pass reports zero swaps.
5. **Tie maximisation.** Best-improvement local search: repeatedly swap one selected agent for an
   unselected agent of the *same bucket* whenever that strictly increases the number of retained
   edges. Quotas and the split are invariant under such a swap.

The subset file keeps only relationships with both endpoints inside it, groups restricted to
selected members (dropped below two members), and the source circles restricted the same way
(dropped below ``backend.simulation.circles.MIN_MEMBERS``, so the result still validates).

Connectivity is bounded by the source graph and is reported, not assumed: a cohort that the source
never ties to anyone outside its own department cannot be connected to the rest by any subset.

``--meals`` (off by default, so an existing subset regenerates byte for byte) adds the sixth pass,
`assign_meals`: every selected agent gets a lunch and a dinner inside the two meal windows, at one of
the campus's two dining halls. See that function for the rule and why the hall is not a coin flip.

    .venv/bin/python scripts/subsample_population.py --n 100 --split "Hopkins Cafe=20" --seed 42 \\
        --meals --out configs/population/homewood100.yaml
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.simulation.circles import MIN_MEMBERS as MIN_CIRCLE_MEMBERS  # noqa: E402
from backend.simulation.world import ARENAS, WORLD_GRAPH, default_arena  # noqa: E402

OTHER = "(other)"          # residual pool: agents matching no declared --split label
MIN_GROUP_MEMBERS = 2      # backend.agents.profile.load_population drops smaller groups anyway
MIN_SEED_MEMBERS = 3       # a snowball seed must be a cluster, not a single tied pair
MAX_REPAIR_SWAPS = 1000    # loop guard; a repair needs at most one swap per mismatched agent


def parse_split(text: str) -> dict[str, int]:
    """'Hopkins Cafe=20' or 'Hopkins Cafe=20,FFC=80' -> {label: target count}, order preserved."""
    targets: dict[str, int] = {}
    for part in (p.strip() for p in (text or "").split(",")):
        if not part:
            continue
        label, sep, count = part.rpartition("=")
        label = label.strip()
        if not sep or not label:
            raise ValueError(f"--split entry {part!r} must read 'LABEL=COUNT'")
        if label == OTHER:
            raise ValueError(f"--split label {OTHER!r} is reserved for the residual pool")
        if label in targets:
            raise ValueError(f"--split names {label!r} twice")
        try:
            targets[label] = int(count)
        except ValueError:
            raise ValueError(f"--split entry {part!r} needs an integer count") from None
        if targets[label] < 0:
            raise ValueError(f"--split count for {label!r} must not be negative")
    return targets


def label_agents(memories: dict[str, list[str]], ids: list[str], labels: list[str]) -> dict[str, str]:
    """Agent id -> naming label, read from the primary seed memories (never the both-names file)."""
    out = {}
    for aid in ids:
        text = " ".join(memories.get(aid) or [])
        found = [label for label in labels if label in text]
        if len(found) > 1:
            raise ValueError(f"agent {aid!r}: seed memory carries several --split labels {found}; "
                             "classify against the primary seed file, not a both-names control")
        out[aid] = found[0] if found else OTHER
    return out


def largest_remainder(total: int, sizes: dict) -> dict:
    """Split `total` over keys in proportion to `sizes`; remainders go to the largest first."""
    base = sum(sizes.values())
    if base == 0:
        return {key: 0 for key in sizes}
    exact = {key: total * value / base for key, value in sizes.items()}
    out = {key: math.floor(value) for key, value in exact.items()}
    spare = total - sum(out.values())
    for key in sorted(sizes, key=lambda k: (-(exact[k] - out[k]), str(k)))[:spare]:
        out[key] += 1
    return out


def bucket_quotas(buckets: dict, targets: dict[str, int], n: int) -> dict:
    """Per-(category, year, label) quotas that hit every label target exactly and keep the mix."""
    declared = sum(targets.values())
    if declared > n:
        raise ValueError(f"--split asks for {declared} agents but --n is {n}")
    by_label = defaultdict(dict)
    for key, ids in buckets.items():
        by_label[key[2]][key] = len(ids)
    for label in targets:
        if label not in by_label:
            raise ValueError(f"no agent's seed memory contains the --split label {label!r}")
    if declared < n and OTHER not in by_label:
        raise ValueError(f"--split accounts for {declared} of {n} agents, but every source agent carries a "
                         "declared label, so the residual pool is empty; raise the counts or add the "
                         "remaining label(s) to --split")
    quotas = {}
    for label, sizes in by_label.items():
        wanted = targets.get(label, n - declared if label == OTHER else 0)
        if wanted > sum(sizes.values()):
            raise ValueError(f"only {sum(sizes.values())} source agents carry {label!r}, need {wanted}")
        quotas.update(largest_remainder(wanted, sizes))
    if sum(quotas.values()) != n:
        raise ValueError(f"quotas sum to {sum(quotas.values())}, expected {n}")
    return quotas


def build_graph(data: dict) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """-> (relationship adjacency, agent -> group ids). Undirected; an agent may be in many groups."""
    adjacency = defaultdict(set)
    for agent in data["agents"]:
        adjacency[agent["id"]] = set()
    for edge in data.get("relationships") or []:
        adjacency[edge["a"]].add(edge["b"])
        adjacency[edge["b"]].add(edge["a"])
    groups_of = defaultdict(set)
    for gid, gmembers in (data.get("groups") or {}).items():
        for member in gmembers:
            groups_of[member].add(gid)
    return dict(adjacency), dict(groups_of)


def internal_edges(adjacency: dict, ids) -> int:
    inside = set(ids)
    return sum(len(adjacency[a] & inside) for a in inside) // 2


def seed_order(data: dict, adjacency: dict) -> list[str]:
    """Group ids by internal edge density (edges per possible member pair), densest first.

    A pair is not a cluster: any two tied agents score 1.0, so groups below MIN_SEED_MEMBERS are
    ranked last rather than winning the seed slot on a single edge.
    """
    density = {}
    for gid, gmembers in data["groups"].items():
        size = len(set(gmembers))
        pairs = size * (size - 1) / 2
        density[gid] = (size >= MIN_SEED_MEMBERS,
                        internal_edges(adjacency, gmembers) / pairs if pairs else 0.0, size)
    return sorted(density, key=lambda gid: (not density[gid][0], -density[gid][1], -density[gid][2], gid))


def grow(seed_group, members, quotas, adjacency, groups_of, tiebreak, ids):
    """Snowball from one group: admit the candidate with the most ties into the set, quotas permitting.

    Returns (selected ids, number of admissions that had no tie into the set at all -- each of those
    starts a new component, which happens only when a quota can no longer be filled from the fringe).
    """
    bucket_of = {aid: key for key, bucket in members.items() for aid in bucket}
    selected: set[str] = set()
    used: Counter = Counter()
    group_hits: Counter = Counter()
    unconnected = 0

    def admit(aid):
        nonlocal unconnected
        if not adjacency[aid] & selected and selected:
            unconnected += 1
        selected.add(aid)
        used[bucket_of[aid]] += 1
        group_hits.update(groups_of.get(aid, ()))

    def admissible(aid):
        return aid not in selected and used[bucket_of[aid]] < quotas.get(bucket_of[aid], 0)

    total = sum(quotas.values())
    inside = set(seed_group)
    for aid in sorted(seed_group, key=lambda a: (-len(adjacency[a] & inside), tiebreak[a])):
        if len(selected) < total and admissible(aid):
            admit(aid)
    while len(selected) < total:
        candidates = [aid for aid in ids if admissible(aid)]
        if not candidates:
            raise ValueError("ran out of admissible agents; quotas exceed the source population")
        admit(min(candidates, key=lambda a: (-len(adjacency[a] & selected),
                                             -sum(group_hits[g] for g in groups_of.get(a, ())),
                                             -len(adjacency[a]), tiebreak[a])))
    return selected, unconnected


def repair_split(selected: set, labels: dict, targets: dict, demographics: dict,
                 adjacency: dict, tiebreak: dict, ids: list[str]) -> int:
    """Swap until every label count matches its target, preferring same category and year.

    Normally a no-op: `bucket_quotas` already pins the split when a label follows the cohort. It
    matters for source populations whose cohorts mix labels, where the snowball can finish inside
    its cohort quotas and still miss the naming targets.
    """
    def cohort(aid):
        return (demographics[aid].get("category"), demographics[aid].get("year"))

    swaps = 0
    while swaps < MAX_REPAIR_SWAPS:
        counts = Counter(labels[aid] for aid in selected)
        deltas = {label: counts.get(label, 0) - target for label, target in targets.items()}
        over = max((label for label in deltas if deltas[label] > 0), default=None,
                   key=lambda label: (deltas[label], label))
        under = min((label for label in deltas if deltas[label] < 0), default=None,
                    key=lambda label: (deltas[label], label))
        if over is None or under is None:
            return swaps
        drop = [aid for aid in selected if labels[aid] == over]
        take = [aid for aid in ids if aid not in selected and labels[aid] == under]
        if not take:
            raise ValueError(f"cannot reach the split: no unselected agent carries {under!r}")
        pairs = [(out, into) for out in drop for into in take]
        same = [pair for pair in pairs if cohort(pair[0]) == cohort(pair[1])]
        same = same or [pair for pair in pairs if cohort(pair[0])[0] == cohort(pair[1])[0]] or pairs
        out, into = min(same, key=lambda pair: (
            len(adjacency[pair[0]] & (selected - {pair[0]})) - len(adjacency[pair[1]] & (selected - {pair[0]})),
            tiebreak[pair[0]], tiebreak[pair[1]]))
        selected.discard(out)
        selected.add(into)
        swaps += 1
    raise ValueError("split repair did not converge")


def maximise_ties(selected: set, members: dict, adjacency: dict, tiebreak: dict) -> int:
    """Best-improvement swaps inside a bucket; quotas, mix and split are invariant. -> swaps applied."""
    bucket_of = {aid: key for key, bucket in members.items() for aid in bucket}
    swaps = 0
    while True:
        best, best_gain = None, 0
        for out in sorted(selected, key=lambda a: tiebreak[a]):
            rest = selected - {out}
            held = len(adjacency[out] & rest)
            for into in members[bucket_of[out]]:
                if into in selected:
                    continue
                gain = len(adjacency[into] & rest) - held
                if gain > best_gain:
                    best, best_gain = (out, into), gain
        if best is None:
            return swaps
        selected.discard(best[0])
        selected.add(best[1])
        swaps += 1


def components(selected: set, adjacency: dict) -> list[list[str]]:
    """Connected components of the retained relationship graph, largest first."""
    seen: set[str] = set()
    found = []
    for start in sorted(selected):
        if start in seen:
            continue
        stack, component = [start], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            seen.add(node)
            stack.extend(adjacency[node] & selected - component)
        found.append(sorted(component))
    return sorted(found, key=lambda c: (-len(c), c[0]))


# --- meals, and the hall split as an exposure manipulation -------------------------------------------------
# Everyone has to eat, and WHERE they eat is a manipulation rather than decoration: an incident sited in one
# dining hall is SEEN by the people who eat there and only HEARD ABOUT by the people who eat at the other one,
# which is the whole point of having a second hall (backend/simulation/world.py "Nolans").
#
# The split is `demographics.residence`, i.e. where the person lives, for three reasons:
#   * it is real and legible -- you eat at the hall on your own side of campus, the way a first-year in the
#     AMRs eats downstairs and someone commuting in up N Charles eats at the Commons hall;
#   * it runs ALONG the social graph instead of cutting randomly across it, so each hall is a sub-community
#     (a convention needs those) rather than a random half of the roster;
#   * it is nonetheless crossed by a large share of the relationships, so a phrase has somewhere to cross.
# Both of the last two are measured and reported in `summarise` -- they are properties of this population, not
# guarantees of the rule, and a different source population could need rebalancing.
#
# One exception, from the other fact the profile carries: if your working day happens AT or NEXT DOOR TO the
# other hall, you take the midday meal there and the evening meal at your own. Those people are the bridge --
# they can witness at one hall and carry it to the other. It is a rule, not a hand-placed set.
MEAL_HALLS = {"on campus": "Dining Hall", "off campus": "Nolans"}
LUNCH_WINDOW = ("11:30", "13:30")
DINNER_WINDOW = ("17:30", "19:00")
MEAL_MINUTES = 30          # two 15-minute ticks at the table: two chances to be paired into a conversation
SLOT_MINUTES = 15          # meal starts land on the tick grid, so nobody arrives mid-tick
# Dining staff keep the source generator's intent -- a break after the service peak, not in the middle of the
# student wave -- as far as a window that must contain everybody's lunch allows: they take the last slots.
LATE_LUNCH_DEPARTMENT = re.compile(r"dining service", re.I)
MEAL_ACTIVITY = re.compile(r"\blunch\b|\bdinner\b|\bmeal\b|having a meal|\beating\b", re.I)
LUNCH_ACTIVITY = {"student": "having lunch and making plans for the afternoon",
                  "faculty": "taking a lunch break", "staff": "taking a meal break"}
DINNER_ACTIVITY = {"student": "having dinner with whoever else is around",
                   "faculty": "having dinner before heading home",
                   "staff": "having dinner at the end of the shift"}
WIND_DOWN_ACTIVITY = "winding down at the end of the day"


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _mins(value) -> int:
    """'11:30' -> 690. YAML 1.1 parses an unquoted 08:30 as the integer 510, which is already minutes."""
    if isinstance(value, int):
        return value
    h, m = str(value).split(":")
    return int(h) * 60 + int(m)


def meal_slots(window: tuple[str, str]) -> list[int]:
    """Every tick-aligned start whose whole meal still fits in the window. Spread is not cosmetic: the engine
    charges `perception.crowd_factor` per other person in the ROOM, so a hall that seats its whole side at
    once (0.92 ** 50) makes each diner nearly blind to what happens there, which is where the grounded memes
    have to be witnessed. Dealing the slots round-robin rather than by a hash keeps the peak at the floor the
    window allows."""
    start, end = _mins(window[0]), _mins(window[1])
    return list(range(start, end - MEAL_MINUTES + 1, SLOT_MINUTES))


def deal_slots(ids: list[str], slots: list[int], seed: int, meal: str) -> dict[str, int]:
    """Spread `ids` evenly over `slots`: order them by their own per-agent key, then deal round-robin. The
    order is deterministic in (--seed, id, meal) and the balance is exact, where `key % len(slots)` would
    leave the peak to chance (it bunched the 64 midday diners into a 38-deep tick; round-robin gives 24)."""
    return {aid: slots[i % len(slots)]
            for i, aid in enumerate(sorted(ids, key=lambda a: (_key(seed, a, meal), a)))}


def routine_anchor(routine: list[dict]) -> str | None:
    """The place this routine spends the most minutes in: where the person's working day actually happens.

    The same notion backend/simulation/reference.py uses to decide what an agent calls its own building, and
    read the same way (a routine entry carries a start only, so the last one is given one hour).
    """
    spent: dict[str, int] = {}
    for i, e in enumerate(routine):
        start = _mins(e["time"])
        end = _mins(routine[i + 1]["time"]) if i + 1 < len(routine) else start + 60
        spent[e["location"]] = spent.get(e["location"], 0) + max(0, end - start)
    return max(sorted(spent), key=lambda k: spent[k]) if spent else None


def assign_meals(agents: list[dict], seed: int) -> dict:
    """Give every agent a lunch and a dinner at a dining hall, in place. -> counts for the summary.

    The existing meal entries are replaced, never added to, so an agent still eats once at midday and the
    entry it used to return to afterwards keeps its wording and simply moves to the new end of the meal.
    """
    halls = sorted(set(MEAL_HALLS.values()))
    for hall in halls:
        if hall not in WORLD_GRAPH:
            raise ValueError(f"meal hall {hall!r} is not a place in backend/simulation/world.py WORLD_GRAPH")
    # The halls are a manipulation only while they are matched: same number of rooms, so an agent's meal is
    # one room's worth of company at either, and a difference between them is exposure and not the venue.
    if len({len(ARENAS[hall]) for hall in halls}) != 1:
        raise ValueError(f"the dining halls are not matched on rooms: {({h: ARENAS[h] for h in halls})}")
    lunch_slots, dinner_slots = meal_slots(LUNCH_WINDOW), meal_slots(DINNER_WINDOW)
    if not lunch_slots or not dinner_slots:
        raise ValueError("a meal window is shorter than one meal")
    # "at or next door to" a hall, from the world graph itself, so a new edge needs no change here.
    near = {hall: {hall, *WORLD_GRAPH[hall]} for hall in halls}
    stats: Counter = Counter()
    per_hall: dict[str, Counter] = {hall: Counter() for hall in halls}
    hall_of: dict[str, str] = {}
    lunch_hall_of: dict[str, str] = {}
    late_of: dict[str, bool] = {}

    # 1. who eats where. Two facts about the person decide it and nothing else is consulted.
    for agent in agents:
        demographics = agent.get("demographics") or {}
        residence = demographics.get("residence")
        if residence not in MEAL_HALLS:
            raise ValueError(f"agent {agent['id']!r}: demographics.residence {residence!r} has no dining hall; "
                             f"expected one of {sorted(MEAL_HALLS)}")
        home_hall = MEAL_HALLS[residence]
        other = next(h for h in halls if h != home_hall)
        anchor = routine_anchor(agent["routine"])
        hall_of[agent["id"]] = home_hall
        lunch_hall_of[agent["id"]] = other if anchor in near[other] else home_hall
        late_of[agent["id"]] = bool(LATE_LUNCH_DEPARTMENT.search(str(demographics.get("department") or "")))
        stats["bridge_agents"] += lunch_hall_of[agent["id"]] != home_hall
        per_hall[home_hall]["evening"] += 1
        per_hall[lunch_hall_of[agent["id"]]]["midday"] += 1

    # 2. when. Dealt per sitting -- each hall's own queue, and the late shift separately -- so that it is the
    #    room an agent is actually in that is level, not the campus's meal load in aggregate.
    sittings: dict[tuple[str, str, bool], list[str]] = defaultdict(list)
    for agent in agents:
        aid = agent["id"]
        sittings[("lunch", lunch_hall_of[aid], late_of[aid])].append(aid)
        sittings[("dinner", hall_of[aid], False)].append(aid)
    start_of: dict[tuple[str, str], int] = {}
    for (meal, _hall, late), ids in sorted(sittings.items()):
        slots = dinner_slots if meal == "dinner" else (lunch_slots[-2:] if late else lunch_slots)
        for aid, minute in deal_slots(ids, slots, seed, meal).items():
            start_of[(aid, meal)] = minute

    # 3. rewrite each routine. The old meal entries are replaced, never added to, so an agent still eats once
    #    at midday, and the entry it used to go back to afterwards keeps its wording and just moves.
    for agent in agents:
        aid = agent["id"]
        category = (agent.get("demographics") or {}).get("category")
        routine = list(agent["routine"])
        old = [i for i, e in enumerate(routine) if MEAL_ACTIVITY.search(e.get("activity") or "")]
        # The entry right after the old meal is what this person goes back to; keep its wording, move its time.
        resume = routine[old[0] + 1] if old and old[0] + 1 < len(routine) else None
        routine = [e for i, e in enumerate(routine) if i not in set(old)]
        lunch_at, dinner_at = start_of[(aid, "lunch")], start_of[(aid, "dinner")]
        clash = [e for e in routine if lunch_at < _mins(e["time"]) < lunch_at + MEAL_MINUTES and e is not resume]
        if clash:
            raise ValueError(f"agent {aid!r}: routine entry {clash[0]} falls inside the lunch window; "
                             "the meal pass would silently drop it")
        home = agent.get("home") or {}
        meals = [_entry(lunch_at, lunch_hall_of[aid], LUNCH_ACTIVITY.get(category, LUNCH_ACTIVITY["student"])),
                 _entry(dinner_at, hall_of[aid], DINNER_ACTIVITY.get(category, DINNER_ACTIVITY["student"])),
                 {"time": _hhmm(dinner_at + MEAL_MINUTES), "location": home["location"],
                  "arena": home["arena"], "activity": WIND_DOWN_ACTIVITY}]
        if resume is not None:
            resume["time"] = _hhmm(lunch_at + MEAL_MINUTES)
        # Last entry wins if the same minute occurs twice, exactly as the source generator resolves it.
        agent["routine"] = sorted({_mins(e["time"]): e for e in routine + meals}.values(),
                                  key=lambda e: _mins(e["time"]))
        stats["agents_with_lunch"] += 1
        stats["agents_with_dinner"] += 1
    return {"lunch_slots": [_hhmm(s) for s in lunch_slots], "dinner_slots": [_hhmm(s) for s in dinner_slots],
            "meal_minutes": MEAL_MINUTES, "counts": dict(stats), "hall_of": hall_of,
            "by_hall": {hall: dict(per_hall[hall]) for hall in halls},
            "peak_in_room_per_tick": hall_occupancy(agents, halls)}


def _entry(minutes: int, location: str, activity: str) -> dict:
    return {"time": _hhmm(minutes), "location": location, "arena": default_arena(location), "activity": activity}


def _key(seed: int, agent_id: str, name: str) -> int:
    """A per-agent, per-mechanism draw. Deterministic in (--seed, id, mechanism) and nothing else, so adding
    a meal never perturbs any other choice the script makes."""
    return int.from_bytes(hashlib.blake2b(f"{seed}:{agent_id}:{name}".encode(), digest_size=8).digest(), "big")


def hall_occupancy(agents: list[dict], halls) -> dict:
    """Peak head-count in each hall in any 15-minute tick, per meal. The engine's attention model charges a
    crowd factor for every other person in the ROOM, so how bunched the slots are is a measurable cost of
    the meal design, not a detail: reported so it can be traded against co-location."""
    peaks: dict[str, dict[str, int]] = {}
    for hall in halls:
        peaks[hall] = {}
        for label, window in (("midday", LUNCH_WINDOW), ("evening", DINNER_WINDOW)):
            best = 0
            for tick in range(_mins(window[0]), _mins(window[1]), SLOT_MINUTES):
                here = 0
                for agent in agents:
                    for e in agent["routine"]:
                        start = _mins(e["time"])
                        if (e["location"] == hall and MEAL_ACTIVITY.search(e.get("activity") or "")
                                and start <= tick < start + MEAL_MINUTES):
                            here += 1
                best = max(best, here)
            peaks[hall][label] = best
    return peaks


def select(data: dict, memories: dict, n: int, targets: dict[str, int], seed: int,
           seed_groups: int) -> tuple[list[str], dict]:
    """-> (selected ids in source order, provenance of the selection)."""
    ids = [agent["id"] for agent in data["agents"]]
    if len(ids) != len(set(ids)):
        raise ValueError("source population contains duplicate agent ids")
    if n > len(ids):
        raise ValueError(f"--n {n} exceeds the {len(ids)} agents in the source population")
    demographics = {agent["id"]: agent.get("demographics") or {} for agent in data["agents"]}
    missing = [aid for aid in ids if aid not in memories]
    if missing:
        raise ValueError(f"{len(missing)} source agents have no seed memory, e.g. {missing[:3]}")
    labels = label_agents(memories, ids, list(targets))
    members = defaultdict(list)
    for aid in ids:
        members[(demographics[aid].get("category"), demographics[aid].get("year"), labels[aid])].append(aid)
    quotas = bucket_quotas(members, targets, n)
    # Per-label targets including the residual pool, so the repair pass sees the whole picture.
    label_targets: Counter = Counter()
    for key, quota in quotas.items():
        label_targets[key[2]] += quota
    adjacency, groups_of = build_graph(data)
    tiebreak = {aid: hashlib.blake2b(f"{seed}:{aid}".encode(), digest_size=8).hexdigest() for aid in ids}

    candidates = seed_order(data, adjacency)[: seed_groups or None]
    if not candidates:
        raise ValueError("the source population has no groups to snowball from")
    best = None
    for rank, gid in enumerate(candidates):
        grown, unconnected = grow(data["groups"][gid], members, quotas, adjacency, groups_of, tiebreak, ids)
        repairs = repair_split(grown, labels, label_targets, demographics, adjacency, tiebreak, ids)
        swaps = maximise_ties(grown, members, adjacency, tiebreak)
        score = (-internal_edges(adjacency, grown), rank, gid)
        if best is None or score < best[0]:
            best = (score, gid, grown, {"split_repair_swaps": repairs, "tie_maximising_swaps": swaps,
                                        "unconnected_additions": unconnected})
    _, gid, selected, provenance = best
    return [aid for aid in ids if aid in selected], {"seed_group": gid, "labels": labels,
                                                     "quotas": quotas, "adjacency": adjacency, **provenance}


def subset_population(data: dict, selected: list[str]) -> dict:
    """The same file shape, restricted: relationships with both ends inside, groups and circles pruned."""
    inside = set(selected)
    agents = [agent for agent in data["agents"] if agent["id"] in inside]
    relationships = [edge for edge in (data.get("relationships") or [])
                     if edge["a"] in inside and edge["b"] in inside]
    groups = {gid: [m for m in gmembers if m in inside]
              for gid, gmembers in (data.get("groups") or {}).items()}
    groups = {gid: gmembers for gid, gmembers in sorted(groups.items())
              if len(gmembers) >= MIN_GROUP_MEMBERS}
    circles = {cid: [m for m in cmembers if m in inside]
               for cid, cmembers in (data.get("circles") or {}).items()}
    circles = {cid: cmembers for cid, cmembers in sorted(circles.items())
               if len(cmembers) >= MIN_CIRCLE_MEMBERS}
    return {"agents": agents, "relationships": relationships, "groups": groups, "circles": circles}


def meal_summary(subset: dict, meals: dict) -> dict:
    """What the hall split is worth as an exposure manipulation: the two sides' sizes, how well connected each
    side is to the other, and whether either side is a pile of fragments. A split nothing crosses measures
    nothing, so the crossing count is reported next to the split and not left to be assumed."""
    hall_of = meals["hall_of"]
    edges = [e for e in subset["relationships"] if e["a"] in hall_of and e["b"] in hall_of]
    crossing = [e for e in edges if hall_of[e["a"]] != hall_of[e["b"]]]
    degree: Counter = Counter()
    for e in edges:
        degree[e["a"]] += 1
        degree[e["b"]] += 1
    adjacency = defaultdict(set)
    for e in edges:
        adjacency[e["a"]].add(e["b"])
        adjacency[e["b"]].add(e["a"])
    out = {k: v for k, v in meals.items() if k != "hall_of"}
    out["split"] = dict(sorted(Counter(hall_of.values()).items()))
    out["crossing_relationships"] = len(crossing)
    out["crossing_fraction"] = round(len(crossing) / len(edges), 4) if edges else None
    out["by_side"] = {}
    for hall in sorted(set(hall_of.values())):
        side = [aid for aid in hall_of if hall_of[aid] == hall]
        inside = set(side)
        reach = [aid for aid in side if adjacency[aid] - inside]
        degrees = sorted(degree[aid] for aid in side)
        out["by_side"][hall] = {
            "agents": len(side),
            "mean_degree": round(sum(degrees) / len(side), 3) if side else None,
            "degree_min_median_max": [degrees[0], degrees[len(degrees) // 2], degrees[-1]] if degrees else None,
            "with_a_tie_to_the_other_hall": len(reach),
            "internal_component_sizes": [len(c) for c in components(inside, adjacency)][:5],
        }
    return out


def summarise(data: dict, subset: dict, selected: list[str], provenance: dict, args,
              meals: dict | None = None) -> dict:
    """Counts only. Nothing here describes conversation, adoption, or any cultural outcome."""
    adjacency = provenance["adjacency"]
    labels = provenance["labels"]
    ids = [agent["id"] for agent in data["agents"]]
    demographics = {agent["id"]: agent.get("demographics") or {} for agent in data["agents"]}
    n = len(selected)
    prefix = set(ids[:n])
    kept = len(subset["relationships"])
    prefix_edges = internal_edges(adjacency, prefix)
    parts = components(set(selected), adjacency)
    source_edges = len(data.get("relationships") or [])
    return {
        "schema_version": 1,
        "selection": "deterministic quota-constrained snowball, split repair, tie-maximising swaps",
        "source_population": str(args.population),
        "source_memories": str(args.memories),
        "seed": args.seed,
        "seed_group": provenance["seed_group"],
        "seed_groups_tried": args.seed_groups or len(data["groups"]),
        "split_repair_swaps": provenance["split_repair_swaps"],
        "tie_maximising_swaps": provenance["tie_maximising_swaps"],
        "unconnected_additions": provenance["unconnected_additions"],
        "source_agents": len(ids),
        "selected_agents": n,
        "naming_split": dict(sorted(Counter(labels[aid] for aid in selected).items())),
        "naming_split_source": dict(sorted(Counter(labels[aid] for aid in ids).items())),
        "categories": dict(sorted(Counter(demographics[aid].get("category") for aid in selected).items())),
        "categories_source_share": {key: round(value / len(ids), 4) for key, value in
                                    sorted(Counter(demographics[aid].get("category") for aid in ids).items())},
        "years": dict(sorted(Counter(demographics[aid].get("year") for aid in selected).items())),
        "relationships": {"source": source_edges, "retained": kept,
                          "retained_fraction": round(kept / source_edges, 4) if source_edges else None,
                          "first_n_prefix_baseline": prefix_edges},
        "mean_degree": {"subset": round(2 * kept / n, 3) if n else None,
                        "source": round(2 * source_edges / len(ids), 3),
                        "first_n_prefix_baseline": round(2 * prefix_edges / n, 3) if n else None},
        "components": {"count": len(parts), "sizes": [len(part) for part in parts],
                       "largest": len(parts[0]) if parts else 0},
        "groups": {"source": len(data.get("groups") or {}), "kept": len(subset["groups"]),
                   "dropped_below_2_members": len(data.get("groups") or {}) - len(subset["groups"])},
        "circles": {"source": len(data.get("circles") or {}), "kept": len(subset["circles"])},
        **({"meals": meal_summary(subset, meals)} if meals else {}),
        "limitations": [
            "Counts describe a synthetic population, not measured campus behaviour.",
            "Retained ties and components bound who could meet through the designed structure; "
            "they do not predict contact, conversation, or naming outcomes.",
            "A cohort the source never ties outside its own department cannot be connected by any subset.",
        ],
    }


def leading_comments(text: str) -> str:
    """The `#` block a file opens with, or "". A subset of a file is still that file's data, so the note that
    says what is in it and why belongs on the cut too -- `yaml.safe_load` drops comments, so it is carried
    across by hand. It is read from the SOURCE, never from whatever happens to be at `--out`, or regenerating
    into an empty directory would not reproduce the committed bytes."""
    kept = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        kept.append(line)
    return "".join(line + "\n" for line in kept)


def write_yaml(path: Path, data, header: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110),
                    encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--population", type=Path, default=Path("configs/population/homewood500.yaml"))
    parser.add_argument("--memories", type=Path,
                        default=Path("configs/population/homewood500_initial_memories.yaml"),
                        help="primary seed memories; the naming label is read from these")
    parser.add_argument("--both-names-memories", default="configs/population/homewood500_both_names_memories.yaml",
                        help="optional recognition-control seeds; pass '' to skip")
    parser.add_argument("--n", type=int, required=True, help="agents to select")
    parser.add_argument("--split", default="", help='exact naming targets, e.g. "Hopkins Cafe=20"')
    parser.add_argument("--seed", type=int, default=42, help="breaks exact ties only")
    parser.add_argument("--meals", action="store_true",
                        help="give every selected agent a lunch and a dinner at a dining hall, and split the "
                             "halls by residence (see assign_meals). Off by default: without it a subset "
                             "regenerates byte for byte as it did before this pass existed.")
    parser.add_argument("--seed-groups", type=int, default=0,
                        help="candidate snowball seeds to try, densest first (0 = every group)")
    parser.add_argument("--out", type=Path, required=True, help="population YAML to write")
    parser.add_argument("--out-memories", type=Path, help="default: <out>_initial_memories.yaml")
    parser.add_argument("--out-both-names", type=Path, help="default: <out>_both_names_memories.yaml")
    parser.add_argument("--summary", type=Path, help="also write the printed summary here as JSON")
    args = parser.parse_args(argv)

    def resolve(path):
        path = Path(path)
        return path if path.is_absolute() else ROOT / path

    stem = args.out.with_suffix("")
    args.out_memories = args.out_memories or Path(f"{stem}_initial_memories.yaml")
    args.out_both_names = args.out_both_names or Path(f"{stem}_both_names_memories.yaml")
    outputs = {resolve(args.out), resolve(args.out_memories), resolve(args.out_both_names)}
    if outputs & {resolve(args.population), resolve(args.memories)}:
        parser.error("--out paths must not overwrite an input file")

    try:
        targets = parse_split(args.split)
        population_text = resolve(args.population).read_text(encoding="utf-8")
        memories_text = resolve(args.memories).read_text(encoding="utf-8")
        data = yaml.safe_load(population_text)
        memories = yaml.safe_load(memories_text) or {}
        selected, provenance = select(data, memories, args.n, targets, args.seed, args.seed_groups)
        subset = subset_population(data, selected)
        meals = assign_meals(subset["agents"], args.seed) if args.meals else None
        write_yaml(resolve(args.out), subset, leading_comments(population_text))
        write_yaml(resolve(args.out_memories), {aid: memories[aid] for aid in selected},
                   leading_comments(memories_text))
        if args.both_names_memories:
            both_text = resolve(args.both_names_memories).read_text(encoding="utf-8")
            both = yaml.safe_load(both_text) or {}
            absent = [aid for aid in selected if aid not in both]
            if absent:
                raise ValueError(f"{len(absent)} selected agents are missing from the both-names file")
            write_yaml(resolve(args.out_both_names), {aid: both[aid] for aid in selected},
                       leading_comments(both_text))
        summary = summarise(data, subset, selected, provenance, args, meals)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"subsampling failed: {exc}\n")
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    if args.summary:
        resolve(args.summary).write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
