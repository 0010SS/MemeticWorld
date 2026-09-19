"""Lightweight daily routines with jitter, small random deviations and campus errands.

GA generates daily plans with the LLM (wake-up hour, hourly schedule, task
decomposition). To save LLM calls we plan deterministically from the profile's
routine and let *events*, *invitations* and *reaction decisions* re-plan.

Errands (`routine.errands_per_day`, `routine.errand_places`): short everyday trips to public campus places
(the mailroom, the shuttle stop, the club board, the track) slotted into free gaps of the routine. They never
displace a routine entry: an errand interrupts an interruptible entry (not a class, meal, rehearsal, shift, or
the wake-up / wind-down) somewhere in its middle, and the agent resumes that entry afterwards. Errands are drawn
from the same `rng` as the routine, after every routine draw (so the routine part of a plan is unchanged), and
the world script plans through this same function, so the engine and the world script agree on every errand.
"""
from __future__ import annotations

import re

from backend.simulation.world import ARENAS, PUBLIC_PLACES, _minutes, default_arena

DETOURS = {
    "Cafe": "grabbing a snack", "Quad": "taking a walk", "Library": "picking up a book",
    "Dining Hall": "getting a quick bite", "Gym": "stretching", "Dorm": "dropping things off",
    "Classroom": "checking the bulletin board", "Research Lab": "checking on an experiment",
    # campus places: the detour lands in the default arena (ARENAS[loc][0])
    "Student Center": "hanging out in the common room", "Auditorium": "catching part of a talk",
    "Engineering Hall": "printing something in the computer lab", "Science Hall": "dropping off a lab notebook",
    "Museum": "looking around the gallery", "Theater": "watching a rehearsal", "Apartments": "visiting a friend",
    "Athletic Center": "shooting hoops", "Admin Building": "dropping off a form at the front desk",
    "Shuttle Stop": "waiting for the shuttle",
}

# place -> [(arena, activity)]: the everyday errands each public place affords
ERRANDS = {
    "Student Center": [("Club Room", "checking the club board"), ("Common Room", "grabbing a snack in the common room"),
                       ("Club Room", "dropping by the club room"), ("Common Room", "printing a flyer")],
    "Auditorium": [("Lobby", "picking up tickets for a talk"), ("Main Hall", "catching part of a guest lecture"),
                   ("Lobby", "looking at the event posters")],
    "Engineering Hall": [("Computer Lab", "printing a lab report"), ("Computer Lab", "using a lab computer"),
                         ("Study Lounge", "asking a friend about a homework problem")],
    "Science Hall": [("Lecture Room", "going to office hours"), ("Teaching Lab", "dropping off a lab notebook"),
                     ("Teaching Lab", "checking the teaching lab schedule")],
    "Museum": [("Gallery", "looking around the gallery"), ("Garden", "walking through the museum garden")],
    "Theater": [("Stage", "watching a rehearsal"), ("Green Room", "dropping off a costume"),
                ("Stage", "helping move scenery")],
    "Athletic Center": [("Track", "running on the track"), ("Field House", "shooting hoops in the field house"),
                        ("Track", "walking a few laps")],
    "Admin Building": [("Mailroom", "picking up a package at the mailroom"), ("Mailroom", "checking the mailbox"),
                       ("Front Desk", "dropping off a form at the front desk")],
    "Shuttle Stop": [("Bench", "waiting for the shuttle"), ("Bench", "checking when the next shuttle comes")],
    "Cafe": [("Counter", "grabbing a coffee to go")],
    "Library": [("Study Tables", "returning a library book"), ("Quiet Floor", "looking for a book")],
    "Quad": [("Lawn", "cutting across the quad")],
}
ERRAND_DEFAULTS = {"errands_per_day": 2, "errand_minutes": [30, 45], "errand_window": ["08:00", "21:00"]}
# routine entries an errand never interrupts: classes, meals, rehearsals / practice, shifts, tutoring, planned
# meet-ups ("... with Sam", "meeting up with ..."; they co-locate people on purpose), waking up and winding down
PROTECTED_PLACES = {"Classroom", "Dining Hall"}
PROTECTED = re.compile(r"\battending\b|lecture|seminar|workshop|tutoring|shift|\beat|lunch|dinner|breakfast|meal|"
                       r"sandwich|rehears|practic|meeting|\bwith\b|sleep|winding down|getting ready", re.I)


def _to_min(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def plan_day(agent, clock, rng) -> list[dict]:
    rc = agent.cfg["routine"]
    start = int(clock.day_start.total_seconds() // 60)
    tm = clock.tick_minutes
    plan = []
    for i, r in enumerate(agent.profile.routine):
        m = _to_min(r.time) + int(rng.integers(-rc["jitter_minutes"], rc["jitter_minutes"] + 1)) if i else _to_min(r.time)
        k = max(0, round((m - start) / tm))
        loc, arena, act = r.location, r.arena or default_arena(r.location), r.activity
        if i and rng.random() < rc["deviation_prob"]:
            alts = [x for x in agent.profile.known_locations if x != loc]
            loc = alts[int(rng.integers(len(alts)))]
            arena, act = default_arena(loc), DETOURS.get(loc, "taking a break")
        if arena not in ARENAS[loc]:
            arena = default_arena(loc)
        plan.append({"k": k, "location": loc, "arena": arena, "activity": act})
    plan.sort(key=lambda e: e["k"])
    plan = add_errands(agent, clock, plan, rng)
    return shift_hook(agent, clock, plan)


def protected(entry: dict) -> bool:
    """Classes, meals, rehearsals / practice, shifts, tutoring, planned meet-ups, waking up and winding down are
    never interrupted."""
    return entry["location"] in PROTECTED_PLACES or bool(PROTECTED.search(entry["activity"]))


def errand_places(cfg: dict) -> list[str]:
    places = list((cfg.get("routine") or {}).get("errand_places") or PUBLIC_PLACES)
    bad = [p for p in places if p not in ARENAS]
    if bad:
        raise ValueError(f"routine.errand_places: unknown places {bad}")
    return places


def errand_options(place: str) -> list[tuple[str, str]]:
    return ERRANDS.get(place) or [(default_arena(place), DETOURS.get(place, "running an errand"))]


def _role_windows(agent, clock) -> list[tuple[int, int]]:
    """The co-op role's shift windows (whether or not the roster is on), so the roster overlay never cuts an errand
    and a plan with the roster on is exactly the roster-off plan with the shifts laid over it."""
    role = getattr(agent.profile, "role", None)
    if not role:
        return []
    from backend.simulation.roster import shift_ticks
    return shift_ticks(agent.cfg, role, clock)


def add_errands(agent, clock, plan: list[dict], rng) -> list[dict]:
    """Insert up to `routine.errands_per_day` errands into free gaps of a sorted routine plan.

    An errand of d ticks starts at s inside the span [k_i, k_i+1) of an interruptible entry i, with s > k_i (the
    agent first gets there) and s + d < k_i+1 (the agent is back before the next entry); at s + d the entry is
    resumed. Errands avoid the role's shift windows, the `routine.errand_window` hours and each other (one tick
    apart). Every errand consumes exactly one rng.random(4) draw whether or not it fits, so the stream is fixed."""
    rc = {**ERRAND_DEFAULTS, **{k: v for k, v in (agent.cfg.get("routine") or {}).items() if v is not None}}
    n = int(rc["errands_per_day"])
    if n <= 0 or not plan:
        return plan
    tm = clock.tick_minutes
    start = int(clock.day_start.total_seconds() // 60)
    tpd = clock.ticks_per_day
    lo_min, hi_min = (int(x) for x in rc["errand_minutes"])
    d_lo = max(1, round(lo_min / tm))
    d_hi = max(d_lo, round(hi_min / tm))
    w0, w1 = (max(0, round((_minutes(t) - start) / tm)) for t in rc["errand_window"])   # YAML may give minutes
    places = [p for p in errand_places(agent.cfg) if p in set(agent.profile.known_locations)]
    windows = _role_windows(agent, clock)
    spans = [(e["k"], plan[i + 1]["k"] if i + 1 < len(plan) else tpd, e) for i, e in enumerate(plan)]
    taken: list[tuple[int, int]] = []          # (start, resume) of the errands placed so far
    used: list[str] = []
    out = list(plan)
    for u in rng.random((n, 4)):
        d = d_lo + min(int(u[0] * (d_hi - d_lo + 1)), d_hi - d_lo)
        slots = []
        for k0, k1, e in spans:
            if protected(e):
                continue
            for s in range(max(k0 + 1, w0), min(k1 - 1, w1) - d + 1):
                if any(s <= ke and s + d >= ks for ks, ke in windows):
                    continue
                if any(s <= b + 1 and s + d >= a - 1 for a, b in taken):
                    continue
                slots.append((s, e))
        cands = [p for p in places if p not in used] or places
        if not slots or not cands:
            continue
        s, e = slots[min(int(u[1] * len(slots)), len(slots) - 1)]
        cands = [p for p in cands if p != e["location"]] or [p for p in places if p != e["location"]]
        if not cands:
            continue
        place = cands[min(int(u[2] * len(cands)), len(cands) - 1)]
        opts = errand_options(place)
        arena, act = opts[min(int(u[3] * len(opts)), len(opts) - 1)]
        if arena not in ARENAS[place]:
            arena = default_arena(place)
        out.append({"k": s, "location": place, "arena": arena, "activity": act})
        out.append({"k": s + d, "location": e["location"], "arena": e["arena"], "activity": e["activity"]})
        taken.append((s, s + d))
        used.append(place)
    out.sort(key=lambda x: x["k"])
    return out


def shift_hook(agent, clock, plan: list[dict]) -> list[dict]:
    """v3 plan hook (ontology v3 §1.6): when `roster.enabled` and the profile has a co-op role, the role's shift
    entries replace the overlapping routine entries. Runs AFTER every routine draw, so it never shifts the
    plan stream (common random numbers). No-op otherwise (v2 behaviour)."""
    role = getattr(agent.profile, "role", None)
    if not role or not ((agent.cfg.get("roster") or {}).get("enabled")):
        return plan
    from backend.simulation.roster import ROLE_ACTIVITY, ROLE_PLACE, shift_ticks
    loc, arena = ROLE_PLACE[role]
    return apply_shifts(plan, shift_ticks(agent.cfg, role, clock),
                        {"location": loc, "arena": arena, "activity": ROLE_ACTIVITY[role]},
                        agent.profile.home)


def apply_shifts(plan: list[dict], windows: list[tuple[int, int]], entry: dict, home: dict | None = None) -> list[dict]:
    """Overlay shift windows [(k_start, k_end), ...] (k_end exclusive) on a routine plan: routine entries
    starting inside a window are dropped, the shift entry starts at k_start, and at k_end the routine resumes
    where it would have been (the last routine entry at or before k_end), unless an entry starts exactly then."""
    routine = sorted(plan, key=lambda e: e["k"])
    out = list(routine)
    for ks, ke in windows:
        out = [e for e in out if not (ks <= e["k"] < ke)]
        out.append({"k": ks, **entry})
        if not any(e["k"] == ke for e in out):
            prev = [e for e in routine if e["k"] <= ke and not (ks <= e["k"] < ke)]
            inside = [e for e in routine if ks <= e["k"] < ke]
            src = (inside or prev or [None])[-1]
            if src is None:
                h = home or {}
                src = {"location": h.get("location"), "arena": h.get("arena"), "activity": "sleeping"}
            out.append({"k": ke, "location": src["location"], "arena": src["arena"], "activity": src["activity"]})
    out.sort(key=lambda e: e["k"])
    return out


def routine_position(agent, k_in_day: int) -> dict:
    cur = {"k": 0, "location": agent.profile.home["location"], "arena": agent.profile.home["arena"],
           "activity": "sleeping"}
    for e in agent.day_plan:
        if e["k"] <= k_in_day:
            cur = e
        else:
            break
    return cur


def next_entry(agent, k_in_day: int) -> dict | None:
    for e in agent.day_plan:
        if e["k"] > k_in_day:
            return e
    return None
