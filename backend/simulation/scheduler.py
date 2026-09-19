"""Lightweight daily routines with jitter and small random deviations.

GA generates daily plans with the LLM (wake-up hour, hourly schedule, task
decomposition). To save LLM calls we plan deterministically from the profile's
routine and let *events*, *invitations* and *reaction decisions* re-plan.
"""
from __future__ import annotations

from backend.simulation.world import ARENAS, default_arena

DETOURS = {
    "Cafe": "grabbing a snack", "Quad": "taking a walk", "Library": "picking up a book",
    "Dining Hall": "getting a quick bite", "Gym": "stretching", "Dorm": "dropping things off",
    "Classroom": "checking the bulletin board", "Research Lab": "checking on an experiment",
}


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
    return shift_hook(agent, clock, plan)


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
