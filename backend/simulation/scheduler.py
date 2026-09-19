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
    return plan


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
