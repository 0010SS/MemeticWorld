"""A MemeWorld agent = profile + upstream GA Scratch (short-term state) + GA
AssociativeMemory (memory stream), plus hooks for experimental modules.

The upstream prompt functions access `persona.name`, `persona.scratch.*` and
`persona.a_mem.*`; this class exposes exactly those.
"""
from __future__ import annotations

import datetime as dt
import zlib
from dataclasses import dataclass, field

import numpy as np

from backend import ga_compat
from backend.agents.profile import AgentProfile
from backend.memory.store import MemoryStream

ga = ga_compat.load()

REL_PHRASES = {
    "roommate": "roommates", "friend": "friends", "classmate": "classmates", "labmate": "labmates",
    "clubmate": "in the same club", "acquaintance": "acquaintances", "stranger": "strangers",
}


def familiarity_phrase(f: float) -> str:
    if f >= 0.85:
        return "know each other very well"
    if f >= 0.55:
        return "know each other fairly well"
    if f >= 0.25:
        return "know each other a little"
    return "barely know each other"


def affinity_phrase(a: float) -> str:
    if a >= 0.7:
        return "get along well"
    if a >= 0.45:
        return "get along fine"
    return "don't click much"


@dataclass
class AgentState:
    location: str
    arena: str
    activity: str
    current_goal: str = ""
    path: list = field(default_factory=list)          # locations traversed this tick (for animation)
    in_conversation: str | None = None                # conversation id
    pre_chat_activity: str | None = None
    forced_by: str | None = None                      # world event forcing this position (sim-only)
    talks_today: int = 0
    last_talk: dict = field(default_factory=dict)     # other_id -> datetime


class CampusMaze:
    """Minimal stand-in for the upstream `Maze`: only access_tile() is used by the
    upstream chat prompt ('<arena> in <sector>'). Our tiles are (location, arena)."""

    def access_tile(self, tile):
        loc, arena = tile
        return {"world": "the Homewood campus", "sector": loc, "arena": arena, "game_object": ""}


class Agent:
    def __init__(self, profile: AgentProfile, cfg: dict, mods, seed: int):
        self.profile = profile
        self.id = profile.id
        self.name = profile.name
        self.cfg = cfg
        self.mods = mods
        self.rng = np.random.default_rng([seed, zlib.crc32(profile.id.encode())])
        self.a_mem = MemoryStream(profile.id)
        sc = ga.Scratch("__memeworld_no_file__")
        sc.name = profile.name
        sc.first_name, sc.last_name = profile.name.split(" ", 1)
        sc.age = profile.demographics.get("age")
        sc.innate = profile.ga_innate()
        sc.learned = profile.ga_learned()
        sc.currently = ""
        sc.lifestyle = profile.ga_lifestyle()
        sc.living_area = f"{profile.home['location']}:{profile.home['arena']}"
        sc.daily_plan_req = profile.ga_daily_plan_req()
        sc.importance_trigger_max = cfg["reflection"]["importance_threshold"]
        sc.importance_trigger_curr = sc.importance_trigger_max
        sc.importance_ele_n = 0
        sc.planned_path = []
        sc.chat = []
        sc.act_description = "getting ready"
        self.scratch = sc
        self.state = AgentState(location=profile.home["location"], arena=profile.home["arena"],
                                activity="sleeping")
        self.day_plan: list[dict] = []

    # --- Generative-Agents-facing views --------------------------------------
    def set_time(self, t: dt.datetime):
        self.scratch.curr_time = t

    def sync_scratch(self):
        s = self.state
        self.scratch.curr_tile = (s.location, s.arena)
        self.scratch.act_description = s.activity
        self.scratch.currently = s.current_goal or f"{self.profile.first_name} is {s.activity}."

    def iss(self) -> str:
        return self.scratch.get_str_iss()

    @property
    def memory_stream(self):
        return self.a_mem.all_nodes()

    @property
    def reflections(self):
        return list(self.a_mem.seq_thought)

    def relationship_line(self, other: "Agent") -> str:
        r = self.profile.rel(other.id)
        rel = REL_PHRASES.get(r.relation_type, r.relation_type)
        return (f"{self.name} and {other.name} are {rel}; they {familiarity_phrase(r.familiarity)} "
                f"and {affinity_phrase(r.affinity)}.")

    def people_line(self, names_to_agents: dict[str, "Agent"]) -> str:
        lines = []
        for other in names_to_agents.values():
            if other.id != self.id:
                lines.append(self.relationship_line(other))
        return "\n".join(lines)

    def snapshot(self) -> dict:
        s = self.state
        return {"id": self.id, "location": s.location, "arena": s.arena, "activity": s.activity,
                "goal": s.current_goal, "path": s.path, "conversation": s.in_conversation,
                "n_memories": len(self.a_mem.id_to_node), "n_reflections": len(self.a_mem.seq_thought)}
