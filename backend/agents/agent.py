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

try:
    from backend.simulation.rngs import seed_rng
except ImportError:  # the world part owns rngs.py; this is the identical scheme (engine._seed_rng)
    def seed_rng(*parts) -> np.random.Generator:
        return np.random.default_rng([zlib.crc32(str(p).encode()) for p in parts])

# named per-agent random substreams (ontology v2 §2.1); "legacy" backs `agent.rng`
STREAMS = ("perceive", "ambient", "encode", "lens", "assoc", "verbatim", "remind", "react", "reflect", "talk",
           "remark", "need", "prime", "legacy")

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


_REFERENCE_CFG: dict | None = None


def set_reference_cfg(cfg: dict | None) -> None:
    """Tell the module-level `CampusMaze()` (conversation.py builds one at import time, with no run in
    scope) which run it is serving, so the upstream chat prompt gets that run's reference mode. The
    Simulation calls this once at construction; tests may call it with None to reset."""
    global _REFERENCE_CFG
    _REFERENCE_CFG = cfg


class CampusMaze:
    """Minimal stand-in for the upstream `Maze`: only access_tile() is used by the
    upstream chat prompt ('<arena> in <sector>'). Our tiles are (location, arena).

    That prompt is the biggest single agent-facing use of a place name in the run, so the tile detail it
    reads is the run's reference rendering: canonical mode returns the keys unchanged, situated mode returns
    descriptions. `viewer` is the agent the prompt is being built for, when the caller knows it (their home
    room then reads "your room"); without one the generic description is used, which is always safe.
    """

    def __init__(self, cfg: dict | None = None, viewer=None):
        self.cfg = cfg
        self.viewer = viewer

    def _cfg(self):
        return self.cfg if self.cfg is not None else _REFERENCE_CFG

    def access_tile(self, tile):
        from backend.simulation import reference
        from backend.simulation.world import WORLD_NAME
        loc, arena = tile
        cfg = self._cfg()
        return {"world": WORLD_NAME, "game_object": "",
                "sector": reference.place(loc, agent=self.viewer, cfg=cfg),
                "arena": reference.room(loc, arena, agent=self.viewer, cfg=cfg)}


class Agent:
    def __init__(self, profile: AgentProfile, cfg: dict, mods, seed: int):
        self.profile = profile
        self.id = profile.id
        self.name = profile.name
        self.cfg = cfg
        self.mods = mods
        self.seed = seed
        self._streams: dict[str, np.random.Generator] = {}
        self.rng = self.stream("legacy")
        self.a_mem = MemoryStream(profile.id)
        sc = ga.Scratch("__memeworld_no_file__")
        sc.name = profile.name
        sc.first_name, sc.last_name = profile.name.split(" ", 1)
        sc.age = profile.demographics.get("age")
        sc.innate = profile.ga_innate()
        sc.learned = profile.ga_learned()
        background = (cfg.get("shared_background") or {}).get("markdown")
        if background:
            sc.learned += "\n\nShared introduction to the world:\n" + background
        sc.currently = ""
        sc.lifestyle = profile.ga_lifestyle()
        sc.living_area = f"{profile.home['location']}:{profile.home['arena']}"   # internal id, never a prompt
        sc.daily_plan_req = profile.ga_daily_plan_req(cfg)
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

    def stream(self, name: str) -> np.random.Generator:
        """This agent's random substream for one mechanism: seed_rng(seed, agent_id, name), created lazily
        and cached. Each mechanism draws only from its own stream, so switching one mechanism on never
        shifts another mechanism's draws (common random numbers across conditions)."""
        g = self._streams.get(name)
        if g is None:
            g = self._streams.setdefault(name, seed_rng(self.seed, self.id, name))
        return g

    # --- Generative-Agents-facing views --------------------------------------
    def set_time(self, t: dt.datetime):
        self.scratch.curr_time = t

    def sync_scratch(self):
        """`curr_tile` stays canonical -- it is the internal id, and CampusMaze.access_tile renders it for
        whatever prompt reads it. `act_description` / `currently` are agent-facing, and the activity they
        quote was already written in the run's reference mode by the engine and the scheduler."""
        s = self.state
        self.scratch.curr_tile = (s.location, s.arena)
        self.scratch.act_description = s.activity
        self.scratch.currently = s.current_goal or f"{self.profile.first_name} is {s.activity}."

    def iss(self) -> str:
        """GA identity stable set: name, age, traits, `currently`, lifestyle and `daily_plan_req`. Its two
        place-bearing parts are `currently` (above) and `daily_plan_req` (profile.ga_daily_plan_req, built
        with this run's cfg in __init__), so the ISS carries no canonical place name in situated mode."""
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

    def snapshot(self, role: str | None = None, cohort: str | None = None) -> dict:
        """Frame view of this agent (UI / observer only; never shown to an agent). `role` and `cohort` are the
        co-op role and founder | newcomer, which the engine knows from the roster (None outside the co-op)."""
        s = self.state
        return {"id": self.id, "location": s.location, "arena": s.arena, "activity": s.activity,
                "goal": s.current_goal, "path": s.path, "conversation": s.in_conversation,
                "n_memories": len(self.a_mem.id_to_node), "n_reflections": len(self.a_mem.seq_thought),
                "active": bool(getattr(s, "active", True)), "role": role, "cohort": cohort,
                "open_matters": self._n_open_matters(), "wordings": self._n_wordings()}

    def _n_open_matters(self) -> int:
        """Open matters (NEED, v2 §2.5) still on the agent's mind: memory alive and strength not faded."""
        om = getattr(self, "open_matters", None)
        if not om:
            return 0
        from backend.memory.need import MIN_STRENGTH, strength
        now = getattr(self.scratch, "curr_time", None)
        hl = float((self.cfg.get("need") or {}).get("half_life_hours", 24))
        return sum(1 for m in om if m["node_id"] in self.a_mem.id_to_node
                   and (now is None or strength(m, now, hl) >= MIN_STRENGTH))

    def _n_wordings(self) -> int:
        """Stuck wordings (WORDING, v2 §2.4) heard from others, held in the agent's live memories."""
        meta = getattr(getattr(self, "ctx", None), "meta", None)
        if meta is None:
            return 0
        n = 0
        for nid in self.a_mem.id_to_node:
            m = meta.get(nid)
            if m is not None and m.wordings:
                n += sum(1 for w in m.wordings if not w.get("self_produced"))
        return n
