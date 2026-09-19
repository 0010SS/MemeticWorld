"""Partial perception: world event beat -> AgentObservation (a subset of facts).

Each fact is noticed independently with probability depending on
participation, location/room, attention (busy with a conversation that ran
into this tick, or not), relationship to the people involved, salience and
experimental modules.
Visibility-restricted facts (e.g. the hidden cause of a mistake) can only be
perceived by the roles that were privy to them.

Every perceived fact also carries the observer's *vantage* (participant, near,
distracted, far) and which of the people involved the observer can recognise.
The viewpoint renderer (backend/agents/viewpoint.py) turns these into the
observer's own version of the fact before it reaches memory (D42).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentObservation:
    id: str
    agent_id: str
    tick: int
    location: str
    arena: str
    source_type: str                      # perception | conversation | overheard
    facts: list = field(default_factory=list)   # [{id,text,salience,involves,...}]
    event_ids: list = field(default_factory=list)   # SIM-ONLY: originating world events
    speakers: list = field(default_factory=list)
    utterance_ids: list = field(default_factory=list)
    partner: str | None = None

    def text(self) -> str:
        return "\n".join(f["text"] for f in self.facts)


RECOGNISE_FAMILIARITY = 0.3   # below this, an observer knows the person by sight at most, not by name


def vantage(agent, fact: dict, beat: dict) -> str:
    if agent.id in fact["involves"]:
        return "participant"
    if beat.get("arena") and agent.state.arena != beat["arena"]:
        return "far"
    if agent.state.in_conversation:
        return "distracted"
    return "near"


def crowd_size(agent) -> int:
    """Other agents in the same building (events are visible building-wide)."""
    return sum(1 for o in agent.ctx.agents.values()
               if o.id != agent.id and o.state.location == agent.state.location)


def attention_prob(agent, fact: dict, beat: dict, cfg: dict) -> float:
    pc = cfg["perception"]
    if agent.id in fact["involves"]:
        return 1.0                                   # participation
    vis = fact["visibility"]
    if vis == "arena":                               # v3: only agents in the beat's arena (participants above)
        if agent.state.location != beat.get("location") or agent.state.arena != beat.get("arena"):
            return 0.0
    elif vis != "all" and agent.id not in vis:
        return 0.0
    p = pc["base_attention"] * (0.5 + 0.5 * fact["salience"])
    if beat.get("arena") and agent.state.arena != beat["arena"]:
        p *= pc["other_arena_factor"]
    if agent.state.in_conversation:
        p *= pc["busy_factor"]
    # busy places: more going on, each thing is less likely to be noticed
    p *= float(pc.get("crowd_factor", 1.0)) ** max(0, crowd_size(agent) - 2)
    fam = max([agent.profile.rel(a).familiarity for a in fact["involves"]] or [0.0])
    p += pc["familiarity_weight"] * fam
    return agent.mods.modify_attention(agent, fact, p)


def observe(agent, beat: dict, event_id: str, cfg: dict, rng, obs_id: str) -> AgentObservation | None:
    """Every fact takes exactly one draw from `rng`, in fact order, even when it cannot be noticed (p = 0):
    a fact's visibility (e.g. link_visibility making a private fact public) then never shifts the draws
    of the other facts. The engine passes one stream per (agent, beat)."""
    kept = []
    for fact in beat["facts"]:
        p = attention_prob(agent, fact, beat, cfg)
        u = rng.random()
        if p > 0 and u < p:
            known = [a for a in fact["involves"] if a != agent.id and
                     agent.profile.rel(a).familiarity >= RECOGNISE_FAMILIARITY]
            kept.append(dict(fact, p_attend=round(p, 3), vantage=vantage(agent, fact, beat),
                             world_text=fact["text"], recognised=known))
    kept = agent.mods.modify_observation(agent, kept)
    if not kept:
        return None
    return AgentObservation(id=obs_id, agent_id=agent.id, tick=beat["tick"], location=agent.state.location,
                            arena=agent.state.arena, source_type="perception", facts=kept,
                            event_ids=[event_id])
