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


def crowd_size(agent, scope: str = "arena") -> int:
    """Other agents competing for this observer's attention. `scope` "arena": people in the observer's own
    room, which is what perceptual load is about; "location": everyone in the building (the v2 scope, kept
    for the optional building term)."""
    here = agent.state.location
    return sum(1 for o in agent.ctx.agents.values()
               if o.id != agent.id and o.state.location == here
               and (scope == "location" or o.state.arena == agent.state.arena))


def attention_prob(agent, fact: dict, beat: dict, cfg: dict) -> float:
    """p(this observer notices this fact): ONE priority, every term multiplying the same quantity.

        p = base_attention * (0.5 + 0.5 * salience)             stimulus strength
            * other_arena_factor          if the beat is in another room of this building
            * busy_factor                 if the observer is still in a conversation
            * crowd_factor ** (others in the ROOM - 2)                 perceptual load
            * building_crowd_factor ** (others elsewhere in the building - 2)   optional, off by default
            * (1 + familiarity_weight * familiarity)            social gain on the priority
        clamped to [0, 1]

    Participants always notice (p = 1); visibility gates to 0 before any of this.

    Two changes from v2 (D74). Load is counted per ROOM, not per building: what competes for attention is
    what is in the observer's own field, and counting the whole building made a quiet dorm room as loaded
    as the dining hall two floors down. And `familiarity_weight` is now a MULTIPLICATIVE gain on the
    priority, x(1 + w * fam), where it used to be an additive bonus, + w * fam. Additive, the social term
    escaped every multiplier, so under load it was the only channel left standing and crowding *raised*
    the share of attention going to familiar faces. Load theory (Lavie 1995) has high load cut processing
    of everything, social cues included; a gain on the priority does that while keeping the
    familiar/stranger ratio intact. The default weight moves 0.25 -> 0.8 to hold the median p_attend of
    familiar observers where it was (measured on 623 reconstructed draws from 15 archived runs: 0.417 vs
    0.418; the old additive bonus was worth a median 0.67 of the stimulus term it was added to).
    """
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
    # busy rooms: more going on in the observer's own field, each thing is less likely to be noticed
    in_room = crowd_size(agent)
    p *= float(pc.get("crowd_factor", 1.0)) ** max(0, in_room - 2)
    bcf = pc.get("building_crowd_factor")
    if bcf:                                          # weaker second-order load from the rest of the building
        p *= float(bcf) ** max(0, crowd_size(agent, "location") - in_room - 2)
    fam = max([agent.profile.rel(a).familiarity for a in fact["involves"]] or [0.0])
    p *= 1.0 + float(pc["familiarity_weight"]) * fam
    return min(1.0, max(0.0, agent.mods.modify_attention(agent, fact, p)))


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
