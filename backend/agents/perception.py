"""Partial perception: world event beat -> AgentObservation (a subset of facts).

Each fact is noticed independently with probability depending on
participation, location/room, attention (busy in a conversation or not),
relationship to the people involved, salience and experimental modules.
Visibility-restricted facts (e.g. the hidden cause of a mistake) can only be
perceived by the roles that were privy to them.
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


def attention_prob(agent, fact: dict, beat: dict, cfg: dict) -> float:
    pc = cfg["perception"]
    if agent.id in fact["involves"]:
        return 1.0                                   # participation
    vis = fact["visibility"]
    if vis != "all" and agent.id not in vis:
        return 0.0
    p = pc["base_attention"] * (0.5 + 0.5 * fact["salience"])
    if beat.get("arena") and agent.state.arena != beat["arena"]:
        p *= pc["other_arena_factor"]
    if agent.state.in_conversation:
        p *= pc["busy_factor"]
    fam = max([agent.profile.rel(a).familiarity for a in fact["involves"]] or [0.0])
    p += pc["familiarity_weight"] * fam
    return agent.mods.modify_attention(agent, fact, p)


def observe(agent, beat: dict, event_id: str, cfg: dict, rng, obs_id: str) -> AgentObservation | None:
    kept = []
    for fact in beat["facts"]:
        p = attention_prob(agent, fact, beat, cfg)
        if p > 0 and rng.random() < p:
            kept.append(dict(fact, p_attend=round(p, 3)))
    kept = agent.mods.modify_observation(agent, kept)
    if not kept:
        return None
    return AgentObservation(id=obs_id, agent_id=agent.id, tick=beat["tick"], location=agent.state.location,
                            arena=agent.state.arena, source_type="perception", facts=kept,
                            event_ids=[event_id])
