"""Prestige bias: some agents carry social status that biases attention,
memory importance and (via a status cue in prompts) imitation.

Scores come from module_params.prestige_bias.scores ({agent_id: 0..1}); if
empty, they default to normalised degree centrality of the social graph.
"""
from __future__ import annotations

from backend.modules.base import AgentModifier


class PrestigeBias(AgentModifier):
    name = "prestige_bias"

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        scores = dict(self.params.get("scores") or {})
        if not scores:
            deg = {a: sum(1 for r in ag.profile.relationships.values() if r.familiarity >= 0.3)
                   for a, ag in ctx.agents.items()}
            mx = max(deg.values()) or 1
            scores = {a: d / mx for a, d in deg.items()}
        self.scores = scores

    def p(self, agent_id) -> float:
        return float(self.scores.get(agent_id, 0.0))

    def modify_attention(self, agent, fact, p):
        boost = max([self.p(a) for a in fact.get("involves", []) if a != agent.id] or [0.0])
        return p + float(self.params.get("attention_gain", 0.3)) * boost

    def modify_memory(self, agent, draft):
        boost = max([self.p(a) for a in draft.get("speakers", []) + draft.get("involves", []) if a != agent.id] or [0.0])
        draft["importance"] = draft["importance"] + round(float(self.params.get("importance_gain", 1.5)) * boost * 2)
        return draft

    def modify_prompt(self, agent, kind, lines, **kw):
        target = kw.get("target")
        if target is not None and self.p(target.id) >= 0.7:
            lines = lines + [f"{target.profile.first_name} is widely admired and well known on campus."]
        return lines

    def frame_state(self, agent_id):
        return {"prestige": round(self.p(agent_id), 3)}
