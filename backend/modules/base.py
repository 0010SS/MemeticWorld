"""Experimental modules = optional selective pressures.

A module never adds fields to the agent profile and never references memes,
phrases or latent events. It may only re-weight ordinary cognition through the
hooks below. The base agent calls `ModuleStack` hooks unconditionally; with no
module enabled every hook is the identity, which is the baseline condition.
"""
from __future__ import annotations


class AgentModifier:
    name = "base"

    def __init__(self, params: dict, ctx):
        self.params = params or {}
        self.ctx = ctx

    # perception: probability that `agent` notices `fact`
    def modify_attention(self, agent, fact: dict, p: float) -> float:
        return p

    # alias required by the spec: modify the whole observation (list of facts) after sampling
    def modify_observation(self, agent, facts: list[dict]) -> list[dict]:
        return facts

    # memory draft before storage: {"text","importance","salience","source_type","speakers","involves"}
    def modify_memory(self, agent, draft: dict) -> dict:
        return draft

    # retrieval scores node_id -> score
    def modify_retrieval(self, agent, nodes: list, scores: dict, focal: str) -> dict:
        return scores

    # extra context lines added to a prompt of `kind` (decide_to_talk / chat / react)
    def modify_prompt(self, agent, kind: str, lines: list[str], **kw) -> list[str]:
        return lines

    # utilities / probabilities used by the planner (kind: "talk_prob", ...)
    def modify_utility(self, agent, kind: str, value: float, **kw) -> float:
        return value

    # notifications
    def on_observation(self, agent, facts: list[dict]):
        pass

    def on_conversation(self, conv: dict, agents: dict):
        pass

    def on_tick(self, tick: int, agents: dict):
        pass

    def frame_state(self, agent_id: str) -> dict:
        return {}


class ModuleStack:
    def __init__(self, modules: list[AgentModifier]):
        self.modules = modules

    def get(self, name):
        for m in self.modules:
            if m.name == name:
                return m
        return None

    def modify_attention(self, agent, fact, p):
        for m in self.modules:
            p = m.modify_attention(agent, fact, p)
        return max(0.0, min(1.0, p))

    def modify_observation(self, agent, facts):
        for m in self.modules:
            facts = m.modify_observation(agent, facts)
        return facts

    def modify_memory(self, agent, draft):
        for m in self.modules:
            draft = m.modify_memory(agent, draft)
        draft["importance"] = max(1, min(10, draft["importance"]))
        return draft

    def modify_retrieval(self, agent, nodes, scores, focal):
        for m in self.modules:
            scores = m.modify_retrieval(agent, nodes, scores, focal)
        return scores

    def modify_prompt(self, agent, kind, lines, **kw):
        lines = list(lines)
        for m in self.modules:
            lines = m.modify_prompt(agent, kind, lines, **kw)
        return lines

    def modify_utility(self, agent, kind, value, **kw):
        for m in self.modules:
            value = m.modify_utility(agent, kind, value, **kw)
        return value

    def on_observation(self, agent, facts):
        for m in self.modules:
            m.on_observation(agent, facts)

    def on_conversation(self, conv, agents):
        for m in self.modules:
            m.on_conversation(conv, agents)

    def on_tick(self, tick, agents):
        for m in self.modules:
            m.on_tick(tick, agents)

    def frame_state(self, agent_id):
        out = {}
        for m in self.modules:
            s = m.frame_state(agent_id)
            if s:
                out[m.name] = s
        return out


def build_modules(cfg: dict, ctx) -> ModuleStack:
    from backend.modules.conformity import Conformity
    from backend.modules.emotion import Emotion
    from backend.modules.prestige import PrestigeBias
    from backend.modules.social_reward import SocialReward
    enabled = cfg.get("modules", {})
    params = cfg.get("module_params", {})
    mods: list[AgentModifier] = []
    if enabled.get("emotion") or enabled.get("social_reward"):
        mods.append(Emotion(params.get("emotion"), ctx))
    if enabled.get("social_reward"):
        mods.append(SocialReward(params.get("social_reward"), ctx))
    if enabled.get("prestige_bias"):
        mods.append(PrestigeBias(params.get("prestige_bias"), ctx))
    if enabled.get("conformity"):
        mods.append(Conformity(params.get("conformity"), ctx))
    return ModuleStack(mods)
