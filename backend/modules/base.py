"""Experimental modules = optional selective pressures.

A module never adds fields to the agent profile and never references memes,
phrases or latent events. It may only re-weight ordinary cognition through the
hooks below. The base agent calls `ModuleStack` hooks unconditionally; with no
module enabled every hook is the identity, which is the baseline condition.

Manipulation checks (ontology v2 §3): every module keeps `counters` of the effects it actually
applied. The stack counts hook effects generically (a hook "applied an effect" when its output
differs from its input), so no module can forget to count; modules add their own counters for
notification hooks (appraisals, rewards). `manipulation_check()` feeds the manifest and the observer's
validity check (a module switched on whose effects never fired is inactive).
"""
from __future__ import annotations

import threading


class AgentModifier:
    name = "base"

    def __init__(self, params: dict, ctx):
        self.params = params or {}
        self.ctx = ctx
        self.counters: dict[str, int] = {}
        self._lock = threading.Lock()   # hooks run inside the engine's worker threads

    def count(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.counters[key] = self.counters.get(key, 0) + n

    def manipulation_check(self) -> dict:
        """What this pressure actually did in the run. Subclasses add module-specific summaries."""
        c = dict(sorted(self.counters.items()))
        return {"active": any(v > 0 for v in c.values()), "counters": c}

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

    # extra context lines added to a prompt of `kind` (decide_to_talk / chat / react).
    # kw: target=<Agent> (dyadic partner) and/or others=[Agent] (group conversation)
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
            q = m.modify_attention(agent, fact, p)
            if q != p:
                m.count("attention")
            p = q
        return max(0.0, min(1.0, p))

    def modify_observation(self, agent, facts):
        for m in self.modules:
            ids = [f.get("id") for f in facts]
            facts = m.modify_observation(agent, facts)
            if [f.get("id") for f in facts] != ids:
                m.count("observation")
        return facts

    def modify_memory(self, agent, draft):
        for m in self.modules:
            before = draft["importance"]
            draft = m.modify_memory(agent, draft)
            if draft["importance"] != before:
                m.count(f"memory.{draft.get('source_type')}")
        draft["importance"] = max(1, min(10, draft["importance"]))
        return draft

    def modify_retrieval(self, agent, nodes, scores, focal):
        for m in self.modules:
            new = m.modify_retrieval(agent, nodes, scores, focal)
            if new != scores:
                m.count("retrieval")
            scores = new
        return scores

    def modify_prompt(self, agent, kind, lines, **kw):
        lines = list(lines)
        for m in self.modules:
            new = m.modify_prompt(agent, kind, lines, **kw)
            if new != lines:
                m.count(f"prompt.{kind}")
            lines = new
        return lines

    def modify_utility(self, agent, kind, value, **kw):
        for m in self.modules:
            new = m.modify_utility(agent, kind, value, **kw)
            if new != value:
                m.count(f"utility.{kind}")
            value = new
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

    def manipulation_checks(self) -> dict:
        """{module name: manipulation_check()} for every enabled module (empty in the baseline)."""
        return {m.name: m.manipulation_check() for m in self.modules}


def build_modules(cfg: dict, ctx) -> ModuleStack:
    """Each module is switched on by exactly its own key. Emotion comes first so that its
    conversation appraisal is available to social reward when both are on."""
    from backend.modules.conformity import Conformity
    from backend.modules.emotion import Emotion
    from backend.modules.prestige import PrestigeBias
    from backend.modules.social_reward import SocialReward
    enabled = cfg.get("modules", {})
    params = cfg.get("module_params", {})
    mods: list[AgentModifier] = []
    if enabled.get("emotion"):
        mods.append(Emotion(params.get("emotion"), ctx))
    if enabled.get("social_reward"):
        mods.append(SocialReward(params.get("social_reward"), ctx))
    if enabled.get("prestige_bias"):
        mods.append(PrestigeBias(params.get("prestige_bias"), ctx))
    if enabled.get("conformity"):
        mods.append(Conformity(params.get("conformity"), ctx))
    return ModuleStack(mods)
