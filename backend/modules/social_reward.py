"""Social reward module: R_i = λ · Δhappiness_j of the conversation partner j.

Requires the emotion module (it provides happiness = valence).
Rewards only the partner's emotional change -- never phrases, spreading or adoption.
Effects:
  * prompt: the agent values lifting other people's mood (the treatment itself),
  * planner: higher utility for starting conversations with people who seem down,
  * memory: conversations that earned reward are encoded as more important
    (a reinforcement signal routed through ordinary memory importance).
"""
from __future__ import annotations

from backend.modules.base import AgentModifier


class SocialReward(AgentModifier):
    name = "social_reward"

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        self.last_reward: dict[str, float] = {}
        self.total: dict[str, float] = {}

    def _emotion(self):
        return self.ctx.mods.get("emotion")

    def on_conversation(self, conv, agents):
        ms = conv.get("module_state", {})
        before, after = ms.get("valence_before"), ms.get("valence_after")
        if not before or not after:
            return
        lam = float(self.params.get("lam", 1.0))
        rewards = {}
        for i in conv["participants"]:
            others = [j for j in conv["participants"] if j != i]
            r = lam * sum(after[j] - before[j] for j in others)
            rewards[i] = round(r, 4)
            self.last_reward[i] = r
            self.total[i] = self.total.get(i, 0.0) + r
        ms["social_reward"] = rewards

    def modify_prompt(self, agent, kind, lines, **kw):
        if kind in ("chat", "react", "decide_to_talk"):
            lines = lines + [f"{agent.profile.first_name} genuinely values making the people around them feel better; "
                             f"seeing someone cheer up is deeply rewarding to {agent.profile.first_name}."]
        return lines

    def modify_utility(self, agent, kind, value, **kw):
        target = kw.get("target")
        emo = self._emotion()
        if kind == "talk_prob" and target is not None and emo is not None:
            need = max(0.0, 0.3 - emo.st(target.id).valence)
            return value * (1.0 + float(self.params.get("lam", 1.0)) * need)
        return value

    def modify_memory(self, agent, draft):
        if draft.get("source_type") == "conversation":
            r = self.last_reward.get(agent.id, 0.0)
            draft["importance"] = draft["importance"] + round(float(self.params.get("importance_gain", 2.0)) * max(0.0, r) * 5)
        return draft

    def frame_state(self, agent_id):
        return {"cumulative_reward": round(self.total.get(agent_id, 0.0), 3)}
