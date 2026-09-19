"""Social reward module: R_i = λ · Δhappiness_j of i's conversation partners j.

Independent of the emotion module (ontology v2 §3). Without emotion, a partner's happiness is how
they come across in what they say: each utterance is appraised with the emotion module's valence
lexicon, and the result is kept inside this module only. Nothing about anyone's mood is ever written
into a prompt (mood lines were themselves repeated as fake conventions, D36). When emotion is on too,
its valence is used instead, so the two pressures compose instead of keeping two moods.
Rewards only partners' emotional change -- never phrases, spreading or adoption.
Effects:
  * prompt: the agent values lifting other people's mood (the treatment itself),
  * planner: higher utility for starting conversations with people who seem down,
  * memory: conversations that earned reward are encoded as more important
    (a reinforcement signal routed through ordinary memory importance).
"""
from __future__ import annotations

from backend.modules.base import AgentModifier
from backend.modules.emotion import EmotionState, lexicon_valence

NEUTRAL = EmotionState().valence   # resting valence, so both valence sources give the same default boost
FADE_PER_HOUR = 0.15               # apparent mood relaxes toward neutral (emotion's default decay rate)


class SocialReward(AgentModifier):
    name = "social_reward"

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        self.apparent: dict[str, float] = {}   # lexicon-appraised valence of what each person last said
        self.last_reward: dict[str, float] = {}
        self.total: dict[str, float] = {}

    def _emotion(self):
        mods = getattr(self.ctx, "mods", None)
        return mods.get("emotion") if mods is not None else None

    def valence(self, agent_id: str) -> float:
        emo = self._emotion()
        if emo is not None:
            return emo.st(agent_id).valence
        return self.apparent.get(agent_id, NEUTRAL)

    def on_conversation(self, conv, agents):
        parts = conv["participants"]
        ms = conv.setdefault("module_state", {})
        if self._emotion() is not None and ms.get("valence_before") and ms.get("valence_after"):
            before, after = ms["valence_before"], ms["valence_after"]
        else:
            before = {j: self.apparent.get(j, NEUTRAL) for j in parts}
            for u in conv["utterances"]:
                v = lexicon_valence(u["text"])
                if v:
                    cur = self.apparent.get(u["speaker"], NEUTRAL)
                    self.apparent[u["speaker"]] = cur + 0.5 * (v - cur)
            after = {j: self.apparent.get(j, NEUTRAL) for j in parts}
        lam = float(self.params.get("lam", 1.0))
        rewards = {}
        for i in parts:
            others = [j for j in parts if j != i]
            # mean over partners: a group's reward is on the same scale as a dyad's
            r = lam * sum(after[j] - before[j] for j in others) / max(1, len(others))
            rewards[i] = round(r, 4)
            self.last_reward[i] = r
            self.total[i] = self.total.get(i, 0.0) + r
        ms["social_reward"] = rewards
        if any(rewards.values()):
            self.count("rewarded_conversations")

    def on_tick(self, tick, agents):
        if not self.apparent:
            return
        d = FADE_PER_HOUR * self.ctx.clock.tick_minutes / 60.0
        for a, v in self.apparent.items():
            self.apparent[a] = v + d * (NEUTRAL - v)

    def modify_prompt(self, agent, kind, lines, **kw):
        if kind in ("chat", "react", "decide_to_talk"):
            lines = lines + [f"{agent.profile.first_name} genuinely values making the people around them feel better; "
                             f"seeing someone cheer up is deeply rewarding to {agent.profile.first_name}."]
        return lines

    def modify_utility(self, agent, kind, value, **kw):
        target = kw.get("target")
        if kind == "talk_prob" and target is not None:
            need = max(0.0, 0.3 - self.valence(target.id))
            return value * (1.0 + float(self.params.get("lam", 1.0)) * need)
        return value

    def modify_memory(self, agent, draft):
        if draft.get("source_type") == "conversation":
            r = self.last_reward.get(agent.id, 0.0)
            draft["importance"] = draft["importance"] + round(float(self.params.get("importance_gain", 2.0)) * max(0.0, r) * 5)
        return draft

    def frame_state(self, agent_id):
        return {"cumulative_reward": round(self.total.get(agent_id, 0.0), 3)}

    def manipulation_check(self):
        out = super().manipulation_check()
        out["valence_source"] = "emotion" if self._emotion() is not None else "lexicon"
        out["total_reward"] = {a: round(r, 4) for a, r in sorted(self.total.items())}
        return out
