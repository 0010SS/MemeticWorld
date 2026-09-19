"""Emotion module: per-agent (valence, arousal), lexicon-appraised.

Appraisal uses a small valence lexicon instead of an LLM call (cheap and
deterministic). Effects when enabled:
  * memory importance += salience_gain * arousal * 10 * |valence|-weighted
  * prompts get a mood line ("Maya is feeling upbeat.")
  * conversations: mild emotional contagion toward the partner's valence
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend.modules.base import AgentModifier

NEG = set("""missed late lost wrong spilled failed ruined broke broken locked stuck soaked embarrassed
laughed-at alone grumpy hungry frazzled unusable forgot forgotten mistake missing throw asleep empty
couldn't cut left rushed distracted allergic cancelled tired stress stressed awful terrible annoying
worst ugh sad upset angry""".split())
POS = set("""perfect best won prize praised lucky great awesome happy love loved fun interesting offered
better understood highest free right exactly together enjoyed nice glad thanks thank laugh laughing
amazing cool wonderful sold funny helped help""".split())
_W = re.compile(r"[a-z']+")


def lexicon_valence(text: str) -> float:
    words = _W.findall(text.lower())
    p = sum(w in POS for w in words)
    n = sum(w in NEG for w in words)
    if p + n == 0:
        return 0.0
    return (p - n) / (p + n)


@dataclass
class EmotionState:
    valence: float = 0.1   # -1..1
    arousal: float = 0.2   # 0..1


class Emotion(AgentModifier):
    name = "emotion"

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        self.state: dict[str, EmotionState] = {}

    def st(self, agent_id) -> EmotionState:
        return self.state.setdefault(agent_id, EmotionState())

    def _appraise(self, agent_id, valence, intensity):
        s = self.st(agent_id)
        s.valence = max(-1.0, min(1.0, s.valence + 0.5 * intensity * (valence - s.valence)))
        # move toward the event's intensity (additive updates saturated arousal at 1.0 within hours)
        s.arousal = max(0.0, min(1.0, s.arousal + 0.5 * (intensity - s.arousal)))

    def on_observation(self, agent, facts):
        for f in facts:
            v = lexicon_valence(f["text"])
            if agent.id in f.get("involves", []):
                self._appraise(agent.id, v, 0.8 * f["salience"])
            elif v != 0:
                self._appraise(agent.id, 0.5 * v, 0.4 * f["salience"])

    def on_conversation(self, conv, agents):
        parts = conv["participants"]
        before = {a: self.st(a).valence for a in parts}
        conv.setdefault("module_state", {})["valence_before"] = before
        for u in conv["utterances"]:
            v = lexicon_valence(u["text"])
            for listener in u["listeners"]:
                if listener in parts:
                    self._appraise(listener, v if v else 0.2, 0.3)
        k = float(self.params.get("contagion", 0.2))
        if len(parts) == 2:
            a, b = parts
            va, vb = self.st(a).valence, self.st(b).valence
            self.st(a).valence += k * (vb - va)
            self.st(b).valence += k * (va - vb)
        conv["module_state"]["valence_after"] = {a: self.st(a).valence for a in parts}

    def on_tick(self, tick, agents):
        d = float(self.params.get("decay_per_hour", 0.15)) * self.ctx.clock.tick_minutes / 60.0
        for a in agents:
            s = self.st(a)
            s.valence += d * (0.1 - s.valence)
            s.arousal += d * (0.2 - s.arousal)

    def modify_memory(self, agent, draft):
        s = self.st(agent.id)
        gain = float(self.params.get("salience_gain", 0.5))
        draft["importance"] = draft["importance"] + round(gain * s.arousal * 4)
        return draft

    def mood_word(self, agent_id) -> str | None:
        """A mood label only for clearly non-neutral states. Injecting a label every time made the
        label itself the most repeated word in the social-reward run (see REPORT.md)."""
        s = self.st(agent_id)
        if s.valence > 0.35:
            return "upbeat" if s.arousal < 0.6 else "excited"
        if s.valence < -0.35:
            return "down" if s.arousal < 0.6 else "stressed"
        return None

    def modify_prompt(self, agent, kind, lines, **kw):
        mine = self.mood_word(agent.id)
        if mine:
            lines = lines + [f"{agent.profile.first_name} is feeling {mine}."]
        target = kw.get("target")
        if target is not None:
            theirs = self.mood_word(target.id)
            if theirs:
                lines = lines + [f"{target.profile.first_name} seems {theirs}."]
        return lines

    def modify_utility(self, agent, kind, value, **kw):
        if kind == "talk_prob":
            return value * (1.0 + 0.3 * self.st(agent.id).valence)
        return value

    def frame_state(self, agent_id):
        s = self.st(agent_id)
        return {"valence": round(s.valence, 3), "arousal": round(s.arousal, 3)}
