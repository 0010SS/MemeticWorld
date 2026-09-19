"""Prestige bias: some agents carry social status that biases attention,
memory importance and (via a status cue in prompts) imitation.

Where the scores come from is `module_params.prestige_bias.scores_source`:
  * degree: normalised degree centrality of the social graph (ties with familiarity >= 0.3);
  * random: the same degree scores randomly re-assigned to agents (seeded by `seed`), which keeps the
    distribution of status but breaks its link to network position (a control for "status" vs "hub");
  * file: `module_params.prestige_bias.scores` ({agent_id: 0..1}; unlisted agents get 0).
"""
from __future__ import annotations

import zlib

import numpy as np

from backend.modules.base import AgentModifier

SOURCES = ("degree", "random", "file")


def degree_scores(agents: dict) -> dict[str, float]:
    deg = {a: sum(1 for r in ag.profile.relationships.values() if r.familiarity >= 0.3)
           for a, ag in agents.items()}
    mx = max(deg.values() or [0]) or 1
    return {a: d / mx for a, d in deg.items()}


class PrestigeBias(AgentModifier):
    name = "prestige_bias"

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        scores = dict(self.params.get("scores") or {})
        # configs written before scores_source existed: explicit scores meant "use these"
        source = self.params.get("scores_source") or ("file" if scores else "degree")
        if source not in SOURCES:
            raise ValueError(f"prestige_bias.scores_source must be one of {SOURCES}, got {source!r}")
        if source == "file" and not scores:
            raise ValueError("prestige_bias.scores_source is 'file' but prestige_bias.scores is empty")
        if source != "file" and scores:
            raise ValueError(f"prestige_bias.scores is set but scores_source is {source!r}; "
                             "set scores_source: file to use them")
        if source == "file":
            scores = {a: float(scores.get(a, 0.0)) for a in ctx.agents}
        else:
            scores = degree_scores(ctx.agents)
            if source == "random":
                ids = sorted(scores)
                seed = (getattr(ctx, "cfg", None) or {}).get("seed", 0)
                rng = np.random.default_rng([zlib.crc32(str(p).encode()) for p in (seed, "prestige_scores")])
                vals = [scores[a] for a in ids]
                scores = {a: vals[int(i)] for a, i in zip(ids, rng.permutation(len(ids)))}
        self.source = source
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
        for other in [kw.get("target"), *(kw.get("others") or [])]:
            if other is not None and self.p(other.id) >= 0.7:
                lines = lines + [f"{other.profile.first_name} is widely admired and well known on campus."]
        return lines

    def frame_state(self, agent_id):
        return {"prestige": round(self.p(agent_id), 3)}

    def manipulation_check(self):
        out = super().manipulation_check()
        out["scores_source"] = self.source
        out["scores"] = {a: round(s, 3) for a, s in sorted(self.scores.items())}
        return out
