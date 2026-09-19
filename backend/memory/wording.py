"""WORDING (ontology v2 §2.4): exact wordings that stuck in an agent's memory, as first-class records.

Verbatim stickiness (encoder.sticky_phrases) decides which heard phrases survive word for word. They
are stored on the memory node's simulator-side sidecar (`MemoryMeta.wordings`), whether or not they are
also quoted in the memory text (`memory.verbatim.render_in_text`).

Production priming (`priming.enabled`) puts a few recently heard wordings back in mind when the agent
speaks. The phrases come only from the agent's OWN memories' wordings (never from the analyzer), and
disappear with the memory when it is forgotten. Nothing here tells the agent to reuse them.
"""
from __future__ import annotations

import datetime as dt
import math


def record(agent, node_id: str, phrases: list, obs) -> list[dict]:
    """Store stuck phrases on `node_id`'s sidecar meta and trace each as a `wording` record.
    `phrases`: dicts from sticky_phrases ({"phrase", "heard_from", "utterance_id"}) or plain strings."""
    ctx = agent.ctx
    m = ctx.meta.get(node_id)
    out = []
    for ph in phrases:
        d = ph if isinstance(ph, dict) else {"phrase": ph}
        heard_from = d.get("heard_from") or (obs.speakers[0] if len(obs.speakers) == 1 else None)
        w = {"phrase": d["phrase"], "heard_from": heard_from, "utterance_id": d.get("utterance_id"),
             "tick": obs.tick, "time": agent.scratch.curr_time.isoformat(timespec="minutes"),
             "self_produced": heard_from == agent.id}
        if m is not None:
            m.wordings.append(w)
        ctx.tracer.log("wording", agent=agent.id, node_id=node_id, observation_id=obs.id,
                       source_type=obs.source_type, **w)
        out.append(w)
    return out


def recent_wordings(agent, now: dt.datetime, window_hours: float, k: int, rng) -> list[str]:
    """Up to k phrases other people said that stuck in the agent's memory within the last `window_hours`,
    sampled without replacement with weight = sum over hearings of exp(-decay_rate * hours ago), i.e.
    the retrieval recency function summed over repetitions."""
    meta = agent.ctx.meta
    dr = float(agent.cfg["memory"]["decay_rate"])
    weight: dict[str, float] = {}
    shown: dict[str, str] = {}
    for n in sorted(agent.a_mem.id_to_node.values(), key=lambda n: n.node_id):
        m = meta.get(n.node_id)
        for w in (m.wordings if m is not None else []):
            if w.get("self_produced"):
                continue
            hours = (now - dt.datetime.fromisoformat(w["time"])).total_seconds() / 3600.0
            if hours < 0 or hours > window_hours:
                continue
            key = w["phrase"].lower()
            shown.setdefault(key, w["phrase"])
            weight[key] = weight.get(key, 0.0) + math.exp(-dr * hours)
    if not weight or k <= 0:
        return []
    keys = sorted(weight)
    p = [weight[x] for x in keys]
    tot = sum(p)
    idx = rng.choice(len(keys), size=min(k, len(keys)), replace=False, p=[x / tot for x in p])
    return [shown[keys[i]] for i in idx]


def priming_line(agent, rng=None, conversation_id: str | None = None) -> str | None:
    """Context line for the conversation prompt, or None (priming off, or nothing heard lately).
    Draws from the agent's "prime" stream unless an rng is given; traces itself as `priming`."""
    pc = agent.cfg.get("priming") or {}
    if not pc.get("enabled"):
        return None
    rng = rng if rng is not None else agent.stream("prime")
    phrases = recent_wordings(agent, agent.scratch.curr_time, float(pc.get("window_hours", 24)),
                              int(pc.get("max_phrases", 3)), rng)
    if not phrases:
        return None
    agent.ctx.tracer.log("priming", agent=agent.id, conversation_id=conversation_id, phrases=phrases)
    return (f"Things {agent.profile.first_name} has heard people say lately: "
            + ", ".join(f'"{p}"' for p in phrases) + ".")
