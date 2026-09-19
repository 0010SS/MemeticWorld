"""NEED (ontology v2 §2.5): unresolved matters stay on an agent's mind and cue conversation.

After encoding, something the agent took part in (a perceived incident involving the agent itself), or
any memory with importance >= `need.min_importance`, becomes an open matter. Its strength decays with a
half-life, and each time the matter's memory is retrieved into something the agent says, the strength is
multiplied by `discuss_decay`. The strongest few are extra retrieval focal points and one context line
in conversations, so the same things keep coming up until they have been talked through.

This is ordinary agent state (like a to-do list in one's head): texts are the agent's own memories;
there is no event id, family or label, and nothing about which expressions are used.
"""
from __future__ import annotations

import datetime as dt

MIN_STRENGTH = 0.1       # below this a matter has faded from mind (~3.3 half-lives)


def _cfg(agent) -> dict:
    return agent.cfg.get("need") or {}


def open_matters(agent) -> list[dict]:
    """agent.open_matters, created lazily: [{"node_id", "text", "created" (ISO), "strength"}]."""
    om = getattr(agent, "open_matters", None)
    if om is None:
        om = agent.open_matters = []
    return om


def strength(matter: dict, now: dt.datetime, half_life_hours: float) -> float:
    hours = max(0.0, (now - dt.datetime.fromisoformat(matter["created"])).total_seconds() / 3600.0)
    return matter["strength"] * 0.5 ** (hours / max(1e-6, half_life_hours))


def note(agent, obs, node) -> dict | None:
    """Open (or refresh) a matter for a newly encoded memory. Returns the matter or None."""
    nc = _cfg(agent)
    if not nc.get("enabled") or node is None:
        return None
    took_part = obs.source_type == "perception" and any(agent.id in f.get("involves", []) for f in obs.facts)
    if not took_part and node.poignancy < float(nc.get("min_importance", 6)):
        return None
    now = agent.scratch.curr_time
    hl = float(nc.get("half_life_hours", 24))
    matters = open_matters(agent)
    m = next((x for x in matters if x["node_id"] == node.node_id), None)   # merged into an open matter
    action = "refresh" if m else "open"
    if m is None:
        m = {"node_id": node.node_id, "text": node.description}
        matters.append(m)
    m.update(created=now.isoformat(timespec="minutes"), strength=1.0)
    matters[:] = [x for x in matters                            # forgotten or faded matters are gone
                  if x["node_id"] in agent.a_mem.id_to_node and strength(x, now, hl) >= MIN_STRENGTH]
    agent.ctx.tracer.log("open_matter", agent=agent.id, action=action, node_id=node.node_id, text=m["text"],
                         strength=1.0, reason="took_part" if took_part else "importance",
                         importance=node.poignancy, observation_id=obs.id, n_open=len(matters))
    return m


def focal(agent, now: dt.datetime, conversation_id: str | None = None) -> list[str]:
    """Texts of the top `max_open` open matters by decayed strength (still in memory, not faded).
    Traced as `open_matter_focal` when non-empty."""
    nc = _cfg(agent)
    if not nc.get("enabled"):
        return []
    hl = float(nc.get("half_life_hours", 24))
    live = [(strength(m, now, hl), m) for m in open_matters(agent) if m["node_id"] in agent.a_mem.id_to_node]
    top = sorted((x for x in live if x[0] >= MIN_STRENGTH), key=lambda x: (-x[0], x[1]["node_id"]))
    top = top[: int(nc.get("max_open", 3))]
    if top:
        agent.ctx.tracer.log("open_matter_focal", agent=agent.id, conversation_id=conversation_id,
                             node_ids=[m["node_id"] for _, m in top], strengths=[round(s, 3) for s, _ in top],
                             texts=[m["text"] for _, m in top])
    return [m["text"] for _, m in top]


def discussed(agent, retrieved_node_ids) -> None:
    """The agent just said something drawing on these memories: matters among them weaken."""
    nc = _cfg(agent)
    if not nc.get("enabled"):
        return
    ids = set(retrieved_node_ids)
    for m in open_matters(agent):
        if m["node_id"] in ids:
            m["strength"] *= float(nc.get("discuss_decay", 0.5))
            agent.ctx.tracer.log("open_matter", agent=agent.id, action="discussed", node_id=m["node_id"],
                                 text=m["text"], strength=round(m["strength"], 4))
