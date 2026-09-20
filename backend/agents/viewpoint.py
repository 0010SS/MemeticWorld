"""Observer-specific viewpoints on world events (D42).

Perception (perception.py) decides WHICH facts an agent notices and tags each
with the agent's vantage and whom the agent can recognise. This module turns
the world's canonical fact sentences into the agent's own perceived version:
participants experience it first-hand, bystanders across the room only glimpse
it, strangers are described by appearance rather than by name.

The renderer only sees the facts the agent perceived (never the event, its
hidden family or any other agent's view), and is told to omit/blur, not add.
A fact rendered as empty was not perceivable from there and is dropped.
The canonical text stays on the fact as `world_text` (simulator-only).

Names of people the observer does not recognise never get through, whatever the
renderer returns: the same hard guard runs on every rendering and on the
fallback used when the reply is an error, unparseable, or lacks an item (the
fact then keeps its world wording with those names replaced, is flagged
`viewpoint_fallback`, and is listed under `fallback` in the trace).
"""
from __future__ import annotations

import re

from backend import ga_compat
from backend.agents.ga_prompts import as_json
from backend.llm.client import llm_purpose
from backend.simulation import reference

ga = ga_compat.load()
PROMPT = str(ga_compat.REPO_ROOT / "backend" / "prompts" / "viewpoint_v1.txt")

TAG = {"participant": "was the person involved", "near": "same room, watching",
       "distracted": "same room, but busy in a conversation", "far": "different room of the same building"}


def _situation(agent) -> str:
    s = agent.state
    where = (f"{agent.name} is at {reference.phrase(s.location, s.arena, agent=agent, cfg=agent.cfg)}, "
             f"{s.activity}")
    if s.in_conversation:
        where += ", in the middle of a conversation"
    return where + "."


def _recognition(agent, facts) -> str:
    ctx = agent.ctx
    involved = sorted({a for f in facts for a in f.get("involves", []) if a != agent.id})
    known = [ctx.agents[a].name for a in involved if any(a in f.get("recognised", []) for f in facts)]
    unknown = [ctx.agents[a].name for a in involved if ctx.agents[a].name not in known]
    first = agent.profile.first_name
    lines = []
    if known:
        lines.append(f"{first} knows {', '.join(known)} by name.")
    if unknown:
        lines.append(f"{first} does not know {', '.join(unknown)} and would not know their name"
                     f" -- describe them only by appearance or what they are doing.")
    return "\n".join(lines) or f"{first} does not know anyone else involved by name."


def unrecognised_names(agent, facts) -> list[tuple[str, str]]:
    """(full name, first name) of everyone involved in these facts whom the agent does not recognise.
    Recognition is per observer, so the whole observation's list guards every item (a name cannot slip
    into another item's rendering either)."""
    known = {a for f in facts for a in f.get("recognised", ())}
    ids = sorted({a for f in facts for a in f.get("involves", ())} - known - {agent.id})
    return [(agent.ctx.agents[a].name, agent.ctx.agents[a].profile.first_name) for a in ids]


def guard_names(text: str, names) -> str:
    """Hard guard (D42): replace the given people's full and first names with "someone"."""
    for full, first in names:
        for n in (full, first):
            text = re.sub(rf"\b{re.escape(n)}\b", "someone", text, flags=re.I)
    return text


def render(agent, obs) -> None:
    """Replace each fact's text with the agent's perceived version, in place."""
    facts = obs.facts
    if not facts or not agent.cfg["perception"].get("viewpoints", True):
        return
    items = "\n".join(f"f{i} [{TAG[f.get('vantage', 'near')]}] {f['world_text']}" for i, f in enumerate(facts))
    prompt = ga.gs.generate_prompt([agent.name, _situation(agent), items, _recognition(agent, facts),
                                    agent.profile.first_name], PROMPT)
    with llm_purpose("viewpoint", agent.id):
        raw = agent.ctx.llm.complete(prompt, max_tokens=300, temperature=0.9)
    d = None if (raw or "").startswith("LLM_ERROR") else as_json(raw)
    d = d if isinstance(d, dict) else {}
    hidden = unrecognised_names(agent, facts)
    kept, fallback = [], []
    for i, f in enumerate(facts):
        v = d.get(f"f{i}")
        if not isinstance(v, str):
            # error / unparseable / item missing: the world wording, still without unknown people's names
            fallback.append(f["id"])
            kept.append(dict(f, text=" ".join(guard_names(f["text"], hidden).split()), viewpoint_fallback=True))
            continue
        v = " ".join(v.split())
        if not v:
            continue                                          # not perceivable from this vantage
        kept.append(dict(f, text=guard_names(v, hidden)))
    obs.facts[:] = kept
    kept_ids = {k["id"] for k in kept}
    agent.ctx.tracer.log("viewpoint", agent=agent.id, observation_id=obs.id, prompt=prompt, response=raw,
                         facts=[{"id": f["id"], "vantage": f.get("vantage"), "world_text": f["world_text"],
                                 "perceived": f["text"]} for f in kept],
                         dropped=[f["id"] for f in facts if f["id"] not in kept_ids], fallback=fallback,
                         originating_event_ids=list(obs.event_ids))
