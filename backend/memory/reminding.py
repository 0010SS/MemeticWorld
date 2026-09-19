"""Spontaneous reminding (D48, extended in ontology v2 §2.3): a new experience may bring an earlier one
to mind.

This is the step where two separate incidents can get connected in an agent's head. The agent is shown
its own earlier memories (a stochastic mix of the most related ones and a few salient older ones, so
that surface-different but structurally similar incidents have a chance to come up despite lexical
embeddings) and asked whether the new experience reminds it of one. A positive answer is stored as a
thought linking both memories; it can later be retrieved into conversations and reflections like any
other thought.

v2:
  * fires after the sources in `reminding.on_sources` (perception, conversation, overheard);
  * earlier reminding thoughts are candidates too, so links compound into chains;
  * every link is a Link record + `memory_link` trace (store.record_link);
  * prompt v2 makes "ordinary, nothing comes to mind" the easy default and asks for a likeness in what
    happened, not in place, people or topic (v1 said yes 56 of 68 times in a live run);
  * a reminding thought draws down the reflection trigger only by `reminding.reflection_weight` x its
    importance (default 0: these thoughts made reflections explode, 198 in one simulated day).

Nothing here refers to event families, names or labels: candidates are chosen from the agent's own
memory stream only, by memory kind (did the agent see or hear about it, which the agent knows) and GA
retrieval scores. Randomness comes only from agent.stream("remind").
"""
from __future__ import annotations

import datetime as dt

from backend import ga_compat
from backend.agents.ga_prompts import as_json
from backend.llm.client import llm_purpose
from backend.memory.retrieval import retrieve
from backend.memory.store import MemoryMeta, record_link

ga = ga_compat.load()
PROMPT = str(ga_compat.REPO_ROOT / "backend" / "prompts" / "reminding_v2.txt")
EXPERIENCE = {"perception", "conversation", "overheard"}
CANDIDATE_SOURCES = EXPERIENCE | {"reminding"}      # earlier links can be extended into chains


def _candidates(agent, new_node, focal: str, rng, n_related: int, n_salient: int) -> list:
    ctx = agent.ctx
    min_age = dt.timedelta(minutes=int(agent.cfg["reminding"].get("min_age_minutes", 90)))
    now = agent.scratch.curr_time
    ok = {n.node_id for n in agent.a_mem.all_nodes()
          if n.node_id != new_node.node_id and now - n.created >= min_age
          and (ctx.meta.get(n.node_id) or MemoryMeta(agent.id, "?")).source_type in CANDIDATE_SOURCES
          and n.poignancy >= 3}
    if not ok:
        return []
    excl = {n.node_id for n in agent.a_mem.all_nodes()} - ok
    rel = retrieve(agent, [focal], k=n_related, rng=rng, exclude_ids=excl, touch=False)[focal].nodes
    rest = sorted((agent.a_mem.id_to_node[i] for i in ok - {n.node_id for n in rel}), key=lambda n: n.node_id)
    extra = []
    if rest and n_salient:
        w = [float(n.poignancy) ** 2 for n in rest]
        tot = sum(w)
        idx = rng.choice(len(rest), size=min(n_salient, len(rest)), replace=False, p=[x / tot for x in w])
        extra = [rest[i] for i in sorted(idx)]
    cands = rel + extra
    order = rng.permutation(len(cands))
    return [cands[i] for i in order]


def _how(agent, obs) -> str:
    """How the new experience reached the agent, in the prompt's words."""
    if obs.source_type == "conversation":
        return f"just talked with {obs.partner}" if obs.partner else "just had a conversation"
    if obs.source_type == "overheard":
        return "just overheard"
    if any(agent.id in f.get("involves", []) for f in obs.facts):
        return "just experienced"
    return "just noticed"


def _flag(v) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() == "true")


def maybe_remind(agent, obs, new_node, rng=None):
    """Ask whether `new_node` (just encoded from `obs`) brings an earlier memory to mind; on a yes, store a
    reminding thought and a Link. `rng` is ignored: draws come from agent.stream("remind") (v2 §2.1)."""
    rc = agent.cfg.get("reminding") or {}
    if not rc.get("enabled") or new_node is None or not obs.facts:
        return None
    if obs.source_type not in rc.get("on_sources", ["perception"]):
        return None
    # world salience gates perceptions; conversation lines carry a fixed salience, so they are not gated
    if obs.source_type == "perception" and max(f["salience"] for f in obs.facts) < float(rc.get("min_salience", 0.45)):
        return None
    rng = agent.stream("remind")
    focal = obs.text()
    cands = _candidates(agent, new_node, focal, rng, int(rc.get("n_related", 4)), int(rc.get("n_salient", 3)))
    if not cands:
        return None
    listing = "\n".join(f"{i + 1}. {n.description}" for i, n in enumerate(cands))
    prompt = ga.gs.generate_prompt([agent.iss(), agent.name,
                                    f"{agent.scratch.curr_time.strftime('%A %H:%M')} at the {agent.state.location}",
                                    focal, listing, agent.profile.first_name, _how(agent, obs)], PROMPT)
    with llm_purpose("reminding", agent.id):
        raw = agent.ctx.llm.complete(prompt, max_tokens=120, temperature=1.0)
    d = as_json(raw) or {}
    ordinary = _flag(d.get("ordinary"))
    try:
        pick = None if ordinary else int(d.get("reminded_of"))
    except (TypeError, ValueError):
        pick = None
    why = d.get("what_felt_alike")
    why = " ".join(str(why).split()).strip().rstrip(".") if isinstance(why, str) and why.strip().lower() not in ("", "null", "none") else None
    ctx = agent.ctx
    node = old = None
    if pick is not None and 1 <= pick <= len(cands):
        old = cands[pick - 1]
        text = (f"{new_node.description} It reminded {agent.name} of an earlier time: {old.description}"
                + (f" What felt alike: {why}." if why else ""))
        emb = ctx.embed(text)
        imp = max(new_node.poignancy, old.poignancy)
        node = agent.a_mem.add("thought", agent.scratch.curr_time, agent.name, "was reminded of", old.subject, text,
                               set(new_node.keywords) | set(old.keywords), imp, emb, [new_node.node_id, old.node_id])
        ctx.meta.set(node.node_id, MemoryMeta(agent_id=agent.id, source_type="reminding", salience=0.6,
                                              originating_event_ids=ctx.meta.events_of([new_node.node_id, old.node_id]),
                                              source_ids=[new_node.node_id, old.node_id]))
        # a thought, not a new experience: it only counts toward the reflection trigger by reflection_weight
        agent.scratch.importance_trigger_curr -= imp * float(rc.get("reflection_weight", 0.0))
        agent.scratch.importance_ele_n += 1
        record_link(agent, new_node.node_id, old.node_id, "reminding", why or "", via=node.node_id,
                    source_type=obs.source_type)
    ctx.tracer.log("reminding", agent=agent.id, observation_id=obs.id, source_type=obs.source_type,
                   new_node=new_node.node_id, candidates=[n.node_id for n in cands], prompt=prompt, response=raw,
                   ordinary=ordinary, reminded_of=old.node_id if node else None, what_felt_alike=why,
                   node_id=node.node_id if node else None, text=node.description if node else None,
                   originating_event_ids=list(obs.event_ids),
                   reminded_of_event_ids=ctx.meta.events_of([old.node_id]) if node else [],
                   chained=bool(node) and (ctx.meta.get(old.node_id) or MemoryMeta(agent.id, "?")).source_type == "reminding")
    return node
