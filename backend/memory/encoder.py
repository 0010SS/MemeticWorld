"""Lossy, reconstructive memory encoding: observation -> symbolic memory node.

Pipeline (world event never touches this code; only an AgentObservation does):
  1. semantic pre-filter (deterministic given the seed):
       - drop low-salience facts with p = noise * (1 - salience)
       - generalise unfamiliar people ("Leo" -> "a student") with
         p = entity_generalization * noise
  2. reconstruction by the LLM in the agent's own framing, with a fidelity
     instruction set by `memory.encoding_noise` (0 = verbatim, no LLM rewrite)
  3. importance = upstream GA poignancy prompt (same for every condition)
  4. experimental modules may adjust the draft (emotion, prestige, social reward)
  5. merge into a near-duplicate recent memory (if noise > 0), else store
  6. forget the weakest memory while over capacity
"""
from __future__ import annotations

import re

from backend import ga_compat
from backend.agents import ga_prompts
from backend.llm.client import llm_purpose
from backend.llm.embeddings import cos
from backend.memory.store import MemoryMeta, recency_score

ga = ga_compat.load()
PROMPT_DIR = ga_compat.REPO_ROOT / "backend" / "prompts"

VERB = {"perception": "saw", "conversation": "heard in a conversation",
        "overheard": "overheard", "self": "did", "record": "read in an attributed record"}


def fidelity_instruction(noise: float, first: str) -> str:
    if noise < 0.15:
        return "Keep essentially every detail, including the exact wording people used."
    if noise < 0.45:
        return (f"Keep the main points and anything that stood out to {first}; small details fade. "
                f"Paraphrase, except for any wording that particularly stuck with {first}.")
    if noise < 0.75:
        return (f"Keep only the gist and what mattered to {first}, in one or two sentences in {first}'s own "
                f"words; minor details and exact wording fade.")
    return ("Only a vague gist survives: one short sentence, no quotes; details, times and secondary "
            "people blur (\"someone\", \"something went wrong\").")


def _prefilter(agent, obs, noise: float, gen: float, rng):
    facts = list(obs.facts)
    ops = []
    if obs.source_type == "perception" and len(facts) > 1 and noise > 0:
        best = max(range(len(facts)), key=lambda i: facts[i]["salience"])
        keep = []
        for i, f in enumerate(facts):
            if i != best and rng.random() < noise * (1 - f["salience"]):
                ops.append({"op": "drop_fact", "fact": f["id"]})
            else:
                keep.append(f)
        facts = keep
    if noise > 0 and gen > 0:
        out = []
        for f in facts:
            text = f["text"]
            for other_id in f.get("involves", []):
                if other_id == agent.id:
                    continue
                if agent.profile.rel(other_id).familiarity < 0.3 and rng.random() < gen * noise:
                    first = agent.ctx.agents[other_id].profile.first_name
                    text = re.sub(rf"\b{re.escape(first)}\b", "a student", text, count=1)
                    text = re.sub(rf"\b{re.escape(first)}\b", "they", text)
                    text = text[0].upper() + text[1:]
                    ops.append({"op": "generalize_entity", "who": other_id})
            out.append(dict(f, text=text))
        facts = out
    return facts, ops


def _keywords(agent, text: str, involves: list[str]) -> set:
    kws = {agent.name.lower()}
    for a in involves:
        kws.add(agent.ctx.agents[a].name.lower())
    for w in re.findall(r"[A-Za-z][a-z]{4,}", text):
        if len(kws) >= 8:
            break
        kws.add(w.lower())
    return kws


def encode(agent, obs, rng) -> object | None:
    """Encode an AgentObservation into agent memory. Returns the node (new or merged)."""
    ctx = agent.ctx
    mc = agent.cfg["memory"]
    noise = float(mc["encoding_noise"])
    facts, ops = _prefilter(agent, obs, noise, float(mc.get("entity_generalization", 0.5)), rng)
    if not facts:
        return None
    raw = "\n".join(f["text"] for f in facts)
    involves = sorted({a for f in facts for a in f.get("involves", [])} | set(obs.speakers))
    first = agent.profile.first_name
    prompt = None
    if noise <= 0:
        if obs.source_type == "perception":
            text = f"{agent.name} saw: " + " ".join(f["text"] for f in facts)
        else:
            text = f"{agent.name} {VERB[obs.source_type]}: " + " ".join(f["text"] for f in facts)
    else:
        people = "\n".join(agent.relationship_line(ctx.agents[a]) for a in involves if a != agent.id)
        prompt = ga.gs.generate_prompt(
            [agent.name, agent.iss(), agent.scratch.curr_time.strftime("%A %H:%M"),
             f"{obs.location} ({obs.arena})", VERB[obs.source_type], raw,
             fidelity_instruction(noise, first), people],
            str(PROMPT_DIR / "encode_memory_v1.txt"))
        with llm_purpose("encode_memory", agent.id):
            text = ctx.llm.complete(prompt, max_tokens=160, temperature=0.7)
        text = text.strip().strip('"').strip()
        if not text or text.startswith("LLM_ERROR"):
            text = f"{agent.name} {VERB[obs.source_type]}: " + " ".join(f["text"] for f in facts)
        text = " ".join(text.split())[:600]
    kind = "event" if obs.source_type in ("perception", "record", "self") else "chat"
    importance = ga_prompts.poignancy(agent, text, "chat" if kind == "chat" else "event")
    salience = max(f["salience"] for f in facts)
    draft = {"text": text, "importance": importance, "salience": salience, "source_type": obs.source_type,
             "speakers": list(obs.speakers), "involves": involves}
    draft = agent.mods.modify_memory(agent, draft)
    emb = ctx.embed(draft["text"])
    now = agent.scratch.curr_time

    # merge into a near-duplicate recent memory
    if noise > 0:
        recent = [n for n in agent.a_mem.all_nodes() if n.type == kind][:40]
        best, best_sim = None, 0.0
        for n in recent:
            s = cos(emb, agent.a_mem.embeddings[n.embedding_key])
            if s > best_sim:
                best, best_sim = n, s
        if best is not None and best_sim >= float(mc["merge_threshold"]):
            best.poignancy = max(best.poignancy, draft["importance"])
            best.last_accessed = now
            m = ctx.meta.get(best.node_id)
            if m:
                for e in obs.event_ids:
                    if e not in m.originating_event_ids:
                        m.originating_event_ids.append(e)
                m.source_ids.append(obs.id)
            ctx.tracer.log("memory_merged", agent=agent.id, node_id=best.node_id, into_text=best.description,
                           dropped_text=draft["text"], similarity=round(best_sim, 3), observation_id=obs.id,
                           originating_event_ids=list(obs.event_ids))
            return best

    s_name = agent.name
    o_name = obs.partner or (ctx.agents[involves[0]].name if involves and involves[0] != agent.id else obs.location)
    pred = {"perception": "saw", "conversation": "chat with", "overheard": "overheard"}.get(obs.source_type, "noticed")
    node = agent.a_mem.add(kind, now, s_name, pred, o_name, draft["text"],
                           _keywords(agent, draft["text"], involves), draft["importance"], emb, [])
    ctx.meta.set(node.node_id, MemoryMeta(agent_id=agent.id, source_type=obs.source_type, salience=salience,
                                          originating_event_ids=list(obs.event_ids),
                                          speakers=list(obs.speakers),
                                          source_ids=[obs.id] + list(obs.utterance_ids)))
    agent.scratch.importance_trigger_curr -= draft["importance"]
    agent.scratch.importance_ele_n += 1
    ctx.tracer.log("memory_encoded", agent=agent.id, node_id=node.node_id, kind=kind, text=node.description,
                   importance=node.poignancy, salience=salience, source_type=obs.source_type,
                   observation_id=obs.id, observation=raw, encoding_ops=ops, prompt=prompt,
                   originating_event_ids=list(obs.event_ids), speakers=list(obs.speakers),
                   utterance_ids=list(obs.utterance_ids))
    forget(agent)
    return node


def add_simple_event(agent, text: str, importance: int, involves: list[str], source_ids=()):
    """Low-importance perception of someone's routine activity (no LLM calls)."""
    ctx = agent.ctx
    emb = ctx.embed(text)
    o = ctx.agents[involves[0]].name if involves else agent.state.location
    node = agent.a_mem.add("event", agent.scratch.curr_time, agent.name, "saw", o, text,
                           _keywords(agent, text, involves), importance, emb, [])
    ctx.meta.set(node.node_id, MemoryMeta(agent_id=agent.id, source_type="perception", salience=0.1,
                                          speakers=[], source_ids=list(source_ids)))
    agent.scratch.importance_trigger_curr -= importance
    agent.scratch.importance_ele_n += 1
    ctx.tracer.log("memory_encoded", agent=agent.id, node_id=node.node_id, kind="event", text=text,
                   importance=importance, salience=0.1, source_type="ambient", observation_id=None,
                   observation=text, encoding_ops=[], prompt=None, originating_event_ids=[], speakers=[],
                   utterance_ids=[])
    forget(agent)
    return node


def forget(agent):
    cap = int(agent.cfg["memory"]["memory_capacity"])
    nodes = agent.a_mem.all_nodes()
    if len(nodes) <= cap:
        return
    now = agent.scratch.curr_time
    dr = float(agent.cfg["memory"]["decay_rate"])
    ranked = sorted(nodes, key=lambda n: ((n.poignancy / 10.0) * recency_score(n, now, dr), n.node_id))
    for n in ranked[: len(nodes) - cap]:
        agent.a_mem.remove(n.node_id)
        agent.ctx.tracer.log("memory_forgotten", agent=agent.id, node_id=n.node_id, text=n.description)
