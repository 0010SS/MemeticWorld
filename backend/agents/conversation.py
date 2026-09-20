"""Conversation = upstream GA agent_chat_v2, shortened and instrumented.

Per conversation and participant: upstream relationship summary (once, not per
utterance as upstream), then for each utterance: stochastic retrieval with GA
focal points [relationship, partner activity, recent lines] and the upstream
iterative_convo prompt. Prompt inputs are limited to persona, observation,
relationship, retrieved memories (incl. private reflections) and the
conversation so far -- never latent labels, analyzer output or other agents'
memories.

`conversation.mealtalk` (off by default) adds one situational line and one retrieval focal point when two
people who still have a meal ahead of them meet inside a meal window: a REASON to refer to a place, never
a name for one. See MEALTALK_LINE below.

`conversation.repair` (off by default) lets the addressee stop the speaker over an expression they have
never heard ("the what?"). The mechanism lives in backend/agents/repair.py; this module owns only the
seam -- its turns are appended to this conversation's utterances and transcript, so they reach memory,
exposure and the observer through the ordinary path rather than a side channel.

Every utterance produces exposure records for the listeners who actually heard it.
"""
from __future__ import annotations

import math
import re

from backend.agents import ga_prompts
from backend.agents.agent import CampusMaze
from backend.agents.perception import AgentObservation
from backend.memory.encoder import encode
from backend.memory.reminding import maybe_remind
from backend.memory.retrieval import merged_nodes, retrieve
from backend.simulation.rngs import seed_rng

MAZE = CampusMaze()


def overhear_streams(participants: list, bystanders: list) -> dict:
    """One stream per bystander, seeded by (seed, "overhear", tick, participants, bystander): whether someone
    overhears a line never depends on the conversation's retrieval draws, on who else stands nearby, or on
    the order in which this tick's conversations were formed. One draw per utterance."""
    first = participants[0]
    who = sorted(a.id for a in participants)
    return {b.id: seed_rng(first.seed, "overhear", first.ctx.tracer.tick, *who, b.id) for b in bystanders}


def overhear_prob(bystander, pc: dict) -> float:
    """overhear_prob, x busy_factor for a bystander who is in another conversation in the same room."""
    return pc["overhear_prob"] * (pc["busy_factor"] if bystander.state.in_conversation else 1.0)


def talk_gate(agent, target, now, rng) -> float:
    cc = agent.cfg["conversation"]
    if agent.state.talks_today >= cc["max_per_agent_per_day"]:
        return 0.0
    last = agent.state.last_talk.get(target.id)
    if last is not None and (now - last).total_seconds() / 60 < cc["pair_cooldown_minutes"]:
        return 0.0
    if agent.state.activity == "sleeping" or target.state.activity == "sleeping":
        return 0.0
    traits = set(agent.profile.personality.get("traits", []))
    soc = 1.0 + 0.35 * ("outgoing" in traits) - 0.3 * ("reserved" in traits)
    fam = agent.profile.rel(target.id).familiarity
    p = cc["base_talk_prob"] * (0.35 + fam) * soc
    return agent.mods.modify_utility(agent, "talk_prob", p, target=target)


def decide_to_talk(agent, target, rng) -> tuple[bool, dict]:
    res = retrieve(agent, [target.name], rng=rng)
    nodes = merged_nodes(res)
    evs = [n for n in nodes if n.type != "thought"]
    ths = [n for n in nodes if n.type == "thought"]
    yes, prompt = ga_prompts.decide_to_talk(agent, target, evs, ths)
    agent.ctx.tracer.log("decision", agent=agent.id, kind="decide_to_talk", target=target.id, prompt=prompt,
                         decision={"action": "TALK" if yes else "CONTINUE", "target": target.id},
                         retrieved=[n.node_id for n in nodes])
    return yes, {"retrieved": [n.node_id for n in nodes]}


CATCHUP_FOCAL = "what has happened lately that stood out"

# ------------------------------------------------------------------ meal arrangement (conversation.mealtalk)
# A REASON TO REFER, not a name. Two people who have not eaten yet and meet inside a meal window have to
# settle where they are going -- the everyday moment at which a place gets referred to at all. The world
# supplies the occasion; which words the agents reach for (the label they were seeded with, one they heard
# from someone else, or a description) is entirely theirs. Nothing here names a place, and nothing here
# asks anyone to invent or adopt a name.
MEALTALK_LINE = "{a} and {b} have not eaten yet and are working out where to eat."
# Retrieval focal point: pulls up what the speaker remembers about arranging and having meals. A seeded
# memory about where the agent eats ("... I eat most days at the dining hall beside the freshman residences
# ... I go to the dining hall by the graduate flats instead") is one such memory and competes on relevance
# like any other -- it is not privileged, only reachable. No canonical place name appears in either: the
# seed files say what the world says (R3/D75), so the memory and the percept name the same referent.
MEALTALK_FOCAL = "arranging a meal: where to eat, and meals they have had before"
# Routine wording for a meal, matched against the plan the WORLD drew (never against a place name).
MEAL_RE = re.compile(r"\b(?:lunch|dinner|breakfast|brunch|supper|meals?|eat|eating|ate|dining)\b", re.I)


def _mins(v) -> int:
    h, m = map(int, str(v).split(":"))
    return h * 60 + m


def meal_window(cfg: dict, now) -> str | None:
    """The configured `conversation.mealtalk.windows` entry `now` falls in (inclusive), else None."""
    mins = now.hour * 60 + now.minute
    for w in ((cfg.get("conversation") or {}).get("mealtalk") or {}).get("windows") or []:
        lo, hi = (_mins(x) for x in str(w).split("-"))
        if lo <= mins <= hi:
            return str(w)
    return None


def next_step_is_meal(agent) -> bool:
    """True when the agent's NEXT routine step is a meal, i.e. they still have to go and eat.

    Read from the day plan the world drew, so it is a fact about the agent's schedule, not about any
    place: an agent already at the table has decided, and gets no cue.
    """
    clock = getattr(getattr(agent, "ctx", None), "clock", None)
    plan = getattr(agent, "day_plan", None)
    if clock is None or not plan:
        return False
    from backend.simulation.scheduler import next_entry
    nxt = next_entry(agent, agent.ctx.tracer.tick % clock.ticks_per_day)
    return bool(nxt and MEAL_RE.search(nxt["activity"]))


def mealtalk(conv_id: str, participants: list) -> dict | None:
    """Decide ONCE per conversation whether it carries the meal-arrangement cue, and trace the decision.

    Fires when the conversation starts inside a meal window and at least one participant still has a meal
    ahead of them, with probability `conversation.mealtalk.prob`. The draw comes from its own seeded
    stream (seed, "mealtalk", tick, participant ids), so switching the cue on never shifts the retrieval,
    overhearing or talk draws (common random numbers across conditions, ontology v2 §2.1).
    """
    first = participants[0]
    mc = ((first.cfg.get("conversation") or {}).get("mealtalk") or {})
    if not mc.get("enabled"):
        return None
    window = meal_window(first.cfg, first.scratch.curr_time)
    if window is None or not any(next_step_is_meal(a) for a in participants):
        return None
    who = sorted(a.id for a in participants)
    if seed_rng(first.seed, "mealtalk", first.ctx.tracer.tick, *who).random() >= float(mc.get("prob", 0.6)):
        return None
    first.ctx.tracer.log("mealtalk", conversation_id=conv_id, participants=who, window=window)
    return {"focal": MEALTALK_FOCAL}


def _mind(speaker, conv_id: str | None) -> dict:
    """v2 WORDING/NEED hooks (off by default), computed ONCE per speaker per conversation from the speaker's
    OWN memory only: recently heard wordings (priming) and open matters (need). A speaker is in one
    conversation at a time, so a per-agent cache keyed by conversation id is thread-safe."""
    cached = getattr(speaker, "_mind_cache", None)
    if conv_id is not None and cached and cached[0] == conv_id:
        return cached[1]
    cfg = speaker.cfg
    out = {"lines": [], "focal": []}
    if (cfg.get("need") or {}).get("enabled"):
        from backend.memory import need as NEED
        om = NEED.focal(speaker, speaker.scratch.curr_time, conversation_id=conv_id)
        out["focal"] = list(om)
        if om:
            out["lines"].append(f"{speaker.profile.first_name} still has on their mind: {om[0]}")
    if (cfg.get("priming") or {}).get("enabled"):
        from backend.memory import wording as WORDING
        line = WORDING.priming_line(speaker, speaker.stream("prime"), conversation_id=conv_id)
        if line:
            out["lines"].append(line)
    speaker._mind_cache = (conv_id, out)
    return out


def _mind_lines(speaker, conv_id: str | None = None) -> list[str]:
    return _mind(speaker, conv_id)["lines"]


def _need_focal(speaker, conv_id: str | None = None) -> list[str]:
    return _mind(speaker, conv_id)["focal"]


def _context(speaker, other, started_by_speaker: bool, trigger_text: str | None, topic: str | None = None,
             conv_id: str | None = None, meal: dict | None = None) -> str:
    s, o = speaker.scratch, other.scratch
    s_act = speaker.state.pre_chat_activity or s.act_description
    o_act = other.state.pre_chat_activity or o.act_description
    if started_by_speaker:
        ctx = (f"{s.name} was {s_act} when {s.name} saw {o.name} in the middle of "
               f"{o_act}.\n{s.name} is initiating a conversation with {o.name}.")
    else:
        ctx = (f"{s.name} was {s_act} when {o.name}, who was {o_act}, "
               f"started a conversation with {s.name}.")
    lines = [speaker.relationship_line(other)]
    if topic == "catchup":
        lines.append(f"{s.name} and {o.name} are catching up on how things have been going lately.")
    elif topic in ("clarify", "handover"):
        from backend.agents import coop_talk
        tl = coop_talk.speaker_topic_line(topic, speaker, other, started_by_speaker)
        if tl:
            lines.append(tl)
    if meal:
        lines.append(MEALTALK_LINE.format(a=s.name, b=o.name))
    lines += _mind_lines(speaker, conv_id)
    if trigger_text:
        lines.append(f"{s.name} just noticed: {trigger_text}")
    lines = speaker.mods.modify_prompt(speaker, "chat", lines, target=other)
    return ctx + "\n" + "\n".join(lines)


def run_conversation(conv_id: str, init, target, bystanders: list, rng, *, opening: str | None = None,
                     trigger: dict | None = None) -> dict:
    """Run a 1..max_utterances conversation. Returns a conversation record.

    trigger: {"agent": id, "text": observation text} -- private context of whoever noticed something.
    Participants' memories are encoded here; overhearer observations are returned for the
    engine to encode afterwards (they may overhear several conversations).
    """
    ctx = init.ctx
    cc = init.cfg["conversation"]
    pc = init.cfg["perception"]
    tracer = ctx.tracer
    parts = [init, target]
    rel = {}
    for a, b in ((init, target), (target, init)):
        r = retrieve(a, [b.name], rng=rng)
        rel[a.id] = ga_prompts.relationship_summary(a, b, {b.name: r[b.name].nodes})
    chat: list[list[str]] = []
    utterances = []
    heard_by: dict[str, list[int]] = {b.id: [] for b in bystanders}
    orng = overhear_streams(parts, bystanders)
    k = int(init.cfg["retrieval"]["top_k"])
    topic = (trigger or {}).get("topic")
    meal = mealtalk(conv_id, parts)
    n_max = int(cc.get("catchup_max_utterances", cc["max_utterances"])) if topic == "catchup" else int(cc["max_utterances"])
    if topic in ("clarify", "handover"):                  # v3 co-op talk budgets (§4.5)
        from backend.agents import coop_talk
        n_max = int(coop_talk.max_utterances(init.cfg, topic) or n_max)
    for i in range(n_max):
        speaker, other = parts[i % 2], parts[(i + 1) % 2]
        trig = trigger["text"] if trigger and trigger["agent"] == speaker.id and i < 2 else None
        last = "".join(": ".join(x) + "\n" for x in chat[-4:])
        focal = [rel[speaker.id], f"{other.name} is {other.state.pre_chat_activity or other.scratch.act_description}"]
        if last:
            focal.append(last)
        if trig:
            focal.append(trig)
        if topic == "catchup":
            focal.append(CATCHUP_FOCAL)
        if meal:
            focal.append(meal["focal"])
        focal += _need_focal(speaker, conv_id)
        per = max(1, math.ceil(k / len(focal)))
        res = retrieve(speaker, focal, k=per, rng=rng)
        nodes = merged_nodes(res, limit=k + 1)
        retrieved = {"memories": nodes}
        curr_context = _context(speaker, other, speaker is init, trig, topic, conv_id, meal)
        if i == 0 and opening:
            text, end = opening, False
            source = "reaction_opening"
        else:
            text, end = ga_prompts.chat_utterance(MAZE, speaker, other, retrieved, curr_context, chat)
            source = "chat_utterance"
        if text is None:
            break
        uid = f"{conv_id}.u{i}"
        listeners = [other.id]
        # WHERE this line lands in `utterances`, not the loop index: with `conversation.repair` on, a line
        # can be followed by repair turns, so the two stop agreeing after the first repair and a bystander
        # would be handed somebody else's words.
        pos = len(utterances)
        for b in bystanders:
            if orng[b.id].random() < overhear_prob(b, pc):
                listeners.append(b.id)
                heard_by[b.id].append(pos)
        ev_ids = ctx.meta.events_of([n.node_id for n in nodes])
        if (speaker.cfg.get("need") or {}).get("enabled"):
            from backend.memory import need as NEED
            NEED.discussed(speaker, [n.node_id for n in nodes])
        u = {"id": uid, "conversation_id": conv_id, "idx": i, "speaker": speaker.id, "text": text,
             "listeners": listeners, "location": speaker.state.location, "arena": speaker.state.arena,
             "retrieved": [n.node_id for n in nodes], "retrieved_event_ids": ev_ids, "source": source}
        utterances.append(u)
        tracer.log("utterance", **u, context=curr_context,
                   retrieval_scores={nid: sc for r in res.values() for nid, sc in r.scores.items()},
                   relationship_summary=rel[speaker.id])
        tracer.log("exposure", utterance_id=uid, speaker_id=speaker.id, listener_ids=listeners, utterance=text,
                   conversation_id=conv_id, location=speaker.state.location, arena=speaker.state.arena)
        chat.append([speaker.name, text])
        # MEMES: the addressee may stop the speaker over something they just said (backend/agents/repair.py,
        # `conversation.repair`, off by default). The turns are appended here rather than logged there, so
        # they go through the ordinary path: they end up in the transcript the next speaker sees, and in the
        # whole-conversation memory both participants encode below. A repair reaches exactly the ears that
        # heard the line it is about and takes no overhearing draw of its own, so the bystanders who heard
        # that line are handed the turns directly instead of being rolled for again.
        if ((speaker.cfg.get("conversation") or {}).get("repair") or {}).get("enabled"):
            from backend.agents import repair as REPAIR
            for r in REPAIR.maybe_repair(conv_id, u, speaker, other, utterances, chat):
                for b in bystanders:
                    if b.id in u["listeners"]:
                        heard_by[b.id].append(len(utterances))
                utterances.append(r)
                chat.append([ctx.agents[r["speaker"]].name, r["text"]])
        if end and i >= 1:
            break
    conv = {"id": conv_id, "participants": [init.id, target.id], "utterances": utterances,
            "location": init.state.location, "arena": init.state.arena,
            "trigger": trigger, "relationship_summaries": rel}
    ctx.mods.on_conversation(conv, ctx.agents)
    tracer.log("conversation", **{k2: v for k2, v in conv.items() if k2 != "utterances"},
               utterance_ids=[u["id"] for u in utterances], transcript=[[ctx.agents[u["speaker"]].name, u["text"]]
                                                                        for u in utterances])
    # participants encode a lossy memory of the whole exchange
    ev_all = []
    for u in utterances:
        for e in u["retrieved_event_ids"]:
            if e not in ev_all:
                ev_all.append(e)
    if trigger:
        for e in trigger.get("event_ids", []):
            if e not in ev_all:
                ev_all.append(e)
    for a, b in ((init, target), (target, init)):
        if not utterances:
            break
        facts = [{"id": u["id"], "text": f'{ctx.agents[u["speaker"]].profile.first_name}: "{u["text"]}"',
                  "salience": 0.6, "involves": [u["speaker"]] if u["speaker"] != a.id else [], "speaker": u["speaker"]}
                 for u in utterances]
        obs = AgentObservation(id=f"{conv_id}.obs.{a.id}", agent_id=a.id, tick=tracer.tick,
                               location=a.state.location, arena=a.state.arena, source_type="conversation",
                               facts=facts, event_ids=ev_all, speakers=[b.id],
                               utterance_ids=[u["id"] for u in utterances], partner=b.name)
        tracer.log("observation", agent=a.id, observation_id=obs.id, source_type="conversation",
                   facts=facts, originating_event_ids=ev_all)
        node = encode(a, obs, a.stream("encode"))
        maybe_remind(a, obs, node, a.stream("remind"))   # LINK after conversations too (v2 §2.3; config-gated)
        if (a.cfg.get("need") or {}).get("enabled"):
            from backend.memory import need as NEED
            NEED.note(a, obs, node)
        a.state.last_talk[b.id] = a.scratch.curr_time
        a.state.talks_today += 1
    overheard = []
    for b in bystanders:
        idx = heard_by[b.id]
        if not idx:
            continue
        us = [utterances[j] for j in idx if j < len(utterances)]
        facts = [{"id": u["id"], "text": f'{ctx.agents[u["speaker"]].profile.first_name}: "{u["text"]}"',
                  "salience": 0.4, "involves": [u["speaker"]], "speaker": u["speaker"]} for u in us]
        overheard.append(AgentObservation(
            id=f"{conv_id}.obs.{b.id}", agent_id=b.id, tick=tracer.tick, location=b.state.location,
            arena=b.state.arena, source_type="overheard", facts=facts, event_ids=ev_all,
            speakers=sorted({u["speaker"] for u in us}), utterance_ids=[u["id"] for u in us],
            partner=ctx.agents[us[0]["speaker"]].name))
    conv["overheard"] = overheard
    return conv
