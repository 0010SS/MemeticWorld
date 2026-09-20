"""Multi-party talk (ontology v2 §3): three or more people at one table.

GA only has dyadic chat (agent_chat_v2). This generalises MemeWorld's version of it
(conversation.run_conversation) with as few changes as possible:
  * speakers take turns round-robin from a random start, up to conversation.group.max_utterances;
    the model may end the conversation once everyone has had a turn;
  * each utterance gets fresh stochastic retrieval with GA-style focal points [relationships,
    what the others are doing, recent lines] and the adapted iterative_convo prompt
    (backend/prompts/group_chat_v1.txt), in which the speaker talks to the whole table;
  * relationship context is the plain relationship statements, not GA's LLM relationship summary
    (that would be N(N-1) extra calls per conversation).
Records and traces have exactly the dyadic shape (a group is simply len(participants) >= 3):
listeners are all other participants plus the bystanders who overheard, every participant encodes one
lossy memory of the exchange, and overheard observations are returned for the engine to encode.
"""
from __future__ import annotations

import math

from backend import ga_compat
from backend.agents.conversation import CATCHUP_FOCAL, _mind_lines, _need_focal, overhear_prob, overhear_streams
from backend.agents.ga_prompts import as_json
from backend.agents.perception import AgentObservation
from backend.llm.client import llm_purpose
from backend.memory.encoder import encode
from backend.memory.reminding import maybe_remind
from backend.memory.retrieval import merged_nodes, retrieve

ga = ga_compat.load()
PROMPT = str(ga_compat.REPO_ROOT / "backend" / "prompts" / "group_chat_v1.txt")
TOPIC_LINES = {"meal": "{names} are sitting at the same table over a meal.",
               "catchup": "{names} are catching up on how things have been going lately."}
DEFAULT_TOPIC_LINE = "{names} are together."
PAST_MINUTES = 480   # GA: an earlier conversation counts as past context for 8 hours


def _names(agents) -> str:
    names = [a.name for a in agents]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _activity(agent) -> str:
    return agent.state.pre_chat_activity or agent.scratch.act_description


def _context(speaker, others: list, topic: str | None, conv_id: str | None = None) -> str:
    # D75: the world describes the place, it never names it. Canonical mode renders exactly the string
    # this line used to build ("the Dining Hall"); situated mode renders a description.
    from backend.simulation import reference
    where = reference.phrase(speaker.state.location, agent=speaker, cfg=speaker.cfg)
    lines = [f"{speaker.name} was {_activity(speaker)} at {where}.",
             TOPIC_LINES.get(topic, DEFAULT_TOPIC_LINE).format(names=_names([speaker, *others]))]
    lines += [speaker.relationship_line(o) for o in others]
    lines += _mind_lines(speaker, conv_id)
    lines = speaker.mods.modify_prompt(speaker, "chat", lines, target=None, others=others)
    return "\n".join(lines)


def _past_context(speaker, others: list) -> str:
    """GA's 'Past Context': the speaker's latest conversation memory with any of these people, if recent."""
    names = {o.name for o in others}
    now = speaker.scratch.curr_time
    for n in speaker.a_mem.seq_chat:                     # newest first
        if names & set(str(n.object).split(", ")):
            mins = int((now - n.created).total_seconds() / 60)
            if mins > PAST_MINUTES:
                return ""
            return (f"{mins} minutes ago, {speaker.name} and {n.object} were already {n.description} "
                    f"This context takes place after that conversation.")
    return ""


def _flag(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "yes", "1")


def group_utterance(speaker, prompt: str) -> tuple[str | None, bool]:
    """One turn: GA's ChatGPT_safe_generate_response_OLD retry wrapper (as upstream iterative_convo) with a
    parser for {"utterance", "end"}; like GA's clean-up it also accepts the first two values in order."""
    def parse(r, prompt=""):
        d = as_json(r)
        if not isinstance(d, dict) or not d:
            raise ValueError(r)
        vals = list(d.values())
        return {"utterance": d.get("utterance", vals[0]),
                "end": _flag(d.get("end", vals[1] if len(vals) > 1 else False))}

    def valid(r, prompt=""):
        try:
            parse(r)
            return True
        except Exception:  # noqa: BLE001
            return False

    with llm_purpose("group_chat_utterance", speaker.id):
        out = ga.gs.ChatGPT_safe_generate_response_OLD(prompt, 3, {"utterance": "...", "end": True}, valid, parse)
    utt = str(out["utterance"]).strip().strip('"').strip()
    if not utt or utt == "..." or utt.startswith("LLM_ERROR"):
        return None, True
    return utt, bool(out["end"])


def run_group_conversation(conv_id: str, participants: list, bystanders: list, rng, *, topic: str = "meal") -> dict:
    """Run a multi-party conversation; returns a record with the keys of run_conversation's."""
    parts = list(participants)
    first = parts[0]
    ctx, cfg = first.ctx, first.cfg
    gc = cfg["conversation"].get("group") or {}
    pc = cfg["perception"]
    tracer = ctx.tracer
    others_of = {a.id: [b for b in parts if b is not a] for a in parts}
    rel = {a.id: "\n".join(a.relationship_line(b) for b in others_of[a.id]) for a in parts}
    chat: list[list[str]] = []
    utterances = []
    heard_by: dict[str, list[int]] = {b.id: [] for b in bystanders}
    orng = overhear_streams(parts, bystanders)
    k = int(cfg["retrieval"]["top_k"])
    # D75: the group chat prompt's "<arena> in <sector>" slot. reference.inside reproduces this exact
    # string in canonical mode and describes the room and the building in situated mode.
    from backend.simulation import reference
    where = reference.inside(first.state.location, first.state.arena, agent=first, cfg=cfg)
    everyone = _names(parts)
    start = int(rng.integers(len(parts)))
    for i in range(int(gc.get("max_utterances", 8))):
        speaker = parts[(start + i) % len(parts)]
        others = others_of[speaker.id]
        last = "".join(": ".join(x) + "\n" for x in chat[-4:])
        focal = [rel[speaker.id], "; ".join(f"{o.name} is {_activity(o)}" for o in others)]
        if last:
            focal.append(last)
        if topic == "catchup":
            focal.append(CATCHUP_FOCAL)
        focal += _need_focal(speaker, conv_id)
        per = max(1, math.ceil(k / len(focal)))
        res = retrieve(speaker, focal, k=per, rng=rng)
        nodes = merged_nodes(res, limit=k + 1)
        curr_context = _context(speaker, others, topic, conv_id)
        convo = "".join(": ".join(x) + "\n" for x in chat) or "[The conversation has not started yet -- start it!]"
        prompt = ga.gs.generate_prompt(
            [f"Here is a brief description of {speaker.name}.\n{speaker.iss()}", speaker.name,
             "".join(f"- {n.description}\n" for n in nodes), _past_context(speaker, others), where,
             curr_context, everyone, convo, speaker.name], PROMPT)
        text, end = group_utterance(speaker, prompt)
        if text is None:
            break
        uid = f"{conv_id}.u{i}"
        listeners = [o.id for o in others]
        for b in bystanders:
            if orng[b.id].random() < overhear_prob(b, pc):
                listeners.append(b.id)
                heard_by[b.id].append(i)
        ev_ids = ctx.meta.events_of([n.node_id for n in nodes])
        if (cfg.get("need") or {}).get("enabled"):
            from backend.memory import need as NEED
            NEED.discussed(speaker, [n.node_id for n in nodes])
        u = {"id": uid, "conversation_id": conv_id, "idx": i, "speaker": speaker.id, "text": text,
             "listeners": listeners, "location": speaker.state.location, "arena": speaker.state.arena,
             "retrieved": [n.node_id for n in nodes], "retrieved_event_ids": ev_ids, "source": "group_chat_utterance"}
        utterances.append(u)
        tracer.log("utterance", **u, context=curr_context,
                   retrieval_scores={nid: sc for r in res.values() for nid, sc in r.scores.items()},
                   relationship_summary=rel[speaker.id])
        tracer.log("exposure", utterance_id=uid, speaker_id=speaker.id, listener_ids=listeners, utterance=text,
                   conversation_id=conv_id, location=speaker.state.location, arena=speaker.state.arena)
        chat.append([speaker.name, text])
        if end and i >= len(parts) - 1:
            break
    conv = {"id": conv_id, "participants": [a.id for a in parts], "utterances": utterances,
            "location": first.state.location, "arena": first.state.arena,
            "trigger": {"agent": None, "text": None, "topic": topic}, "relationship_summaries": rel}
    ctx.mods.on_conversation(conv, ctx.agents)
    tracer.log("conversation", **{k2: v for k2, v in conv.items() if k2 != "utterances"},
               utterance_ids=[u["id"] for u in utterances], transcript=[[ctx.agents[u["speaker"]].name, u["text"]]
                                                                        for u in utterances])
    ev_all = []
    for u in utterances:
        for e in u["retrieved_event_ids"]:
            if e not in ev_all:
                ev_all.append(e)
    for a in parts:
        if not utterances:
            break
        others = others_of[a.id]
        facts = [{"id": u["id"], "text": f'{ctx.agents[u["speaker"]].profile.first_name}: "{u["text"]}"',
                  "salience": 0.6, "involves": [u["speaker"]] if u["speaker"] != a.id else [], "speaker": u["speaker"]}
                 for u in utterances]
        obs = AgentObservation(id=f"{conv_id}.obs.{a.id}", agent_id=a.id, tick=tracer.tick,
                               location=a.state.location, arena=a.state.arena, source_type="conversation",
                               facts=facts, event_ids=ev_all, speakers=[o.id for o in others],
                               utterance_ids=[u["id"] for u in utterances], partner=", ".join(o.name for o in others))
        tracer.log("observation", agent=a.id, observation_id=obs.id, source_type="conversation",
                   facts=facts, originating_event_ids=ev_all)
        node = encode(a, obs, a.stream("encode"))
        maybe_remind(a, obs, node, a.stream("remind"))   # LINK after conversations too (v2 §2.3; config-gated)
        if (cfg.get("need") or {}).get("enabled"):       # NEED: as in run_conversation (D60 has no exception)
            from backend.memory import need as NEED
            NEED.note(a, obs, node)
        for o in others:
            a.state.last_talk[o.id] = a.scratch.curr_time
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
