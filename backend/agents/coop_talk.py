"""Scheduled and repair talk at the co-op (ontology v3 §4.4-4.5): clarify, handover, meeting.

This module does not run conversations itself; it gives the engine / CoopWorld the pieces to run them through
v2's ``run_conversation`` and ``run_group_conversation``:

* forced-talk tuples in the engine's ``forced_talks`` format ``(init_id, target_id, opening, trigger)``;
* context lines per topic (the only topic text agents ever see; no agenda, nothing about what to say);
* per-topic utterance budgets from ``comm.*``;
* trace summaries (``clarification``, ``handover``, ``meeting``).

Nothing here names a cause, class, regime, job id or arm. Job ids only travel in ``trigger["event_ids"]``
(SIM-ONLY provenance: the conversation memory links to the job event, as a REACT TALK links to its event).
"""
from __future__ import annotations

CLARIFY_LINE = "{s} is running a job on the laser cutter and asks {o} something."
HANDOVER_LINE = "{s} and {o} are at the laser cutter at the shift change."
MEETING_LINE = "The co-op's members are meeting at the Makerspace to go over how the week has been going."

DYAD_TOPIC_LINES = {"clarify": CLARIFY_LINE, "handover": HANDOVER_LINE}
GROUP_TOPIC_LINES = {"meeting": MEETING_LINE}      # group_conversation.TOPIC_LINES format ({names} optional)

DEFAULTS = {"clarify": {"enabled": False, "max_per_job": 1, "max_utterances": 4},
            "handover": {"enabled": False, "time": "13:15", "max_utterances": 6},
            "meeting": {"enabled": False, "days": [2, 5], "time": "18:30", "max_utterances": 10,
                        "max_participants": 8}}


def comm_cfg(cfg: dict, topic: str) -> dict:
    """comm.<topic> with the contract defaults (ontology v3 §8.2) filled in."""
    return {**DEFAULTS.get(topic, {}), **((cfg.get("comm") or {}).get(topic) or {})}


def enabled(cfg: dict, topic: str) -> bool:
    return bool(comm_cfg(cfg, topic).get("enabled"))


def max_utterances(cfg: dict, topic: str | None) -> int | None:
    """Utterance budget for a co-op topic, or None for any other topic (use v2's budget)."""
    if topic not in DEFAULTS:
        return None
    return int(comm_cfg(cfg, topic)["max_utterances"])


def dyad_topic_line(topic: str | None, init, target) -> str | None:
    """The context line for a dyadic co-op talk, fixed from the initiator's side (s = initiator, e.g. the
    operator who asks), identical for both speakers. None for other topics."""
    tpl = DYAD_TOPIC_LINES.get(topic or "")
    return tpl.format(s=init.name, o=target.name) if tpl else None


def speaker_topic_line(topic: str | None, speaker, other, started_by_speaker: bool) -> str | None:
    """Drop-in for conversation._context(speaker, other, started_by_speaker, ...)."""
    init, tgt = (speaker, other) if started_by_speaker else (other, speaker)
    return dyad_topic_line(topic, init, tgt)


def register_group_topics(topic_lines: dict | None = None) -> dict:
    """Add the meeting line to group_conversation.TOPIC_LINES (or to the dict given). Idempotent."""
    if topic_lines is None:
        from backend.agents import group_conversation
        topic_lines = group_conversation.TOPIC_LINES
    for k, v in GROUP_TOPIC_LINES.items():
        topic_lines.setdefault(k, v)
    return topic_lines


# --------------------------------------------------------------------------- forced talks
def clarify_talk(operator, target, question: str, job_id: str) -> tuple:
    """Choice G (§4.4): the operator's question opens a v2 dyadic conversation in phase 6 of the same tick."""
    return (operator.id, target.id, question,
            {"agent": operator.id, "text": None, "topic": "clarify", "event_ids": [job_id], "job": job_id})


def handover_pair(jobs: list[dict]) -> tuple[str, str] | None:
    """(outgoing, incoming): the operator of the day's last AM job (12:00) and of the first PM job (13:30)."""
    am = sorted((j for j in jobs if j.get("kind", "job") == "job" and j.get("shift") == "am"),
                key=lambda j: j.get("start_tick", j.get("slot", 0)))
    pm = sorted((j for j in jobs if j.get("kind", "job") == "job" and j.get("shift") == "pm"),
                key=lambda j: j.get("start_tick", j.get("slot", 0)))
    if not am or not pm:
        return None
    a, b = am[-1].get("operator"), pm[0].get("operator")
    if not a or not b or a == b:
        return None
    return a, b


def handover_talk(outgoing, incoming) -> tuple:
    """Handover (§4.5): the outgoing operator opens; no opening line is imposed."""
    return (outgoing.id, incoming.id, None, {"agent": None, "text": None, "topic": "handover"})


def meeting_participants(cfg: dict, active_ids) -> list[str]:
    """Every active member (sorted ids), capped at comm.meeting.max_participants."""
    return sorted(active_ids)[: int(comm_cfg(cfg, "meeting")["max_participants"])]


def is_meeting_day(cfg: dict, day: int) -> bool:
    mc = comm_cfg(cfg, "meeting")
    return bool(mc.get("enabled")) and day in {int(d) for d in mc.get("days") or []}


def _hhmm(t) -> str:
    return t if isinstance(t, str) else t.strftime("%H:%M")


def is_time(cfg: dict, topic: str, now) -> bool:
    """Whether ``now`` (datetime or "HH:MM") is the configured start of a scheduled topic."""
    return _hhmm(now) == str(comm_cfg(cfg, topic).get("time"))


# --------------------------------------------------------------------------- trace summaries
def _conv_ids(conv: dict) -> dict:
    return {"conversation_id": conv.get("id"), "participants": list(conv.get("participants") or []),
            "utterance_ids": [u["id"] for u in conv.get("utterances") or []],
            "n_utterances": len(conv.get("utterances") or [])}


def clarification_record(job_id: str, attempt: int, operator_id: str, target_id: str, question: str,
                         conv: dict | None = None) -> dict:
    """Fields for ``tracer.log("clarification", **rec)``."""
    rec = {"job": job_id, "attempt": attempt, "agent": operator_id, "target": target_id, "question": question}
    if conv is not None:
        rec.update(_conv_ids(conv))
    return rec


def handover_record(day: int, outgoing: str, incoming: str, conv: dict | None = None) -> dict:
    rec = {"day": day, "outgoing": outgoing, "incoming": incoming}
    if conv is not None:
        rec.update(_conv_ids(conv))
    return rec


def meeting_record(day: int, participants: list, conv: dict | None = None) -> dict:
    rec = {"day": day, "invited": list(participants)}
    if conv is not None:
        rec.update(_conv_ids(conv))
    return rec
