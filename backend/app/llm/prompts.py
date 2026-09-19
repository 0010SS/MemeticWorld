"""Agent prompts.

RESEARCH RULE: nothing here may ask, hint, or show an agent how to invent language.
- No instructions like "invent slang", "make memes", "be viral", "nickname things",
  "develop running jokes", "create conventions".
- No example utterances. Any example line would get copied by every agent and then show up
  in the analysis as a fake meme. JSON formats describe fields in words only.
`tests/test_prompts.py` renders every prompt and fails on banned terms. Personality prose may
say someone is funny or serious; that is temperament, not an instruction to coin phrases.
"""
from __future__ import annotations

from collections.abc import Sequence

SYSTEM = """You are {name}, {role}.
{personality}
Things you care about: {interests}.

You are going about an ordinary day on campus. You only know what you can see and hear right now and what you remember.
When you say something, write exactly the words {name} would say out loud in that moment. Keep it short: one or two sentences."""

DECISION = """What do you do now? Your options:
- MOVE: walk somewhere. From here you can walk to: {neighbors}.
- TALK: say something to one person who is here with you{talk_hint}.
- REACT: say something out loud to nobody in particular, or react to what is happening.
- CONTINUE_ACTIVITY: keep doing what you are doing.
- IDLE: take a break and do nothing in particular.

Reply with only a JSON object with exactly these keys: "action", "target", "activity", "utterance", "reason", "importance".
- "action": one of MOVE, TALK, REACT, CONTINUE_ACTIVITY, IDLE
- "target": the place name for MOVE, the person's name for TALK, otherwise null
- "activity": a few words describing what you are doing after this choice
- "utterance": the exact words you say for TALK or REACT, otherwise null
- "reason": a short private reason for your choice
- "importance": an integer from 1 (mundane) to 10 (unforgettable) for how much this moment matters to you"""

REPLY = """{observation}

You are in a conversation with {partner} ({relationship}).
{memories}

Conversation so far:
{history}

Reply with only a JSON object with exactly these keys: "utterance", "end_conversation".
- "utterance": the exact words you say next to {partner}, or null if you say nothing
- "end_conversation": true if the conversation ends after this, otherwise false"""


def render_system(name: str, role: str, personality: str, interests: Sequence[str]) -> str:
    return SYSTEM.format(name=name, role=role, personality=personality, interests=", ".join(interests))


def describe_relationship(label: str | None, closeness: float) -> str:
    if closeness >= 0.7:
        feel = "you're close"
    elif closeness >= 0.45:
        feel = "you're friends"
    elif closeness >= 0.2:
        feel = "you know each other a bit"
    else:
        feel = "you barely know each other"
    return f"{label}; {feel}" if label else feel


def render_memories(memories: Sequence[tuple[str, str]]) -> str:
    """memories: (sim_time, text) pairs, oldest first."""
    if not memories:
        return "Relevant memories:\n- Nothing comes to mind."
    return "Relevant memories:\n" + "\n".join(f"- ({when}) {text}" for when, text in memories)


def render_observation(*, time_label: str, location: str, location_description: str, activity: str,
                       scheduled: tuple[str, str] | None, nearby: Sequence[tuple[str, str, str]],
                       happenings: Sequence[str], memories: Sequence[tuple[str, str]] | None) -> str:
    """nearby: (name, relationship description, visible activity). memories=None omits the section."""
    lines = [f"Time: {time_label}", f"Location: {location} ({location_description})",
             f"You are currently: {activity}"]
    if scheduled:
        lines.append(f"Your usual schedule has you at the {scheduled[0]} right now ({scheduled[1]}).")
    lines += ["", "Nearby:"]
    lines += [f"- {name} ({rel}) is {act}." for name, rel, act in nearby] or ["- Nobody else is here."]
    if happenings:
        lines += ["", "Happening here:"] + [f"- {h}" for h in happenings]
    if memories is not None:
        lines += ["", render_memories(memories)]
    return "\n".join(lines)


def render_decision(observation: str, neighbors: Sequence[str], nearby_names: Sequence[str]) -> str:
    talk_hint = f" ({', '.join(nearby_names)})" if nearby_names else " (nobody is here right now)"
    return observation + "\n\n" + DECISION.format(neighbors=", ".join(neighbors), talk_hint=talk_hint)


def render_reply(observation: str, partner: str, relationship: str, memories: Sequence[tuple[str, str]],
                 history: Sequence[tuple[str, str]]) -> str:
    """history: (speaker label, text); the agent's own lines should be labelled "You"."""
    lines = "\n".join(f'{who}: "{text}"' for who, text in history)
    return REPLY.format(observation=observation, partner=partner, relationship=relationship,
                        memories=render_memories(memories), history=lines)
